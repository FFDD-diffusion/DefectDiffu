import os
import torch
torch.cuda.is_available()
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from models_add_cross_concate import DiT
from diffusion import create_diffusion
from autoencoder import *
import clip.clip as clip
from diffusers.models import AutoencoderKL
import argparse

def rgb_to_gray(tensor):

    r, g, b = tensor[:, 0], tensor[:, 1], tensor[:, 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return gray

def iterative_thresholding_batch(gray_tensor):

    gray_np = gray_tensor.detach().cpu().numpy()
    binarized = np.zeros_like(gray_np, dtype=np.uint8)

    for i in range(gray_np.shape[0]):
        img = gray_np[i]
        T = img.mean()
        prev_T = -1

        while abs(T - prev_T) > 1e-4:
            prev_T = T
            G1 = img[img >= T]
            G2 = img[img < T]
            m1 = G1.mean() if G1.size > 0 else 0
            m2 = G2.mean() if G2.size > 0 else 0
            T = (m1 + m2) / 2

        binarized[i] = (img >= T).astype(np.uint8)

    return torch.from_numpy(binarized).to(gray_tensor.device)

def binarize_tensor_iterative(x):

    gray = rgb_to_gray(x)
    binary = iterative_thresholding_batch(gray)
    return binary.unsqueeze(1)


def get_label(data_path):
    label_list1 = []
    for name_class in os.listdir(data_path):
        for class_object in os.listdir(data_path + '/' + name_class + '/img'):
            if class_object!='good':
                label_list1.append(class_object + ' ' + name_class)
    return label_list1


def gen(args):
    data_path = args.data
    label_list = get_label(data_path)

    image_size = args.imagesize
    device = "cuda"
    latent_size = image_size // 8

    model_clip, preprocess_clip = clip.load('RN50', device)

    model = DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, input_size=latent_size, num_classes=1000).to(device)

    diffusion = create_diffusion(timestep_respacing="50")
    vae = AutoencoderKL.from_pretrained(args.vae).to(device)
    state_dict = torch.load(args.ckpt)
    model.load_state_dict(state_dict['model_state_dict'])
    num_img = args.batchsize
    while True:
        for c in label_list:

            y_null_product = torch.cat([clip.tokenize(f"a photo of good industry")] * num_img).to(device)
            with torch.no_grad():
                y_null_product = model_clip.encode_text(y_null_product)
            y_null_product /= y_null_product.norm(dim=-1, keepdim=True)
            y_null_product = y_null_product.float()
            y_null_product = y_null_product.to(device)

            y_null_good = torch.cat([clip.tokenize(f"a photo of good {c.split()[1]}")] * num_img).to(device)
            with torch.no_grad():
                y_null_good = model_clip.encode_text(y_null_good)
            y_null_good /= y_null_good.norm(dim=-1, keepdim=True)
            y_null_good = y_null_good.float()
            y_null_good = y_null_good.to(device)

            only_good = torch.cat([clip.tokenize(f"a photo of good")] * num_img).to(device)
            defect = torch.cat([clip.tokenize(f"a photo of {c.split()[0]}")] * num_img).to(device)
            classes = torch.cat([clip.tokenize(f"a photo of {c.split()[1]}")] * num_img).to(device)
            classes_industry = torch.cat([clip.tokenize(f"a photo of industry")] * num_img).to(device)
            y_all = torch.cat([clip.tokenize(f"a photo of {c}")] * num_img).to(device)

            with torch.no_grad():
                only_good = model_clip.encode_text(only_good)
                defect = model_clip.encode_text(defect)
                classes = model_clip.encode_text(classes)
                classes_industry = model_clip.encode_text(classes_industry)
                y_all = model_clip.encode_text(y_all)

            only_good /= only_good.norm(dim=-1, keepdim=True)
            only_good = only_good.float()
            only_good = only_good.to(device)

            defect /= defect.norm(dim=-1, keepdim=True)
            defect = defect.float()
            defect = defect.to(device)

            classes_industry /= classes_industry.norm(dim=-1, keepdim=True)
            classes_industry = classes_industry.float()
            classes_industry = classes_industry.to(device)

            classes /= classes.norm(dim=-1, keepdim=True)
            classes = classes.float()
            classes = classes.to(device)

            y_all /= y_all.norm(dim=-1, keepdim=True)
            y_all = y_all.float()
            y_all = y_all.to(device)

            y_defect_class = [defect, classes, y_all]
            y_good_class = [only_good, classes, y_null_good]
            y_good_industry = [only_good, classes_industry, y_null_product]
            z = torch.randn(num_img, 4, latent_size, latent_size, device=device)

            z = torch.cat([z, z], 0)

            y = [y_defect_class, y_good_class]
            for num in np.arange(0.5, 3, 0.5):
                model_kwargs = dict(y=y, cfg_scale=num)

                with torch.no_grad():
                    samples, cross = diffusion.p_sample_loop(
                            model.forward_with_cfg_2, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True,
                            device=device
                        )

                img_gen, _ = samples.chunk(2, dim=0)

                mask_gen, _ = cross.chunk(2, dim=0)

                with torch.no_grad():
                    img_gen = vae.decode(img_gen / 0.18215).sample
                    mask_gen = vae.decode(mask_gen / 0.18215).sample

                save_image(img_gen,
                                f"img/{c.split()[1]} {c.split()[0]}_{num}.png",
                                nrow=4, normalize=True)
                mask_gen = binarize_tensor_iterative(mask_gen)
                # print(mask_gen.shape)
                mask_gen = (mask_gen * 255).to(torch.uint8)
                mask_gen = mask_gen.float() / 255.0
                save_image(mask_gen,
                                f"img/{c.split()[1]} {c.split()[0]}_{num}_mask.png",
                                nrow=4, normalize=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batchsize", type=int, default=4)
    parser.add_argument("--free", type=int, default=1)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--imagesize", type=int, choices=[256, 512], default=256)
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Optional path to fintune checkpoint.")
    parser.add_argument("--vae", type=str, required=True,
                        help="Optional path to a vae checkpoint.")
    args = parser.parse_args()
    gen(args)


