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
from torchvision.transforms import Lambda
from torch.utils.data import Dataset
import argparse


def main(args):
    device = "cuda"
    model_clip, _ = clip.load('RN50', device)

    data_path = args.data
    image_size = args.imagesize
    batch_size = args.batchsize
    latent_size = image_size // 8

    model = DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, input_size=latent_size, num_classes=1000).to(device)
    state_dict = torch.load(args.ckpt)
    model.load_state_dict(state_dict, strict=False)

    diffusion = create_diffusion(timestep_respacing="")
    vae = AutoencoderKL.from_pretrained(args.vae).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-8)

    
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        Lambda(lambda t: (t * 2) - 1),
    ])

    transform_mask = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        Lambda(lambda t: (t * 2) - 1),
    ])
    transform_resize_mask = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(latent_size),
        transforms.CenterCrop(latent_size),
    ])
    transform_mask_loss = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(latent_size//2),
        transforms.CenterCrop(latent_size//2),
    ])
    class Dataset_self(Dataset):
        def __init__(self,img_root, preprocess):
            self.img_root = img_root
            self.img_process = preprocess
            self.img = []
            self.label_word = []
            self.label_mask = []
            for name_class in os.listdir(self.img_root):
                for bin_classes in os.listdir(self.img_root + '/' + name_class):
                    if bin_classes == 'img':
                        for defect in os.listdir(self.img_root + '/' + name_class + '/' + bin_classes):
                            for name_img in os.listdir(self.img_root + '/' + name_class + '/' + bin_classes + '/' + defect):
                                self.img.append(self.img_root + '/' + name_class + '/' + bin_classes + '/' + defect + '/' + name_img)
                                self.label_word.append(defect + ' ' + name_class)
                                self.label_mask.append(self.img_root + '/' + name_class + '/' + 'ground_truth'
                                                   + '/' + defect + '/' + name_img[:-4] + '_mask.png')

        def __len__(self):
            return len(self.img)

        def __getitem__(self, idx):
            img_path = self.img[idx]
            label_mask_path = self.label_mask[idx]
            label_mask_img = Image.open(label_mask_path)
            label_mask = self.img_process[1](label_mask_img)
            mask_resize = self.img_process[2](label_mask_img)
            # print(mask_resize.shape)
            mask_loss = self.img_process[3](label_mask_img)
            mask_loss = mask_loss[0, :, :]
            mask_loss[mask_loss != 0] = 1
            mask_resize_res = torch.cat([mask_resize, mask_resize[0, :, :].unsqueeze(0)], dim=0)
            label = self.label_word[idx]
            image = Image.open(img_path).convert('RGB')
            image = self.img_process[0](image)
            return image, label, label_mask, mask_resize_res, mask_loss
    dataset = Dataset_self(img_root= data_path, preprocess=[transform, transform_mask, transform_resize_mask, transform_mask_loss])

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=12,
        pin_memory=True,
        drop_last=True
    )
    model.train()
    EPOCH = 1501

    for epoch in range(EPOCH):
        for x, y, mask, mask_resize, mask_loss in loader:
            x = x.to(device)
            mask = mask.to(device)
            mask_resize = mask_resize.to(device)
            mask_loss = mask_loss.to(device)

            drop_rat = 0.2
            if args.free==2:
                for i in range(len(y)):
                    c = y[i]
                    if c.split()[0] == 'good':
                        rat_1 = torch.rand(1)
                        if rat_1 < drop_rat:
                            y[i] = 'good industry'
                        
                    else:
                        rat = torch.rand(1)
                        if rat < drop_rat:
                            y[i] = ('good ' + c.split()[1])
            else:
                for i in range(len(y)):
                    c = y[i]
                    if c.split()[0] != 'good':
                        rat_1 = torch.rand(1)
                        if rat_1 < drop_rat:
                            y[i] = ('good ' + c.split()[1])

            defect = torch.cat([clip.tokenize(f"a photo of {c.split()[0]}") for c in y]).to(device)
            classes = torch.cat([clip.tokenize(f"a photo of {c.split()[1]}") for c in y]).to(device)
            y_all = torch.cat([clip.tokenize(f"a photo of {c}") for c in y]).to(device)

            with torch.no_grad():
                defect = model_clip.encode_text(defect)
                classes = model_clip.encode_text(classes)
                y_all = model_clip.encode_text(y_all)


            defect /= defect.norm(dim=-1, keepdim=True)
            defect = defect.float()
            defect = defect.to(device)

            classes /= classes.norm(dim=-1, keepdim=True)
            classes = classes.float()
            classes = classes.to(device)

            y_all /= y_all.norm(dim=-1, keepdim=True)
            y_all = y_all.float()
            y_all = y_all.to(device)

            with torch.no_grad():
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
                mask_gt = vae.encode(mask).latent_dist.sample().mul_(0.18215)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=[defect, classes, y_all])
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs, mask_resize=mask_resize, mask_att=mask_loss, label_mask=mask_gt)
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            print(epoch, loss)
  
        if epoch % 100 == 0 and 2000>= epoch >= 100:
            torch.save({
                'model_state_dict': model.state_dict(),
            },
            f'checkpoint/model_{epoch}.pth')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batchsize", type=int, default=2)
    parser.add_argument("--free", type=int, default=1)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--imagesize", type=int, choices=[256, 512], default=256)
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Optional path to a DiT checkpoint.")
    parser.add_argument("--vae", type=str, required=True,
                        help="Optional path to a vae checkpoint.")
    args = parser.parse_args()
    main(args)

