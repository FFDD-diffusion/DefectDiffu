# Few-shot Defect Image Generation based on Consistency Modeling (ECCV 2024)
## Introduction

This repository is an official PyTorch implementation of [Few-shot Defect Image Generation based on Consistency Modeling](https://arxiv.org/pdf/2408.00372) (ECCV 2024). 

<div align="center"><img src="./img/fig 2 architecture.jpg" width="800"></div>

## Setup
The pre-trained DiT models can be downloaded directly here as well:

| DiT Model     | Image Resolution | 
|---------------|------------------|
| [DiT-256](https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-256x256.pt) | 256x256          |
| [DiT-512](https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-512x512.pt) | 512x512          | 

We use the pretrained VAE model from Stable Diffusion:
https://huggingface.co/stabilityai/sd-vae-ft-mse/tree/main

## Train
```
cd DefectDiffu
python train.py \
    --ckpt model_path \
    --batchsize 2 \
    --vae /vae_path \
    --data /mvtec_path
```

## Test
```
cd DefectDiffu
python test.py \
    --ckpt model_path \
    --vae /vae_path \
    --data /mvtec_path
```


## Citing
If you find this code useful in your research, please consider citing us:
```
@InProceedings{defectdiffu,
author={Shi, Qingfeng and Wei, Jing and Shen, Fei and Zhang, Zhengtao},
title={Few-Shot Defect Image Generation Based on Consistency Modeling},
booktitle={European Conference on Computer Vision (ECCV 2024)},
year={2024},
pages={360--376}
}
``` 
