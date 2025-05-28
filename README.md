# Few-shot Defect Image Generation based on Consistency Modeling (ECCV 2024)
## Introduction

This repository is an official PyTorch implementation of [Few-shot Defect Image Generation based on Consistency Modeling](https://arxiv.org/pdf/2408.00372) (ECCV 2024). 

<div align="center"><img src="./img/fig 2 architecture.jpg" width="800"></div>

## Setup
The pre-trained DiT models can be downloaded directly here as well:

| DiT Model     | Image Resolution | FID-50K | Inception Score | Gflops | 
|---------------|------------------|---------|-----------------|--------|
| [XL/2](https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-256x256.pt) | 256x256          | 2.27    | 278.24          | 119    |
| [XL/2](https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-512x512.pt) | 512x512          | 3.04    | 240.82          | 525    |


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
