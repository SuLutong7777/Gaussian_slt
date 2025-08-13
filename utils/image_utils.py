########################## 定义了图像质量计算相关函数 ######################
## psnr(img1, img2): 计算两张图像的psnr值
###########################################################################

import torch

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))