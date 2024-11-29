######## 实现了与3D几何处理、神经网络训练辅助以及随机状态初始化等相关的工具函数 ########
# inverse_sigmoid(x): 计算 Sigmoid 函数的逆运算。
# PILtoTorch(pil_image, resolution): 将 PIL 图像对象转换为 PyTorch Tensor;按指定分辨率调整大小，支持 RGB 和 RGBA 图像。
# get_expon_lr_func: 动态调整学习率的函数生成器。支持初始学习率、最终学习率、延迟衰减等配置。
# strip_lowerdiag(L) / strip_symmetric(sym): 提取矩阵的下三角元素，用于简化矩阵操作或特征提取。
# build_rotation(r): 通过四元数 r 构建 3x3 的旋转矩阵。
# build_scaling_rotation(s, r): 构建带缩放的旋转矩阵，将缩放矩阵和旋转矩阵组合。
# safe_state(silent): 初始化随机种子和系统状态，支持控制输出日志的静默模式。
#################################################################################

from datetime import datetime
import random
import torch
import sys
import numpy as np

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

# 将PIL图像转换为PyTorch Tensor，并调整到指定分辨率
def PILtoTorch(pil_image, resolution):
    # 调整图像大小
    resized_image_PIL = pil_image.resize(resolution)
    # 转化numpy数组
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    # 将最后一个通道移动至第一维度，如果是灰度图没有最后一个通道维度，则添加后移动
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim = -1).permute(2, 0, 1)

# 将四元数转换为3D旋转矩阵
def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

# 构建协方差矩阵的一半
def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

# 提取3x3矩阵的下三角部分
def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

# 动态调整学习率的函数生成器，用于优化过程中控制学习率的变化
# lr_init：初始学习率，表示优化开始时的学习率。
# lr_final：最终学习率，优化结束时的学习率。
# lr_delay_steps：延迟步数，在前几个优化步骤中调整学习率。
# lr_delay_mult：延迟倍率，用于在延迟阶段降低学习率。
# max_steps：最大优化步数，定义学习率衰减的范围。
def get_expon_lr_func(lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000):
    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp
    return helper

def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))