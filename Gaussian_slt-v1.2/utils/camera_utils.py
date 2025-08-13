#################### 相机类定义与序列化 ####################
## Camera: 定义了一个相机类，封装了相机信息
##########################################################


import numpy as np
from .graphics_utils import fov2focal
from PIL import Image
import cv2
import torch
from torch import nn
from utils.general_utils import PILtoTorch
from utils.graphics_utils import getWorld2View2
from utils.graphics_utils import getProjectionMatrix

# 封装一个相机类，用于描述和管理相机的相关参数和操作，包括相机的位姿、投影矩阵、深度图信息以及图像数据等
"""
输入参数:
uid:相机的唯一标识符
image:输入图像
image_name:图像的文件名
resolution:图像的目标分辨率
R / T:相机的旋转矩阵和平移向量，表示相机位姿
FoVx / FoVy:相机的水平和垂直视场角
depth_params:深度图参数，包括 scale 和 offset 等，用于调整深度图的值域
invdepthmap:反深度图
trans / scale:额外的平移和缩放参数，用于调整相机位姿
data_device:指定模型和数据的计算设备(默认 "cuda")
train_test_exp / is_test_dataset / is_test_view:控制训练和测试相关的透明度调整
"""
class Camera(nn.Module):
    def __init__(self, uid, image, image_name, resolution, R, T, FoVx, FoVy, depth_params, invdepthmap, 
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device='cuda', is_test_dataset = False):
        super(Camera, self).__init__()
        self.uid = uid
        self.R = torch.tensor(R).float().cuda()
        self.T = torch.tensor(T).float().cuda()
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.is_test_dataset = is_test_dataset
        ############### 设置计算设备,优先设置指定设备，如果失败则用默认cuda设备 ###############
        try:
            self.data_device = data_device
        except Exception as e:
            print(e)
            print(f"[warning] 用户指定计算设备{data_device}设置失败,使用默认cuda设备")
            self.data_device = torch.device("cuda")
        ############### 图像处理 ###############
        # 将输入图像转换为tensor，并调整分辨率
        resized_image_rgb = PILtoTorch(image, resolution)    # resized_image_rgb:  torch.Size([4, 800, 800])
        if resized_image_rgb.shape[0] == 4:
            rgb, alpha = resized_image_rgb[:3, ...], resized_image_rgb[3:, ...]  # alpha shape: [1, H, W]
            white_bg = torch.ones_like(rgb)
            gt_image = rgb * alpha + white_bg * (1 - alpha)
        else:
            gt_image = resized_image_rgb[:3, ...]
        ############### 透明度调整 ###############
        # 如果在进行训练-测试实验并且当前视图是测试视图
        self.original_image = gt_image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]
        fx = 0.5 * self.image_width / np.tan(FoVx / 2)
        fy = 0.5 * self.image_height / np.tan(FoVy / 2)
        cx = self.image_width / 2
        cy = self.image_height / 2
        self.K = torch.tensor([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=torch.float32, device=self.data_device)
        ############### 深度图处理 ###############
        # 初始化深度图和相关掩膜
        self.invdepthmap = None
        self.depth_reliable = False
        if invdepthmap is not None:
            # 创建深度掩膜
            self.depth_mask = torch.ones_like(self.alpha_mask)
            # 调整反深度图的分辨率，并假定深度图初始有效
            self.invdepthmap = cv2.resize(invdepthmap, resolution)
            self.invdepthmap[self.invdepthmap < 0] = 0
            self.depth_reliable = True
            # 根据深度参数调整值域
            # 检查 scale 是否在合理范围内（[0.2 * med_scale, 5 * med_scale]）。如果超出范围，设置深度不可靠，并将深度掩膜清零。
            if depth_params is not None:
                if depth_params["scale"] < 0.2 * depth_params["med_scale"] or depth_params["scale"] > 5 * depth_params["med_scale"]:
                    self.depth_reliable = False
                    self.depth_mask *= 0
                # 如果 scale > 0，对深度图应用缩放和偏移：
                if depth_params["scale"] > 0:
                    self.invdepthmap = self.invdepthmap * depth_params["scale"] + depth_params["offset"]
            if self.invdepthmap.ndim != 2:
                self.invdepthmap = self.invdepthmap[..., 0]
            # 将深度图标准化并转换成tensor格式
            self.invdepthmap = torch.from_numpy(self.invdepthmap[None]).to(self.data_device)
        
        # 相机的近远裁剪平面
        self.zfar = 100.0
        self.znear = 0.01
        # 相机缩放和平移
        self.trans = trans
        self.scale = scale
        # 从世界坐标系到相机坐标系的变换矩阵
        self.world_view_transform = torch.tensor(getWorld2View2(self.R, self.T, trans, scale)).transpose(0, 1).cuda()
        # 相机的投影矩阵，从3D坐标投影到2D坐标
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        # 计算全局投影矩阵
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        # 相机中心计算 得到视图到世界的变换矩阵。[3, :3]：提取第 4 行的前三个分量，即平移向量
        self.camera_center = self.world_view_transform.inverse()[3, :3]
