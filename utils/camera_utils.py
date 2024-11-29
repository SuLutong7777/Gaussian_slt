#################### 相机类定义与序列化 ####################
## Camera: 定义了一个相机类，封装了相机信息
## loadCam(): 从单个相机信息中，加载图像、深度图，并根据分辨率缩放参数生成Camera对象
## cameraList_from_camInfos(): 将多个相机信息封装为多个相机类列表
## camera_to_JSON(): 将相机的位姿和参数信息转换为可序列化的 JSON 格式，便于存储或进一步处理
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
resolution:图像的目标分辨率。
colmap_id:相机在 COLMAP 中的唯一标识符。
R / T:相机的旋转矩阵和平移向量，表示相机位姿。
FoVx / FoVy:相机的水平和垂直视场角。
depth_params:深度图参数，包括 scale 和 offset 等，用于调整深度图的值域。
image:输入图像。
invdepthmap:反深度图。
image_name:图像的文件名。
uid:相机的唯一标识符。
trans / scale:额外的平移和缩放参数，用于调整相机位姿。
data_device:指定模型和数据的计算设备（默认 "cuda"）。
train_test_exp / is_test_dataset / is_test_view:控制训练和测试相关的透明度调整。
"""
class Camera(nn.Module):
    def __init__(self, resolution, colmap_id, R, T, FoVx, FoVy, depth_params, image, invdepthmap, image_name,
                 uid, trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device='cuda', train_test_exp=False,
                 is_test_dataset = False, is_test_view=False):
        super(Camera, self).__init__()
        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        ############### 设置计算设备,优先设置指定设备，如果失败则用默认cuda设备 ###############
        try:
            self.data_device = data_device
        except Exception as e:
            print(e)
            print(f"[warning] 用户指定计算设备{data_device}设置失败,使用默认cuda设备")
            self.data_device = torch.device("cuda")
        ############### 图像处理 ###############
        # 将输入图像转换为tensor，并调整分辨率
        resized_image_rgb = PILtoTorch(image, resolution)
        # 提取RGB通道作为原始图像
        gt_image = resized_image_rgb[:3, ...]
        # 如果图像包含alpha通道，则生成透明度掩膜，如果不包含，默认全白
        self.alpha_mask = None
        if resized_image_rgb.shape[0] == 4:
            self.alpha_mask = resized_image_rgb[3:4, ...].to(self.data_device)
        else:
            self.alpha_mask = torch.ones_like(resized_image_rgb[0:1, ...].to(self.data_device))
        ############### 透明度调整 ###############
        # 如果在进行训练-测试实验并且当前视图是测试视图
        if train_test_exp and is_test_view:
            # 如果当前数据集是测试集
            if is_test_dataset:
                self.alpha_mask[..., :self.alpha_mask.shape[-1] // 2] = 0
            else:
                self.alpha_mask[..., self.alpha_mask.shape[-1] // 2:] = 0
        self.original_image = gt_image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]
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
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        # 相机的投影矩阵，从3D坐标投影到2D坐标
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        # 计算全局投影矩阵
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        # 相机中心计算 得到视图到世界的变换矩阵。[3, :3]：提取第 4 行的前三个分量，即平移向量
        self.camera_center = self.world_view_transform.inverse()[3, :3]

WARNED = False

# 从单个相机信息中，加载图像、深度图，并根据分辨率缩放参数生成Camera对象
def loadCam(sys_param, id, cam_info, resolution_scale, is_nerf_synthetic, is_test_dataset):
    
    # 图像加载
    image = Image.open(cam_info.image_path)
    # 深度图加载并归一化
    if cam_info.depth_path != "":
        try:
            if is_nerf_synthetic:
                invdepthmap = cv2.imread(cam_info.depth_path, -1).astype(np.float32) / 512
            else:
                invdepthmap = cv2.imread(cam_info.depth_path, -1).astype(np.float32) / float(2**16)

        except FileNotFoundError:
            print(f"Error: The depth file at path '{cam_info.depth_path}' was not found.")
            raise
        except IOError:
            print(f"Error: Unable to open the image file '{cam_info.depth_path}'. It may be corrupted or an unsupported format.")
            raise
        except Exception as e:
            print(f"An unexpected error occurred when trying to read depth at {cam_info.depth_path}: {e}")
            raise
    else:
        invdepthmap = None
    # 分辨率调整
    orig_w, orig_h = image.size
    if sys_param['resolution'] in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * sys_param['resolution'])), round(orig_h/(resolution_scale * sys_param['resolution']))
    else:  # should be a type that converts to float
        if sys_param['resolution'] == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / sys_param['resolution']
    

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    return Camera(resolution, colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FoVx, FoVy=cam_info.FoVy, depth_params=cam_info.depth_params,
                  image=image, invdepthmap=invdepthmap,
                  image_name=cam_info.image_name, uid=id, data_device=sys_param['data_device'],
                  train_test_exp=sys_param['train_test_exp'], is_test_dataset=is_test_dataset, is_test_view=cam_info.is_test)

# 该函数用于批量加载多个相机配置，并返回一个 Camera 对象列表
def cameraList_from_camInfos(cam_infos, resolution_scale, sys_param, is_nerf_synthetic, is_test_dataset):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(sys_param, id, c, resolution_scale, is_nerf_synthetic, is_test_dataset))

    return camera_list

# 将相机的位姿和参数信息转换为可序列化的 JSON 格式，便于存储或进一步处理
def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FoVy, camera.height),
        'fx' : fov2focal(camera.FoVx, camera.width)
    }
    return camera_entry