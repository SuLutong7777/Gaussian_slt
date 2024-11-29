########################## 高斯模型的各种函数 ##########################
# __init__.py：构造函数，初始化各种属性
# setup_functions：初始化各种数学函数
# capture：返回当前模型的状态，包括所有重要的属性和优化器状态
# restore：从保存的模型参数恢复模型状态
# training_setup：初始化模型的参数，优化器和学习率等
# get_*：用于返回模型的各种参数或经过激活函数处理的值
#     get_scaling: 返回经过指数激活函数处理后的缩放值。
#     get_rotation: 返回经过规范化处理后的旋转向量。
#     get_xyz: 返回点的位置信息。
#     get_features: 返回合并的特征（_features_dc 和 _features_rest）。
#     get_features_dc: 返回深度特征（_features_dc）。
#     get_features_rest: 返回其他特征（_features_rest）。
#     get_opacity: 返回经过 Sigmoid 激活后的透明度值。
#     get_exposure: 返回曝光值（尚未定义具体属性）
#     get_exposure_from_name：根据图像名称返回曝光值
#     get_covariance：返回协方差矩阵
# create_from_pcd：点云数据初始化
# load_ply：从PLY文件加载已保存的点云数据，并将其恢复为模型的参数，支持加载训练期间的曝光信息
# save_ply：将当前模型的点云数据（包括位置、颜色、透明度、旋转、尺度等）保存为PLY文件格式，以便进行可视化或后续使用
# 点云稠密化：
#     prune_points：点云裁剪
#     prune_optimizer：根据给定的掩码修剪优化器中的参数，只保留对应的有效点，移除无效点
#     densification_postfix：在点云数据稠密化（增加更多点）后，更新模型的参数。
#     densify_and_split：根据梯度阈值和场景范围，选择需要稠密化的点，并在它们的附近生成新的点。
#     densify_and_clone：克隆满足条件的点并扩展，进行模型稠密化操作。
#     densify_and_prune：在稠密化之后，进行点云的修剪，移除不需要的点。
# 优化器的张量管理：
#     replace_tensor_to_optimizer：用新的张量替换优化器中的对应参数，确保每次更新后优化器能正确地继续优化新参数。
#     cat_tensors_to_optimizer：将新的张量扩展到优化器中，支持在训练过程中动态地增加或合并点云数据。
# update_learning_rate：根据当前训练的迭代步数，动态调整学习率
# oneupSHdegree：增加球谐函数的阶数。如果当前阶数小于最大阶数，便增加它
##########################################################################################

import json
import os
import torch
import numpy as np
from torch import nn
from plyfile import PlyData, PlyElement
from utils.general_utils import build_scaling_rotation, strip_symmetric, inverse_sigmoid, get_expon_lr_func, build_rotation
from utils.graphics_utils import BasicPointCloud
from utils.sh_utils import RGB2SH
from utils.system_utils import mkdir_p
from simple_knn._C import distCUDA2
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass

class GaussianModel:
    def __init__(self, sh_degree, optimizer_type = "default"):
        # 当前活跃的球谐阶数和最大球谐函数阶数
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        # 位置信息
        self._xyz = torch.empty(0)
        # 特征信息
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        # 缩放
        self._scaling = torch.empty(0)
        # 旋转
        self._rotation = torch.empty(0)
        # 透明度
        self._opacity = torch.empty(0)
        # 二维空间半径
        self.max_radii2D = torch.empty(0)
        # 坐标梯度累积
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        # 优化器
        self.optimizer_type = optimizer_type
        self.optimizer = None
        # 记录稠密程度或稀疏性比例
        self.percent_dense = 0
        # 空间学习率的比例因子
        self.spatial_lr_scale = 0
        self.setup_functions()

    ################# 定义了一些激活函数和数学运算 #################
    def setup_functions(self):
        # 从缩放和旋转信息构建协方差矩阵
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            # 利用旋转四元数和缩放系数构建一个矩阵L
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            # 得到矩阵的上三角参数
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        # 获取协方差矩阵
        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    # @property 装饰器，只读属性，防止直接修改_xyz，保护数据完整性
    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    # 利用相机名称获取曝光信息
    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]

    ################# 从已有文件加载点云信息 #################
    def load_ply(self, ply_path, use_train_test_exp = False):
        print("数据读取模块！！！！-----从特定步骤文件初始化点云信息")
        ############## 加载ply文件 ##############
        plydata = PlyData.read(ply_path)
        # print("ply_path: ", ply_path)
        # print("plydata: ", plydata)
        ############## 加载曝光文件 ############## ????
        if use_train_test_exp:
            exposure_file = os.path.join(os.path.dirname(ply_path), os.pardir, os.pardir, "exposure.json")
            if os.path.exists(exposure_file):
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
                print(f"Pretrained exposures loaded.")
            else:
                print(f"No exposure to be loaded at {exposure_file}")
                self.pretrained_exposures = None
        ############## 加载点云数据位置坐标 ##############
        # 转换成numpy数组，[N, 3]
        xyz = np.stack((np.asarray(plydata.elements[0]['x']),
                        np.asarray(plydata.elements[0]['y']),
                        np.asarray(plydata.elements[0]['z'])), axis=1)
        ############## 加载透明度信息 ##############
        # 在最后增加一个维度 [N, 1]
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        ############## 加载DC特征信息 ##############
        # 这些通常是表示球谐变换中低阶项的系数
        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
        ############## 提取额外特征 ##############
        # 提取所有以f_rest_为开头的特征并排序
        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        print("extra_f_names: ", len(extra_f_names))
        print("max_sh_degree: ", 3*(self.max_sh_degree + 1) ** 2 - 3)
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # 调整特征形状
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))
        ############## 提取旋转和缩放信息 ##############
        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
        ############## 将数据转换为torch张量并设置可训练,并移动至GPU上 ############## .contiguous()确保存储是连续的
        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self.active_sh_degree = self.max_sh_degree

    ################# 初始化并处理点云数据以及相应的相机信息 #################
    # pcd: 点云对象（BasicPointCloud），包含了3D点的位置和颜色。
    # cam_infos: 相机信息的列表，用于后续处理。
    # spatial_lr_scale: 空间学习率缩放因子，可能用于调整训练时的学习率。
    def create_from_pcd(self, pcd:BasicPointCloud, cam_infos, spatial_lr_scale):
        print("数据读取模块！！！！-----从数据集中读取的信息初始化点云数据以及相机信息")
        self.spatial_lr_scale = spatial_lr_scale
        # 将点云数据点坐标转换为张量并移至GPU上
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        # 将RGB颜色转换为球面谐波
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        # 初始化一个张量，用于存储每个点的特征，形状为(点的数量, 3, (max_sh_degree + 1) ** 2)，前三个通道存储颜色，其他通道初始化为0
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0
        # 计算每个点与原点的欧几里得距离的平方 pcd.points:[N, 3],dist2:[N, ], 0.0000001将计算出的最小值限制在0.0000001避免出现0，导致后面梯度问题
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        # 计算每个点的尺度因子，基于距离的对数 [N, 3]?
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        # 初始化旋转参数，第一个元素位置为1， 表示初始情况下每个点的旋转均为单位矩阵（没有旋转）[N ,4]
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device='cuda')
        rots[:, 0] = 1
        # 初始化透明度 [N, 1]
        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device='cuda'))

        ################ 模型参数初始化 ################
        self._xyz = self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        # 点云特征分为两部分，_features_dc存储直接的颜色特征，_features_rest存储剩余的特征 features:[点的数量，3，(max_sh_degree + 1) ** 2]
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        # 存储每个点的尺度信息
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        # 初始化为零，可能用于进一步计算点的最大半径或其他几何约束，形状为[N,]
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        # 创建一个字典，将相机图像名称映射到相应的相机索引
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self.pretrained_exposures = None
        # 为每个相机初始化一个曝光矩阵，这个矩阵是一个 3x4 的张量，表示相机的曝光信息
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

    # 初始化模型的参数，优化器和学习率等
    def training_setup(self, sys_param):
        self.percent_dense = sys_param['percent_dense']
        # 初始化为形状为 (self.get_xyz.shape[0], 1) 的零矩阵，用于累积梯度和梯度规范化，分配在 GPU 上
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        l = [
            {'params': [self._xyz], 'lr': sys_param['position_lr_init'] * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': sys_param['feature_lr'], "name": "f_dc"},
            {'params': [self._features_rest], 'lr': sys_param['feature_lr'] / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': sys_param['opacity_lr'], "name": "opacity"},
            {'params': [self._scaling], 'lr': sys_param['scaling_lr'], "name": "scaling"},
            {'params': [self._rotation], 'lr': sys_param['rotation_lr'], "name": "rotation"}
        ]
        # 设置优化器
        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.exposure_optimizer = torch.optim.Adam([self._exposure])
        # 动态调整 self._xyz 的学习率,设置调度器
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=sys_param['position_lr_init']*self.spatial_lr_scale,
                                                    lr_final=sys_param['position_lr_final']*self.spatial_lr_scale,
                                                    lr_delay_mult=sys_param['position_lr_delay_mult'],
                                                    max_steps=sys_param['position_lr_max_steps'])
        # 为曝光优化器设置调度器
        self.exposure_scheduler_args = get_expon_lr_func(lr_init=sys_param['exposure_lr_init'],
                                                         lr_final=sys_param['exposure_lr_final'],
                                                         lr_delay_steps=sys_param['exposure_lr_delay_steps'],
                                                         lr_delay_mult=sys_param['exposure_lr_delay_mult'],
                                                         max_steps=sys_param['iterations_MAX'])

    # 从检查点恢复模型状态 xyz_gradient_accum:梯度累积
    def restore(self, model_args, sys_param):
        (self.active_sh_degree,
         self._xyz,
         self._features_dc,
         self._features_rest,
         self._scaling,
         self._rotation,
         self._opacity,
         self.max_radii2D,
         xyz_gradient_accum,
         denom,
         opt_dict,
         self.spatial_lr_scale) = model_args
        self.training_setup(sys_param)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        # 恢复优化器状态
        self.optimizer.load_state_dict(opt_dict)

    # 更新学习率
    def update_learning_rate(self, iteration):
        # print("训练模块！！！！-----该更新学习率")
        # 曝光优化器学习率更新
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)
        # xyz参数组学习率更新
        for param_group in self.optimizer.param_groups:
            if param_group['name'] == 'xyz':
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
        return lr
    
    # 增加球谐函数阶数
    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
            print("训练模块！！！！-----增加一次球谐函数阶数")

    # 保存点云文件
    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))
        # 提取模型shujv
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        # 定义ply属性
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        # 创建一个空的 NumPy 数组 elements，大小为点的数量 xyz.shape[0]，数据类型为 dtype_full
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        # 使用 map() 将每个点的属性（attributes 的每一行）转换为元
        elements[:] = list(map(tuple, attributes))
        # 使用 PlyElement.describe() 创建 PLY 格式的顶点元素描述
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    # 根据模型的数据结构构造点云属性的列表，为 PLY 文件的保存定义数据字段的名称
    def construct_list_of_attributes(self):
        # 初始化属性列表
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # 遍历直接特征张量 self._features_dc 的所有通道，为每个通道生成属性名称
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        # 遍历每个尺度通道（self._scaling.shape[1]），生成属性名称s
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        # 遍历旋转张量的每个通道，生成属性名称 
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l
    
    # 更新稠密化过程中的统计数据
    # viewspace_point_tensor：表示视空间中的点云张量，可能包含这些点的位置信息。
    # update_filter：一个布尔型或索引张量，用于筛选需要更新的点。
    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        # 对 viewspace_point_tensor 的梯度进行累积，用于后续稠密化分析。
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)
        # 记录每个点被更新的次数的张量。
        self.denom[update_filter] += 1

    # 根据梯度和透明度等条件对点云进行稠密化和裁剪
    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii):
        # 梯度归一化与修正，累积梯度除以累积计数，得到每个点的平均梯度
        grads = self.xyz_gradient_accum / self.denom
        # 如果某些点的梯度未被更新可能导致 NaN，这里通过将这些梯度设为0进行修正
        grads[grads.isnan()] = 0.0
        # 设置临时半径
        self.tmp_radii = radii
        # 稠密化操作 基于梯度信息 grads 和阈值 max_grad，可能通过复制现有点或插入新点来增加密度。
        # 参数 extent 可能表示场景的空间范围，用于限制稠密化的范围。
        self.densify_and_clone(grads, max_grad, extent)
        # 同样基于梯度信息，但可能采用分裂已有点的方式来增加密度。
        # 两者的区别可能在于实现逻辑，如复制 vs. 分裂。
        self.densify_and_split(grads, max_grad, extent)
        # 裁剪点云，如果某点的透明度（get_opacity 返回值）小于阈值 min_opacity，标记为需裁剪
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        # 大点裁剪，将透明度过低或尺寸过大的点同时标记为需裁剪
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        # 恢复半径并释放显存
        tmp_radii = self.tmp_radii
        self.tmp_radii = None

        torch.cuda.empty_cache()

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    # 负责将稠密化过程生成的新点云属性整合到已有点云中，并初始化相关的优化参数
    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    # 根据梯度和其他条件选取符合要求的点，并将这些点的属性复制为新的点云
    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_tmp_radii = self.tmp_radii[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii)

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tmp_radii)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.tmp_radii = self.tmp_radii[valid_points_mask]

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors