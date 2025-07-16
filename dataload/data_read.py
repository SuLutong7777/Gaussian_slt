import os
import random
from pathlib import Path
import numpy as np
import json
from PIL import Image
from plyfile import PlyData, PlyElement
from typing import NamedTuple
from utils.system_utils import searchForMaxIteration
from utils.graphics_utils import focal2fov, fov2focal, getWorld2View2, BasicPointCloud
from utils.sh_utils import SH2RGB
from utils.camera_utils import Camera
from model.gaussian_model import GaussianModel

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    is_nerf_synthetic: bool
   
# 构建场景类
class Scene:
    def __init__(self, sys_param, gaussian_model:GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        self.sys_param = sys_param
        self.gaussian_model = gaussian_model
        self.model_path = self.sys_param['model_path']  # 存储模型路径
        self.source_path = self.sys_param['source_path']
        self.background = self.sys_param['white_background']
        self.eval = self.sys_param['eval_mode']
        self.device = self.sys_param['data_device']

        ################## 加载已有模型进行训练 ##################
        self.loaded_iter = None
        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("加载迭代次数{}的训练模型".format(self.loaded_iter))

        ################## 加载训练和测试数据 ##################
        self.scene_info = self.SceneLoadBlender(self.source_path, self.background, self.eval)

        if shuffle:                                                        # 打乱相机顺序， 确保训练和测试数据的随机性
            random.shuffle(self.scene_info.train_cameras)
            random.shuffle(self.scene_info.test_cameras)
        self.cameras_extent = self.scene_info.nerf_normalization['radius'] # 设置相机尺度信息
        print("self.cameras_extent: ", self.cameras_extent)

        ################## 初始化点云结构 ##################
        # 加载特定迭代结果
        if self.loaded_iter:
            self.gaussian_model.load_ply(os.path.join(self.model_path, 'point_cloud', 'iteration_' + str(self.loaded_iter), 'point_cloud.ply'))
        # 否则初始化一个点云
        else:
            self.gaussian_model.create_from_pcd(self.scene_info.point_cloud, self.scene_info.train_cameras, self.cameras_extent)
        print("数据读取模块！！！！-----好耶好耶, 数据集读取完毕!")

    def getTrainCameras(self):
        return self.scene_info.train_cameras

    def getTestCameras(self):
        return self.scene_info.test_cameras

    # 加载blender数据集
    def SceneLoadBlender(self, source_path, white_background, eval, extension = '.png', ):
        print("数据读取模块！！！！-----读取blender数据集")
        ###################### 相机参数相关 ######################
        print("读取训练集数据")
        train_cam_infos = self.readCamerasFromTransforms(source_path, 'transforms_train.json', white_background, False, extension)
        print("读取测试集数据")
        test_cam_infos = self.readCamerasFromTransforms(source_path, 'transforms_test.json', white_background, True, extension)

        if not eval:                                # 判断是否为测试模式，如果不是，将测试集也放入训练集
            train_cam_infos.extend(test_cam_infos)
            test_cam_infos = []
        # 相机归一化处理
        nerf_normalization = self.getNerfppNorm(train_cam_infos)

        ###################### 点云相关 ######################
        ply_path = os.path.join(source_path, "points3d.ply")
        if not os.path.exists(ply_path):
            num_pts = 100_000                       # 如果在指定路径下没有找到.ply格式的点云文件，代码会生成一个随机的点云数据，点数为100,000
            print(f'产生{num_pts}个随机点云')
            xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
            shs = np.random.random((num_pts, 3)) / 255.0
            self.storePly(ply_path, xyz, SH2RGB(shs))
        try:
            pcd = self.fetchPly(ply_path)
        except:
            pcd = None

        scene_info = SceneInfo(point_cloud=pcd,
                               train_cameras=train_cam_infos,
                               test_cameras=test_cam_infos,
                               nerf_normalization=nerf_normalization,
                               ply_path=ply_path,
                               is_nerf_synthetic=True)
        return scene_info

    # 从json文件中读取相机信息
    def readCamerasFromTransforms(self, source_path, transformsfile, white_background, is_test, extension = '.png'):
        cam_infos = []
        with open(os.path.join(source_path, transformsfile)) as json_file:
            jsons = json.load(json_file)
            # 相机视场角
            fovx = jsons['camera_angle_x']
            frames = jsons['frames']
            for idx, frame in enumerate(frames):
                ########## 图像相关 ##########
                cam_name = frame['file_path'] + extension          # 图像名字
                image_path = os.path.join(source_path, cam_name)   # 图像路径
                image_name = Path(image_path).stem                 # 提取不带扩展名的图像名称
                
                image_orin = Image.open(image_path)   # (w, h)
                orig_w, orig_h = image_orin.size                   # 分辨率提取
                resolution = (int(orig_w), int(orig_h))
                
                ########## 内参相关 ##########
                fovy = focal2fov(fov2focal(fovx, image_orin.size[0]), image_orin.size[1])       # 利用fovx得到fovy
                FoVx = fovx
                FoVy = fovy

                ########## 外参相关 ##########
                c2w = np.array(frame['transform_matrix'])
                c2w[:3, 1:3] *= -1                         # 坐标系转换，从OpenGL/Blender坐标系(Y up, Z back)转换到COLMAP坐标系(Y down, Z forward)
                w2c = np.linalg.inv(c2w)
                R = np.transpose(w2c[:3, :3])
                T = w2c[:3, 3]
                cam_infos.append(Camera(uid=idx, image=image_orin, image_name=image_name,  resolution=resolution, 
                                         R=R, T=T, FoVx=FoVx, FoVy=FoVy, depth_params=None, invdepthmap=None,
                                         data_device=self.device, is_test_dataset=is_test))
        return cam_infos

    # 归一化相机数据
    def getNerfppNorm(self, cams_info):
        # 计算一组相机中心的几何中心和对角线距离
        def get_center_and_diag(cam_centers):
            # 将列表中的相机中心(列向量)水平拼接成3xN的矩阵
            cam_centers = np.hstack(cam_centers)
            # 沿着维度axis计算均值(1表示计算每一行的均值)，并且结果保持原来的维度 [3x1]
            avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
            center = avg_cam_center
            # 求每个相机中心到中心点的距离
            dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
            diagonal = np.max(dist)
            # 将center转化为一维数组
            return center.flatten(), diagonal
        
        # 获取相机中心矩阵
        cams_centers = []
        for cam in cams_info:
            W2C = getWorld2View2(cam.R, cam.T)
            C2W = np.linalg.inv(W2C)
            cams_centers.append(C2W[:3, 3:4])
        center, diagonal = get_center_and_diag(cams_centers)
        radius = diagonal * 1.1
        translate = -center
        return {"translate": translate, "radius":radius}

    # 将3D点云数据保存为ply文件
    def storePly(self, ply_path, xyzs, rgbs):
        # 定义一个复合数据类型，并定义了每个点的属性
        # x, y, z: 3D 坐标（'f4'表示 32 位浮动点数）
        # nx, ny, nz: 法线向量（同样为 32 位浮动点数）。
        # red, green, blue: 颜色（无符号字节，范围0-255）
        dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
                ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
        normals = np.zeros_like(xyzs)

        elements = np.empty(xyzs.shape[0], dtype=dtype)
        attributes = np.concatenate((xyzs, normals, rgbs), axis=1)
        elements[:] = list(map(tuple, attributes))
        # 创建ply合适数据并存入文件
        vertex_element = PlyElement.describe(elements, 'vertex')
        ply_data = PlyData([vertex_element])
        ply_data.write(ply_path)

    # 从一个 PLY 文件中读取点云数据，包括每个点的坐标、颜色和法线
    def fetchPly(self, ply_path):
        plydata = PlyData.read(ply_path)
        # 获取顶点数据
        vertices = plydata['vertex']
        positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
        return BasicPointCloud(points=positions, colors=colors, normals=normals)

    def save(self, iteration):
        # 创建存储点云信息的文件
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussian_model.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        # 生成曝光字典
        exposure_dict = {
            image_name: self.gaussian_model.get_exposure_from_name(image_name).detach().cpu().numpy().tolist()
            for image_name in self.gaussian_model.exposure_mapping
        }
        # 保存曝光字典
        with open(os.path.join(self.model_path, "exposure.json"), "w") as f:
            json.dump(exposure_dict, f, indent=2)
