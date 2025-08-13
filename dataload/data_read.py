import os
import random
from pathlib import Path
import numpy as np
import json
import struct
import collections
from PIL import Image
from plyfile import PlyData, PlyElement
from typing import NamedTuple

import torch
from utils.system_utils import searchForMaxIteration
from utils.graphics_utils import focal2fov, fov2focal, getWorld2View2, qvec2rotmat, BasicPointCloud
from utils.sh_utils import SH2RGB
from utils.camera_utils import Camera
from model.gaussian_model import GaussianModel

CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
Cameras = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    is_nerf_synthetic: bool

class Images(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)
    
CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}

CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model)
                         for camera_model in CAMERA_MODELS])

# 构建场景类
class Scene:
    def __init__(self, sys_param, gaussian_model:GaussianModel, load_iteration=None, shuffle=True, scale = 4.0):
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
        if os.path.exists(os.path.join(self.sys_param['source_path'], 'transforms_train.json')):
            self.scene_info = self.SceneLoadBlender(self.source_path, self.background, self.eval)
        elif os.path.exists(os.path.join(self.sys_param['source_path'], 'sparse')):
            self.scene_info = self.SceneLoadColmap(self.source_path, self.eval, scale=scale)

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
    
    def SceneLoadColmap(self, source_path, eval, scale):
        print("数据读取模块！！！！-----读取colmap数据集")
        ###################### 读取相机内外参数 ######################
        try:
            cameras_extrinsic_file = os.path.join(source_path, 'sparse/0', 'images.bin')
            cameras_intrinsic_file = os.path.join(source_path, 'sparse/0', 'cameras.bin')
            cam_extrinsics = self.read_extrinsics_binary(cameras_extrinsic_file)
            cam_intrinsics = self.read_intrinsics_binary(cameras_intrinsic_file)
            print("数据读取模块！！！！-----二进制文件中外参信息(cam_extrinsics): ", len(cam_extrinsics))
            print("数据读取模块！！！！-----二进制文件中内参信息(cam_intrinsics): ", len(cam_intrinsics))
        except:
            print("数据读取模块！！！！-----二进制文件读取失败，尝试使用文本文件读取")
            cameras_extrinsic_file = os.path.join(source_path, 'sparse/0', 'images.txt')
            cameras_intrinsic_file = os.path.join(source_path, 'sparse/0', 'cameras.txt')
            cam_extrinsics = self.read_extrinsics_text(cameras_extrinsic_file)
            cam_intrinsics = self.read_intrinsics_text(cameras_intrinsic_file)
            print("数据读取模块！！！！-----文本文件中外参信息(cam_extrinsics): ", len(cam_extrinsics))
            print("数据读取模块！！！！-----文本文件中内参信息(cam_intrinsics): ", len(cam_intrinsics))

        ######################## 选择测试相机 ######################
        if eval:
            llffhold = 8
            cam_names = [cam_extrinsics[cam_id].name for cam_id in cam_extrinsics]
            cam_names = sorted(cam_names)
            test_cam_names_list = [name for idx, name in enumerate(cam_names) if idx % llffhold == 0]  # 每8张图作为测试集
        else:
            test_cam_names_list = []
        ####################### 将相机内外参数合并在一起 ######################
        cam_infos_unsorted = self.readColmapCameras(
            cam_extrinsics, cam_intrinsics,
            images_folder = os.path.join(source_path, 'images'),
            test_cam_name_list= test_cam_names_list, scale=scale
        )
        # 划分训练和测试数据集
        cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x: x.image_name)
        train_cam_infos = [c for c in cam_infos if not c.is_test_dataset]
        test_cam_infos = [c for c in cam_infos if c.is_test_dataset]
        # 得到归一化相机数据
        nerf_normalization = self.getNerfppNorm(train_cam_infos)
        ####################### 加载或生成点云数据 #######################
        ply_path = os.path.join(source_path, "sparse/0/points3D.ply")
        bin_path = os.path.join(source_path, "sparse/0/points3D.bin")
        # 检查是否存在ply文件，如果不存在利用bin或者txt文件生成ply文件
        if not os.path.exists(ply_path):
            xyz, rgb, _ = self.read_points3D_binary(bin_path)
            self.storePly(ply_path, xyz, rgb)
            
        try:
            print("数据读取模块！！！！-----点云ply文件存在! 好耶!")
            # 提取ply点云信息
            pcd = self.fetchPly(ply_path)
            print("数据读取模块！！！！-----初始点云信息pcd: ")
        except:
            pcd = None
        scene_info = SceneInfo(point_cloud=pcd,
                               train_cameras=train_cam_infos,
                               test_cameras=test_cam_infos,
                               nerf_normalization=nerf_normalization,
                               ply_path=ply_path,
                               is_nerf_synthetic=False)
        return scene_info

    # 读取colmap生成的外参txt文件
    def read_extrinsics_text(self, cam_extrinsics_file):
        print("数据读取模块！！！！-----读取colmap数据集txt格式外参文件")
        images = {}
        with open(cam_extrinsics_file, 'r') as fid:
            while True:
                # 读取一行
                line = fid.readline()
                if not line:
                    break
                # 跳过行首行尾空格制表符等
                line = line.strip()
                if len(line) > 0 and line[0] != '#':
                    elems = line.split()
                    # 图像id
                    image_id = int(elems[0])
                    # 旋转矩阵四元数
                    qvec = np.array(tuple(map(float, elems[1:5])))
                    tvec = np.array(tuple(map(float, elems[5:8])))
                    # 对应相机id
                    camera_id = int(elems[8])
                    # 图像名称
                    image_name = elems[9]
                    elems = fid.readline().split()
                    # 2D点坐标
                    xys = np.column_stack([tuple(map(float, elems[0::3])),
                                        tuple(map(float, elems[1::3]))])
                    point3D_ids = np.array(tuple(map(int, elems[2::3])))
                    images[image_id] = Image(
                        id=image_id, qvec=qvec, tvec=tvec,
                        camera_id=camera_id, name=image_name,
                        xys=xys, point3D_ids=point3D_ids)
        return images

    # 读取colmap生成的内参txt文件
    def read_intrinsics_text(self, cam_intrinsics_file):
        print("数据读取模块！！！！-----读取colmap数据集txt格式内参文件")
        cameras = {}
        with open(cam_intrinsics_file, "r") as fid:
            while True:
                line = fid.readline()
                if not line:
                    break
                line = line.strip()
                if len(line) > 0 and line[0] != "#":
                    elems = line.split()
                    camera_id = int(elems[0])
                    model = elems[1]
                    assert model == "PINHOLE", "While the loader support other types, the rest of the code assumes PINHOLE"
                    width = int(elems[2])
                    height = int(elems[3])
                    params = np.array(tuple(map(float, elems[4:])))
                    cameras[camera_id] = Camera(id=camera_id, model=model,
                                                width=width, height=height,
                                                params=params)
        return cameras

    # 读取colmap生成的外参二进制文件
    def read_extrinsics_binary(self, cam_extrinsics_file):
        print("数据读取模块！！！！-----读取colmap数据集二进制外参文件")
        # 初始化一个图片数据字典
        images = {}
        with open(cam_extrinsics_file, 'rb') as fid:
            # 读取文件中图像的数量, Q代表8字节的无符号长整型
            num_images = self.read_next_bytes(fid, 8, 'Q')[0]
            # 遍历每个图像
            for _ in range(num_images):
                image_params = self.read_next_bytes(fid, num_bytes=64, format_char_sequence='idddddddi')
                # 图像唯一标识符
                image_id = image_params[0]
                # 图像的旋转矩阵四元数
                qvec = np.array(image_params[1:5])
                # 图像的平移三维向量
                tvec = np.array(image_params[5:8])
                # 对应相机id
                camera_id = image_params[8]
                # 读取图像名称
                image_name = ""
                current_char = self.read_next_bytes(fid, 1, 'c')[0]
                while current_char != b"\x00":
                    image_name += current_char.decode("utf-8")
                    current_char = self.read_next_bytes(fid, 1, 'c')[0]
                # 读取图像中2D点数量
                num_points2D = self.read_next_bytes(fid, 8, 'Q')[0]
                # 读取2D点坐标和对应3D点id
                x_y_id_s = self.read_next_bytes(fid, 24*num_points2D, 'ddq'*num_points2D)
                # 每隔3个数取一次，取第一个数和第二个数 使用 map(float, ...) 将这些坐标值转换为浮点数。
                # 最后，tuple(...) 将转换后的浮点数值组合成一个元组。
                xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])), tuple(map(float, x_y_id_s[1::3]))])
                point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
                images[image_id] = Images(
                    id = image_id, qvec = qvec, tvec = tvec,
                    camera_id = camera_id, name = image_name,
                    xys = xys, point3D_ids = point3D_ids 
                )
        return images

    # 读取colmap生成的内参二进制文件
    def read_intrinsics_binary(self, cam_intrinsics_file):
        print("数据读取模块！！！！-----读取colmap数据集二进制内参文件")
        # 初始化一个相机参数文件
        cameras = {}
        with open(cam_intrinsics_file, 'rb') as fid:
            # 读取相机数量
            num_cameras = self.read_next_bytes(fid, 8, 'Q')[0]
            # 遍历每个相机并读取属性
            for _ in range(num_cameras):
                camera_params = self.read_next_bytes(fid, 24, 'iiQQ')
                # 相机id
                camera_id = camera_params[0]
                # 相机模型id
                model_id = camera_params[1]
                # 利用model_id查找相机模型名称
                model_name = CAMERA_MODEL_IDS[model_id].model_name
                # 图像宽度
                width = camera_params[2]
                # 图像高度
                height = camera_params[3]
                # 查询该相机参数数量并获取内参值
                num_params = CAMERA_MODEL_IDS[model_id].num_params
                intr_params = self.read_next_bytes(fid, 8*num_params, "d"*num_params)
                # 存储相机类
                cameras[camera_id] = Cameras(
                    id = camera_id,
                    model = model_name,
                    width = width,
                    height = height,
                    params = np.array(intr_params)
                )
        return cameras

    # 合并相机的内外参数信息
    def readColmapCameras(self, cam_extrinsics, cam_intrinsics, images_folder, test_cam_name_list, scale = 1.0):
        cam_infos = []
        for idx, key in enumerate(cam_extrinsics):
            # print('Reading camera {}/{}'.format(idx + 1, len(cam_extrinsics)))

            # 获取相机外参字典
            extr = cam_extrinsics[key]
            # 获取相机内参字典
            intr = cam_intrinsics[extr.camera_id]
            # 相机分辨率
            height = intr.height / scale
            width = intr.width / scale
            # 相机id
            uid = intr.id
            # 该图片对应的相机外参
            R = np.transpose(qvec2rotmat(extr.qvec))
            T = np.array(extr.tvec)
            # print("intr.model: ", intr.model)
            # 相机焦距
            if intr.model=="SIMPLE_PINHOLE" or intr.model=="SIMPLE_RADIAL":
                focal_length_x = intr.params[0] / scale
                FovY = focal2fov(focal_length_x, height)
                FovX = focal2fov(focal_length_x, width)
            elif intr.model=="PINHOLE":
                focal_length_x = intr.params[0] / scale
                focal_length_y = intr.params[1] / scale
                FovY = focal2fov(focal_length_y, height)
                FovX = focal2fov(focal_length_x, width)
            else:
                assert False, '无法处理相机模型'

            # 图片路径
            image_path = os.path.join(images_folder, extr.name)
            image_orin = Image.open(image_path)   # (w, h)
            orig_w, orig_h = image_orin.size                   # 分辨率提取
            orig_w, orig_h = orig_w / scale, orig_h / scale
            resolution = (int(orig_w), int(orig_h))
            image_name = extr.name
            # 将该图片所有信息存入字典
            cam_infos.append(Camera(uid=uid, image=image_orin, image_name=image_name, resolution=resolution,
                                  R=R, T=T, FoVx=FovX, FoVy= FovY, depth_params=None, invdepthmap=None,
                                  data_device=self.device, is_test_dataset = image_name in test_cam_name_list))
        return cam_infos

    # 从点云二进制文件中读取信息
    def read_points3D_binary(self, bin_path):
        with open(bin_path, 'rb') as fid:
            # 读取点云数量
            num_points = self.read_next_bytes(fid, 8, 'Q')[0]
            # 构造点云坐标、RGB、误差矩阵
            xyzs = np.empty((num_points, 3))
            rgbs = np.empty((num_points, 3))
            errors = np.empty((num_points, 1))

            for p_id in range(num_points):
                point_params = self.read_next_bytes(fid, 43, 'QdddBBBd')
                xyz = np.array(point_params[1:4])
                rgb = np.array(point_params[4:7])
                error = np.array(point_params[7])
                # 每个点可以有一个轨迹（多个观测），首先读取轨迹的长度，再读取每个轨迹元素的索引。
                track_length = self.read_next_bytes(fid, 8, "Q")[0]
                track_elems = self.read_next_bytes(fid, 8*track_length, "ii"*track_length)
                xyzs[p_id] = xyz
                rgbs[p_id] = rgb
                errors[p_id] = error
        return xyzs, rgbs, errors
    
    # 从二进制文件中读取指定数量的字节
    def read_next_bytes(self, fid, num_bytes, format_char_sequence, endian_character = "<"):
        data = fid.read(num_bytes)
        # 返回解包数据，一个元组
        return struct.unpack(endian_character + format_char_sequence, data)

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
            cam_centers = torch.cat(cam_centers, dim=1)
            # 沿着维度axis计算均值(1表示计算每一行的均值)，并且结果保持原来的维度 [3x1]
            center = torch.mean(cam_centers, dim=1, keepdim=True)
            # 求每个相机中心到中心点的距离
            dist = torch.norm(cam_centers - center, dim=0, keepdim=True)
            diagonal = torch.max(dist)
            # 将center转化为一维数组
            return center.flatten(), diagonal
                
        # 获取相机中心矩阵
        cams_centers = []
        for cam in cams_info:
            W2C = getWorld2View2(cam.R, cam.T)
            C2W = torch.linalg.inv(W2C)
            cams_centers.append(C2W[:3, 3:4])
        center, diagonal = get_center_and_diag(cams_centers)
        radius = diagonal * 1.1
        translate = -center
        return {"translate": translate, "radius":radius}

    # COLMAP产生的相机位姿的世界坐标系不一定是啥样
    # 这个操作将COLMAP生成的坐标系进行转换，变成以环绕中心为世界坐标系原点的全新分布坐标
    # pca是指主成分分析，Principal Component Analysis，一种数据降维方法
    # 主成分分析可以看这个：https://zhuanlan.zhihu.com/p/37777074
    # 这段代码实现PAC用的是上面这个链接中3.5的(1)方法
    # 输入poses为[N, 4, 4], 必须为c2w矩阵，不能为w2c
    def transform_poses_pca(self, poses_c2w):
        poses_c2w = poses_c2w.detach().cpu()
        # 获取所有相机的中心点
        trans = poses_c2w[:, :3, 3]
        # 取平均值
        trans_mean = torch.mean(trans, dim=0)
        # 中心化，相当于取所有点的平均中心为新坐标原点
        # 生成新的相机中心位置 [194, 3]
        trans = trans - trans_mean
        # 计算特征值eigval，和特征向量eigvec
        # 注意，这两个算出来是复数格式，有实部和虚部，即使虚部为0，也会保留
        # 所以这里要除去虚部(虚部全部算出来都是0)
        # trans.T @ trans: [3,3], 注意，这个过程在计算平移向量集合的协方差（正常有个除以n的系数，但是不影响特征向量）
        # eigval:[3], eigvec:[3,3]
        # eigval, eigvec = torch.linalg.eig(trans.T @ trans)
        # 转成Numpy做，pytorch版本的特征向量符号与Numpy不一致
        eigval, eigvec = np.linalg.eig(np.array(trans).T @ np.array(trans))
        eigval = torch.from_numpy(eigval)
        eigvec = torch.from_numpy(eigvec)
        # print(eigval, eigvec)
        # exit()
        # 对所有特征值进行从大到小的排序，获取排序的索引
        inds = torch.argsort(eigval.real, descending=True)
        # 同时排序特征向量
        # eigvec = eigvec[:, inds].real
        eigvec = eigvec[:, inds]
        # print(eigvec, "2222")
        # 将特征向量转置，构造投影矩阵，将所有坐标点投影到新的坐标系下
        # 这个新的坐标系的轴就是数据的主成分轴。
        # 这里eigvec为[3,3]，因为数据一共有三个主成分，分别为x,y,z，都需要保留，所以上面链接中的k值取3，就等同于不用筛选
        # eigvec中，每一列是特征向量，转置之后变成行，在进行投影的时候就是rot@trans，x,y,z维度能对应
        rot = eigvec.T
        # 保持坐标系变换后与原来规则相同
        # 在三维空间中，一个合法的旋转矩阵应该是正交的且行列式为1，这保证了坐标系变换保持了空间的右手规则。
        # 如果行列式小于0，表明旋转矩阵将导致坐标系翻转，违反了右手规则。
        # 一个矩阵的行列式（np.linalg.det(rot)）告诉我们这个矩阵是保持空间的定向（右手或左手）不变还是改变了空间的定向。具体来说：
        # 如果行列式大于0，说明变换后的坐标系保持原有的定向（即如果原坐标系是右手的，变换后仍然是右手的）。
        # 如果行列式小于0，说明变换后的坐标系改变了原有的定向（即从右手变为了左手，或从左手变为了右手）。
        if torch.linalg.det(rot) < 0:
            rot = torch.diag(torch.tensor([1.0, 1.0, -1.0])) @ rot

        # 构建完整的[R|T]变换矩阵，直接针对原始的pose信息，不再单纯考虑trans
        # 尺寸是[3, 4]
        transform_mat = torch.cat([rot, rot @ -trans_mean[:, None]], dim=-1)
        # 转为[4, 4]
        transform_mat = torch.cat([transform_mat, torch.tensor([[0, 0, 0, 1.]])], dim=0)
        # 整体RT矩阵转换[N, 4, 4]
        poses_recentered = transform_mat @ poses_c2w

        # 检查坐标轴方向
        # 检查在新坐标系中，相机指向的平均方向的y分量是否向下。如果是的话，这意味着变换后的位姿与常规的几何或物理约定（例如，通常期望的y轴向上）不符。
        if poses_recentered.mean(axis=0)[2, 1] < 0:
            poses_recentered = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0])) @ poses_recentered
            transform_mat = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0])) @ transform_mat
        
        # 原始相机方向向量的平均（这里假设第三列是前向向量）
        orig_forward = poses_c2w[:, :3, 2].mean(0)
        new_forward = poses_recentered[:, :3, 2].mean(0)

        # 如果方向相反（dot product < 0），就翻转 Z 和 X（保持右手系）
        if (orig_forward @ new_forward) < 0:
            poses_recentered = torch.diag(torch.tensor([-1.0, 1.0, -1.0, 1.0])) @ poses_recentered
            transform_mat = torch.diag(torch.tensor([-1.0, 1.0, -1.0, 1.0])) @ transform_mat

        # 对数据进行归一化，收敛到[-1, 1]之间
        scale_factor = 1. / torch.max(torch.abs(poses_recentered[:, :3, 3]))
        poses_recentered[:, :3, 3] *= scale_factor
        poses_recentered[:, 3, :] = torch.tensor([0.0, 0.0, 0.0, 1.0]).repeat(poses_recentered.shape[0], 1)
        transform_mat = torch.diag(torch.tensor([scale_factor] * 3 + [1])) @ transform_mat
        
        return poses_recentered, transform_mat

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
