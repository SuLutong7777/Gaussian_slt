import os
import random
from pathlib import Path
import struct
import sys
import numpy as np
import collections
import json
from PIL import Image
from plyfile import PlyData, PlyElement
from typing import NamedTuple
import torch
from utils.system_utils import searchForMaxIteration
from utils.graphics_utils import focal2fov, fov2focal, getWorld2View2, qvec2rotmat, BasicPointCloud
from utils.sh_utils import SH2RGB
from utils.camera_utils import camera_to_JSON, cameraList_from_camInfos
from utils.general_utils import PILtoTorch
from model.gaussian_model import GaussianModel

CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])

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
CAMERA_MODEL_NAMES = dict([(camera_model.model_name, camera_model)
                           for camera_model in CAMERA_MODELS])

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FoVy: np.array
    FoVx: np.array
    depth_params: dict
    image_path: str
    image_name: str
    depth_path: str
    width: int
    height: int
    is_test: bool


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    is_nerf_synthetic: bool

    def to_dict(self):
        """
        将 SceneInfo 转换为字典格式，确保所有字段可序列化
        """
        return {
            "point_cloud": self.point_cloud.to_dict(),
            "train_cameras": self.train_cameras,
            "test_cameras": self.test_cameras,
            "nerf_normalization": {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in self.nerf_normalization.items()},
            "ply_path": self.ply_path,
            "is_nerf_synthetic": self.is_nerf_synthetic
        }

class Images(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)
    
# 构建场景类
class Scene:
    gaussian_model: GaussianModel
    def __init__(self, sys_param, gaussian_model:GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        self.sys_param = sys_param
        self.gaussian_model = gaussian_model
        # 存储模型路径
        self.model_path = self.sys_param['model_path']
        # 加载已有模型进行训练
        self.loaded_iter = None
        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("加载迭代次数{}的训练模型".format(self.loaded_iter))
        ################## 加载训练和测试数据 ##################
        self.train_cameras = {}
        self.test_cameras = {}
        # 两种格式的数据集:Colmap和Blender
        if os.path.exists(os.path.join(self.sys_param['source_path'], 'sparse')):
            #加载colmap数据集
            scene_info = self.SceneLoadColmap(self.sys_param['source_path'], self.sys_param['images_path'], self.sys_param['depths_path'],
                                              self.sys_param['eval_mode'], self.sys_param['train_test_exp'])
        elif os.path.exists(os.path.join(self.sys_param['source_path'], 'transforms_train.json')):
            # 加载blender格式的数据集
            scene_info = self.SceneLoadBlender(self.sys_param['source_path'], self.sys_param['white_background'],
                                               self.sys_param['depths_path'], self.sys_param['eval_mode'])
        else:
            assert False, "不能读取到数据集"
        ################# 将初始读取数据存到json文件中 ###################
        # # 将 SceneInfo 保存为 JSON
        # output_file = "/home/sulutong/3dgs-slt/scene_info.json"
        # with open(output_file, "w", encoding="utf-8") as f:
        #     json.dump(scene_info_list.to_dict(), f, ensure_ascii=False, indent=4)
        # print("scene_info: ", scene_info)
        ################## 如果没有加载模型，重新初始化相机信息 ##################
        ## 复制点云文件; 生成相机信息文件
        if not self.loaded_iter:
            # 将数据集中的ply点云文件复制到模型路径下，并命名为input.ply
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, 'input.ply'), 'wb') as dest_file:
                dest_file.write(src_file.read())
            # 创建相机列表并保存为cameras.json,将训练集和测试集都合并到其中
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, 'cameras.json'), 'w') as file:
                json.dump(json_cams, file)
        ################## 打乱相机顺序， 确保训练和测试数据的随机性 ##################
        if shuffle:
            random.shuffle(scene_info.train_cameras)
            random.shuffle(scene_info.test_cameras)
        ################## 设置相机尺度信息 ##################
        self.cameras_extent = scene_info.nerf_normalization['radius']
        print("self.cameras_extent: ", self.cameras_extent)
        ###### 加载训练和测试相机数据，按分辨率缩放后的训练相机和测试相机数据 #######
        for resolution_scale in resolution_scales:
            print("加载训练集...")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, sys_param,
                                                                            scene_info.is_nerf_synthetic, False)
            print("加载测试集...")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, sys_param,
                                                                           scene_info.is_nerf_synthetic, True)

        ################## 如果没有加载模型，重新初始化相机信息 ##################
        ################## 初始化点云结构 ##################
        # 加载特定迭代结果
        if self.loaded_iter:
            self.gaussian_model.load_ply(os.path.join(self.model_path, 'point_cloud', 'iteration_' + str(self.loaded_iter), 'point_cloud.ply'),self.sys_param['train_test_exp'])
        # 否则初始化一个点云
        else:
            self.gaussian_model.create_from_pcd(scene_info.point_cloud, scene_info.train_cameras, self.cameras_extent)
            # ######################## 收集所有参数并处理为可序列化的格式 ########################
            # gaussians_list = self.gaussian_model
            # gaussian_params = {}
            # try:
            #     # 检查并提取 Gaussian 的 spatial_lr_scale 属性
            #     if hasattr(gaussians_list, "spatial_lr_scale"):
            #         value = getattr(gaussians_list, "spatial_lr_scale")
            #         if isinstance(value, torch.Tensor):  # 处理 PyTorch Tensor
            #             gaussian_params["spatial_lr_scale"] = value.detach().cpu().numpy().tolist()
            #         elif isinstance(value, torch.nn.Parameter):  # 处理 PyTorch Parameter
            #             gaussian_params["spatial_lr_scale"] = value.detach().cpu().numpy().tolist()
            #         elif isinstance(value, np.ndarray):  # 处理 NumPy 数组
            #             gaussian_params["spatial_lr_scale"] = value.tolist()
            #         else:  # 对于其他类型
            #             gaussian_params["spatial_lr_scale"] = value
            #     else:
            #         print("Gaussian 对象中没有 spatial_lr_scale 属性")
            # except Exception as e:
            #     print(f"无法序列化 spatial_lr_scale 属性：{e}")

            # # 将 spatial_lr_scale 写入 JSON 文件
            # output_file = "/home/sulutong/gaussian-splatting-main/gaussian_spatial_lr_scale_slt.json"
            # with open(output_file, "w") as f:
            #     json.dump(gaussian_params, f, ensure_ascii=False, indent=4)
            # print(f"！！！！调试中！！！！ Gaussian 模型的参数已成功存储到 {output_file}")
            # ####################################################################################
        print("数据读取模块！！！！-----好耶好耶, 数据集读取完毕!")
        
    def getTrainCameras(self, scale = 1.0):
        return self.train_cameras[scale]
    
    def getTestCameras(self, scale = 1.0):
        return self.test_cameras[scale]
            
    # 加载blender数据集
    def SceneLoadBlender(self, source_path, white_background, depths_path, eval, extension = '.png'):
        print("数据读取模块！！！！-----读取blender数据集")
        # 深度文件夹路径
        depths_folder = os.path.join(source_path, depths_path) if depths_path != "" else ""
        print("读取训练集数据")
        train_cam_infos = self.readCamerasFromTransforms(source_path, 'transforms_train.json', depths_folder, white_background, False, extension)
        print("读取测试集数据")
        test_cam_infos = self.readCamerasFromTransforms(source_path, 'transforms_test.json', depths_folder, white_background, True, extension)
        # 判断是否为测试模式，如果不是，将测试集也放入训练集
        if not eval:
            train_cam_infos.extend(test_cam_infos)
            test_cam_infos = []
        # 相机归一化处理
        nerf_normalization = self.getNerfppNorm(train_cam_infos)
        ply_path = os.path.join(source_path, "points3d.ply")
        if not os.path.exists(ply_path):
            # 如果在指定路径下没有找到.ply格式的点云文件，代码会生成一个随机的点云数据，点数为100,000
            num_pts = 100_000
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
    def readCamerasFromTransforms(self, source_path, transformsfile, depths_folder, white_background, is_test, extension = '.png'):
        cam_infos = []
        with open(os.path.join(source_path, transformsfile)) as json_file:
            jsons = json.load(json_file)
            # 相机视场角
            fovx = jsons['camera_angle_x']
            frames = jsons['frames']
            for idx, frame in enumerate(frames):
                # 图像名称
                cam_name = os.path.join(source_path, frame["file_path"] + extension)
                # cam_name = frame['file_path'] + extension
                # 相机外参矩阵
                c2w = np.array(frame['transform_matrix'])
                # 坐标系转换，从OpenGL/Blender坐标系(Y up, Z back)转换到COLMAP坐标系(Y down, Z forward)
                c2w[:3, 1:3] *= -1
                w2c = np.linalg.inv(c2w)
                R = np.transpose(w2c[:3, :3])
                T = w2c[:3, 3]
                # 图像路径
                image_path = os.path.join(source_path, cam_name)
                # 提取不带扩展名的图像名称
                image_name = Path(image_path).stem
                image = Image.open(image_path)
                # 处理图像数据
                # 图像转换成RGBA格式
                image_data = np.array(image.convert("RGBA"))
                # 设置背景色
                bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])
                # 像素归一化
                image_norm = image_data / 255.0
                # 混合背景色
                arr = image_norm[:, :, :3] * image_norm[:, :, 3:4] + bg * (1 - image_norm[:, :, 3:4])
                # 将像素范围返回0-255， 并转换为RGB图像
                image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), 'RGB')

                # 利用fovx得到fovy
                fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
                FoVx = fovx
                FoVy = fovy
                depth_path = os.path.join(depths_folder, f'{image_name}.png') if depths_folder != "" else ""
                cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FoVy=FoVy, FoVx=FoVx, depth_params=None,
                                      image_path=image_path ,image_name=image_name, depth_path=depth_path,
                                      width=image.size[0], height=image.size[1], is_test=is_test))
        return cam_infos

    # 加载colmap数据集
    """
    source_path: colmap生成场景数据所在文件夹路径
    images_path: 存放图像的路径
    depths_path: 存放深度图文件的路径
    eval: 是否开启评估模式
    train_test_exp: 是否划分训练集和测试集,若为False则全部为训练集
    llffhold: 如果 eval 为 True 且 llffhold 被指定，则用于每隔 llffhold 个相机采样一个测试相机
    """
    def SceneLoadColmap(self, source_path, images_path, depths_path, eval, train_test_exp, llffhold=8):
        ########## 读取相机的内外参数 ##########
        print("数据读取模块！！！！-----读取colamp数据集")
        # 如果没有二进制文件，则读取txt文件
        try:
            cameras_extrinsic_file = os.path.join(source_path, 'sparse/0', 'images.bin')
            cameras_intrinsic_file = os.path.join(source_path, 'sparse/0', 'cameras.bin')
            cam_extrinsics = self.read_extrinsics_binary(cameras_extrinsic_file)
            cam_intrinsics = self.read_intrinsics_binary(cameras_intrinsic_file)
            print("数据读取模块！！！！-----二进制文件中外参信息(cam_extrinsics): ", len(cam_extrinsics))
            print("数据读取模块！！！！-----二进制文件中内参信息(cam_intrinsics): ", len(cam_intrinsics))
        except:
            cameras_extrinsic_file = os.path.join(source_path, 'sparse/0', 'images.txt')
            cameras_intrinsic_file = os.path.join(source_path, 'sparse/0', 'cameras.txt')
            cam_extrinsics = self.read_extrinsics_text(cameras_extrinsic_file)
            cam_intrinsics = self.read_intrinsics_text(cameras_intrinsic_file)
            print("数据读取模块！！！！-----txt文件中外参信息(cam_extrinsics): ", len(cam_extrinsics))
            print("数据读取模块！！！！-----txt文件中内参信息(cam_intrinsics): ", len(cam_intrinsics))
        ########## 加载和处理深度参数 ##########
        depths_params_file = os.path.join(source_path, 'sparse/0', 'depth_params.json')
        depths_params = None
        if depths_path != "":
            print("数据读取模块-----depths_path存在!!!!, 加载深度参数")
            try:
                with open(depths_params_file, 'r') as f:
                    depths_params = json.load(f)
                    # 计算中位数缩放值, 首先提取所有缩放值
                all_scales = np.array([depths_params[key]["scale"] for key in depths_params])
                # 检查是否有大于0的缩放值，如果有计算中位值
                # all_scales > 0返回一个含有True和False的数组，.sum()计算true的个数
                if (all_scales > 0).sum():
                    med_scale = np.median(all_scales[all_scales > 0])
                else:
                    med_scale = 0
                for key in depths_params:
                    depths_params[key]['scale'] = med_scale
            except FileNotFoundError:
                print(f'深度信息文件未找到！')
            except Exception as e:
                print(f"An unexpected error occurred when trying to open depth_params.json file: {e}")
                sys.exit(1)
        ########## 选择测试相机 ##########
        if eval:
            print('数据读取模块-----选择测试相机')
            if '360' in source_path:
                llffhold = 8
            if llffhold:
                print("-----------------LLFF HOLD----------------")
                # 提取字典中每张图像的名称，并生成名称列表
                cam_names = [cam_extrinsics[cam_id].name for cam_id in cam_extrinsics]
                cam_names = sorted(cam_names)
                # 每隔llffhold个进行抽样
                test_cam_names_list = [name for idx, name in enumerate(cam_names) if idx % llffhold == 0]
            else:
                # 从文件夹中读取测试相机
                with open(os.path.join(source_path, 'sparse/0', 'test.txt'), 'r') as file:
                    test_cam_names_list = [line.strip() for line in file]
        else:
            test_cam_names_list = []
        ########## 读取相机信息，划分训练和测试相机 ##########
        reading_dir = "images" if images_path == None else images_path
        # 将相机内外参数合并在一起
        cam_infos_unsorted = self.readColmapCameras(
            cam_extrinsics, cam_intrinsics, depths_params,
            images_folder = os.path.join(source_path, reading_dir),
            depths_folder = os.path.join(source_path, depths_path) if depths_path != "" else "",
            test_cam_name_list= test_cam_names_list
        )
        cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x: x.image_name)
        print("数据读取模块！！！！-----相机内外参数合并: ", len(cam_infos))
        # 划分训练集和测试集
        train_cam_infos = [c for c in cam_infos if train_test_exp or not c.is_test]
        test_cam_infos = [c for c in cam_infos if c.is_test]
        print("数据读取模块！！！！-----训练集相机: ", len(train_cam_infos))
        print("数据读取模块！！！！-----测试集相机: ", len(test_cam_infos))
        ########## 得到归一化相机数据 ##########
        nerf_normalization = self.getNerfppNorm(train_cam_infos)
        ########## 加载或生成点云数据 ##########
        ply_path = os.path.join(source_path, "sparse/0/points3D.ply")
        bin_path = os.path.join(source_path, "sparse/0/points3D.bin")
        txt_path = os.path.join(source_path, "sparse/0/points3D.txt")
        # print("ply_path: ", ply_path)
        # 检查是否存在ply文件，如果不存在利用bin或者txt文件生成ply文件
        if not os.path.exists(ply_path):
            print("数据读取模块！！！！-----点云ply文件不存在! 坏耶!")
            try:
                xyz, rgb, _ = self.read_points3D_binary(bin_path)
            except:
                xyz, rgb, _ = self.read_points3D_text(txt_path)
            # 生成一个ply文件
            self.storePly(ply_path, xyz, rgb)
        try:
            print("数据读取模块！！！！-----点云ply文件存在! 好耶!")
            # 提取ply点云信息
            pcd = self.fetchPly(ply_path)
            print("数据读取模块！！！！-----初始点云信息pcd: ")
        except:
            pcd = None
        pcd_list = pcd
        ######################## 将初始读取数据存到json文件里 ########################
        # def camera_info_to_dict(camera):
        #     """
        #     将 CameraInfo 对象转换为字典，并确保字段可序列化
        #     """
        #     return {
        #         "uid": camera.uid,
        #         "R": camera.R.tolist() if isinstance(camera.R, np.ndarray) else camera.R,
        #         "T": camera.T.tolist() if isinstance(camera.T, np.ndarray) else camera.T,
        #         "FovY": camera.FoVy.tolist() if isinstance(camera.FoVy, np.ndarray) else camera.FoVy,
        #         "FovX": camera.FoVx.tolist() if isinstance(camera.FoVx, np.ndarray) else camera.FoVx,
        #         "depth_params": camera.depth_params,
        #         "image_path": camera.image_path,
        #         "image_name": camera.image_name,
        #         "depth_path": camera.depth_path,
        #         "width": camera.width,
        #         "height": camera.height,
        #         "is_test": camera.is_test
        #     }
        # train_cam_infos_list = [camera_info_to_dict(c) for c in cam_infos if train_test_exp or not c.is_test]
        # test_cam_infos_list = [camera_info_to_dict(c) for c in cam_infos if c.is_test]
        # 生成一个场景字典
        scene_info = SceneInfo(point_cloud=pcd,
                               train_cameras=train_cam_infos,
                               test_cameras=test_cam_infos,
                               nerf_normalization=nerf_normalization,
                               ply_path=ply_path,
                               is_nerf_synthetic=False)
        # scene_info_list = SceneInfo(point_cloud=pcd_list,
        #                        train_cameras=train_cam_infos_list,
        #                        test_cameras=test_cam_infos_list,
        #                        nerf_normalization=nerf_normalization,
        #                        ply_path=ply_path,
        #                        is_nerf_synthetic=False)
        # return scene_info, scene_info_list
        return scene_info

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
                cameras[camera_id] = Camera(
                    id = camera_id,
                    model = model_name,
                    width = width,
                    height = height,
                    params = np.array(intr_params)
                )
        return cameras

    # 从二进制文件中读取指定数量的字节
    """
    fid: 二进制文件的文件对象
    num_bytes: 指定要从文件中读取的字节数
    format_char_sequence: 告诉 struct.unpack 如何解包这些字节
        c:单个字符
        e:2 字节的字符
        f:4 字节的浮动点数
        d:8 字节的双精度浮动点数
        h:2 字节的有符号短整型
        H:2 字节的无符号短整型
        i:4 字节的有符号整型
        I:4 字节的无符号整型
        l:4 字节的有符号长整型
        L:4 字节的无符号长整型
        q:8 字节的有符号长整型
        Q:8 字节的无符号长整型
    endian_character: 指定字节顺序,它控制数据的存储方式是大端还是小端
        "@":默认本机字节顺序(与平台相关)。
        "=":标准字节顺序，意味着数据在内存中的存储格式是平台无关的。
        "<":小端字节序(Little Endian)。
        ">":大端字节序(Big Endian)。
        "!":网络字节顺序(即大端)。
    """
    def read_next_bytes(self, fid, num_bytes, format_char_sequence, endian_character = "<"):
        data = fid.read(num_bytes)
        # 返回解包数据，一个元组
        return struct.unpack(endian_character + format_char_sequence, data)
    
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

    # 合并相机的内外参数信息
    def readColmapCameras(self, cam_extrinsics, cam_intrinsics, depths_params, images_folder, depths_folder, test_cam_name_list):
        cam_infos = []
        for idx, key in enumerate(cam_extrinsics):
            sys.stdout.write('\r')
            sys.stdout.write('Reading camera {}/{}'.format(idx + 1, len(cam_extrinsics)))
            sys.stdout.flush()

            # 获取相机外参字典
            extr = cam_extrinsics[key]
            # 获取相机内参字典
            intr = cam_intrinsics[extr.camera_id]
            # 相机分辨率
            height = intr.height
            width = intr.width
            # 相机id
            uid = intr.id
            # 该图片对应的相机外参
            R = np.transpose(qvec2rotmat(extr.qvec))
            T = np.array(extr.tvec)
            # 相机焦距
            if intr.model=="SIMPLE_PINHOLE":
                focal_length_x = intr.params[0]
                FovY = focal2fov(focal_length_x, height)
                FovX = focal2fov(focal_length_x, width)
            elif intr.model=="PINHOLE":
                focal_length_x = intr.params[0]
                focal_length_y = intr.params[1]
                FovY = focal2fov(focal_length_y, height)
                FovX = focal2fov(focal_length_x, width)
            else:
                assert False, '无法处理相机模型'
            # 计算图片文件扩展名长度
            n_remove = len(extr.name.split('.')[-1]) + 1
            depth_params = None
            if depths_params is not None:
                try:
                    depth_params = depths_params[extr.name[:n_remove]]
                except:
                    print('\n', key, '没有找到深度信息参数')
            # 图片路径
            image_path = os.path.join(images_folder, extr.name)
            image_name = extr.name
            depth_path = os.path.join(depths_folder, f"{extr.name[:n_remove]}.png") if depths_folder != "" else ""
            # 将该图片所有信息存入字典
            cam_info = CameraInfo(
                uid=uid, R=R, T=T, FoVy=FovY, FoVx=FovX, depth_params=depth_params, image_path=image_path,
                image_name=image_name, depth_path=depth_path, width=width, height=height, is_test=image_name in test_cam_name_list
            )
            cam_infos.append(cam_info)
        sys.stdout.write('\n')
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
                 
    # 从点云文本文件中读取信息
    def read_points3D_text(self, txt_path):
        # 得到点云数量
        xyzs = None
        rgbs = None
        errors = None
        num_points = 0
        with open(txt_path, "r") as fid:
            while True:
                line = fid.readline()
                if not line:
                    break
                line = line.strip()
                if len(line) > 0 and line[0] != "#":
                    num_points += 1
        # 构造点云坐标、RGB、误差矩阵
        xyzs = np.empty((num_points, 3))
        rgbs = np.empty((num_points, 3))
        errors = np.empty((num_points, 1))
        count = 0
        with open(txt_path, "r") as fid:
            while True:
                line = fid.readline()
                if not line:
                    break
                line = line.strip()
                if len(line) > 0 and line[0] != "#":
                    elems = line.split()
                    xyz = np.array(tuple(map(float, elems[1:4])))
                    rgb = np.array(tuple(map(int, elems[4:7])))
                    error = np.array(float(elems[7]))
                    xyzs[count] = xyz
                    rgbs[count] = rgb
                    errors[count] = error
                    count += 1

        return xyzs, rgbs, errors

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
