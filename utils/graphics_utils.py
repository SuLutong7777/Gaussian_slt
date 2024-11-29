##################### 定义了点云投影变换的相关函数 ######################
## BasicPointCloud：定义了点云类
## fov2focal(fov, pixels)：视场角转焦距函数
## focal2fov(focal, pixels)：焦距转视场角函数
## getWorld2View2：计算世界坐标系 (World) 到视图坐标系 (View) 的变换矩阵，还允许对相机位置进行平移 (translate) 和缩放 (scale) 的调整
## getProjectionMatrix(znear, zfar, fovX, fovY)：计算相机的投影矩阵，从3D坐标转换到2D坐标
## qvec2rotmat(qvec)：四元数转为旋转矩阵
## rotmat2qvec(R)：旋转矩阵转为四元数
########################################################################

import math
import numpy as np
import torch
from typing import NamedTuple

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

    def to_dict(self):
        """
        将 BasicPointCloud 转换为字典以支持 JSON 序列化
        """
        return {
            "points": self.points.tolist() if self.points is not None else None,
            "colors": self.colors.tolist() if self.colors is not None else None,
            "normals": self.normals.tolist() if self.normals is not None else None
        }

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

# 计算世界坐标系 (World) 到视图坐标系 (View) 的变换矩阵 
# 还允许对相机位置进行平移 (translate) 和缩放 (scale) 的调整
def getWorld2View2(R, T, translate=np.array([.0, .0, .0]), scale=1.0):
    # 构造W2C
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = T
    Rt[3, 3] = 1.0
    # 调整相机中心
    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

# 相机的投影矩阵，从3D坐标转换到2D坐标
def getProjectionMatrix(znear, zfar, fovX, fovY):
    # 利用视场角和裁剪平面计算视锥边界
    tanHalfFovX = math.tan(fovX / 2)
    tanHalfFovY = math.tan(fovY / 2)
    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right
    # 初始化投影矩阵
    P = torch.zeros(4, 4)
    z_sign = 1.0
    # 填充投影矩阵
    P[0, 0] = 2.0 * znear / (right - left)
    P[0, 2] = (right + left) / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    P[3, 2] = z_sign
    return P

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec