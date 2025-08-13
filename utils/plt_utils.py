import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 初始化画布
def init_show_figure(show_info=True, ax_lim=True):
    all_fig = plt.figure(figsize=(9, 10))
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['mathtext.default'] = 'regular'
    ax = Axes3D(all_fig, auto_add_to_figure=False)
    # ax.view_init(elev=45, azim=60, roll=20)
    all_fig.add_axes(ax)
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    
    if ax_lim:
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
        ax.set_zlim(-1.0, 1.0)

    if show_info:
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        ax.set_zlabel("Z Axis")
        # ax.grid(False)
        # ax.axis(False)
        # ax.set_xlim(-1.0, 1.0)
        # ax.set_ylim(-1.0, 1.0)
        # ax.set_zlim(-1.0, 1.0)
    else:
        ax.grid(False)
        ax.axis(False)
    
    plt.ion()
    plt.gca().set_box_aspect((1, 1, 1))

    return ax

def draw_space_lines(c2w_mat, ax, line_w=0.5):
    pts_3d = np.array(c2w_mat[:, :3, 3].detach().cpu()) # [N, 3]
    ax.plot(pts_3d[:, 0], pts_3d[:, 1], pts_3d[:, 2], linestyle='--', color='red', marker='o', label='Dashed Line', linewidth=line_w)

# RT_mat:[N, 4, 4]
def show_camera_position(intr_mat, RT_mat, img_w, fig_ax, color = (0.7,0.2,0.7), cam_size=0.025, line_w=0.5, auto_axis_range=False):

    # # 判断一下路径是否存在
    # save_path = Path("/home/gaoyu/MC2_ZipNeRF_Real")
    # os.makedirs(save_path, exist_ok=True)
    # plt.cla()

    # color_pd = (0,0.6,0.7)
    # fig_ax.set_xlim(-1., 1.)
    # fig_ax.set_ylim(-1., 1.)
    # fig_ax.set_zlim(0., 1.)
    # [84, 3, 4]
    clip_pose = RT_mat[:, :3, :]
    
    
    draw_camera_shape(clip_pose, intr_mat, color, img_w, fig_ax, cam_size, line_w)

    # if auto_axis_range:
    #     # 计算每个相机中心在世界坐标系下的位置
    #     cam_centers = []
    #     for i in range(0, len(RT_mat), step):
    #         R = RT_mat[i, :, :3]
    #         T = RT_mat[i, :, 3]
    #         cam_center = -R.T @ T  # 世界坐标下相机中心
    #         cam_centers.append(cam_center)
    #     cam_centers = np.array(cam_centers)

    #     # 自动设置坐标轴范围
    #     min_vals = cam_centers.min(axis=0)
    #     max_vals = cam_centers.max(axis=0)
    #     padding = 0.1 * (max_vals - min_vals + 1e-6)

    #     fig_ax.set_xlim(min_vals[0] - padding[0], max_vals[0] + padding[0])
    #     fig_ax.set_ylim(min_vals[1] - padding[1], max_vals[1] + padding[1])
    #     fig_ax.set_zlim(min_vals[2] - padding[2], max_vals[2] + padding[2])
    # else:
    #     fig_ax.set_xlim(-2., 0.)
    #     fig_ax.set_ylim(0., 2.)
    #     fig_ax.set_zlim(2., 5.)

    # fig_ax.set_xlabel("X")
    # fig_ax.set_ylabel("Y")
    # fig_ax.set_zlabel("Z")
    # fig_ax.set_title("Camera Positions")

# opengl formate camera coord
# extr_mat：[C2W]
def draw_camera_shape(extr_mat, intr_mat, color, img_w, ax, cam_size=0.25, line_w=0.5):
    # extr_mat: [84, 3, 4]
    # intr_mat: [84, 3, 3]
    cam_line = cam_size
    focal = intr_mat[:,0,0]*cam_line/img_w
    
    cam_pts_1 = torch.stack([-torch.ones_like(focal)*cam_line/2,
                             -torch.ones_like(focal)*cam_line/2,
                             focal], -1)[:,None,:].to(extr_mat.device)
    cam_pts_2 = torch.stack([-torch.ones_like(focal)*cam_line/2,
                             torch.ones_like(focal)*cam_line/2,
                             focal], -1)[:,None,:].to(extr_mat.device)
    cam_pts_3 = torch.stack([torch.ones_like(focal)*cam_line/2,
                             torch.ones_like(focal)*cam_line/2,
                             focal], -1)[:,None,:].to(extr_mat.device)
    cam_pts_4 = torch.stack([torch.ones_like(focal)*cam_line/2,
                             -torch.ones_like(focal)*cam_line/2,
                             focal], -1)[:,None,:].to(extr_mat.device)
    
    cam_pts_1 = cam_pts_1 @ extr_mat[:, :3, :3].transpose(-2,-1) + extr_mat[:, :3, 3][:,None,:]
    cam_pts_2 = cam_pts_2 @ extr_mat[:, :3, :3].transpose(-2,-1) + extr_mat[:, :3, 3][:,None,:]
    cam_pts_3 = cam_pts_3 @ extr_mat[:, :3, :3].transpose(-2,-1) + extr_mat[:, :3, 3][:,None,:]
    cam_pts_4 = cam_pts_4 @ extr_mat[:, :3, :3].transpose(-2,-1) + extr_mat[:, :3, 3][:,None,:]
    
    # [N, 5, 3]
    cam_pts = torch.cat([cam_pts_1, cam_pts_2, cam_pts_3, cam_pts_4, cam_pts_1], dim=-2)

    for i in range(4):
        # [84, 2, 3]
        cur_line_pts = torch.stack([cam_pts[:,i,:], cam_pts[:,i+1,:]], dim=-2).to('cpu')
        for each_cam in cur_line_pts:
            ax.plot(each_cam[:,0],each_cam[:,1],each_cam[:,2],color=color,linewidth=line_w)
    extr_T = extr_mat[:, :3, 3]
    
    for i in range(4):
        # [84, 2, 3]
        cur_line_pts = torch.stack([extr_T, cam_pts[:,i,:]], dim=-2).to('cpu')
        for each_cam in cur_line_pts:
            ax.plot(each_cam[:,0],each_cam[:,1],each_cam[:,2],color=color,linewidth=line_w)
    extr_T = extr_T.to('cpu')

    ax.scatter(extr_T[:,0],extr_T[:,1],extr_T[:,2],color=color,s=5) 
    ax.set_aspect('equal')   

    