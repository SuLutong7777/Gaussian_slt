import torch
import numpy as np
import open3d as o3d

class Visiual_All_Data():
    def __init__(self):
        # 显示窗口的尺寸
        self.window_h = 720
        self.window_w = 1080
        # self.background_color = [1.0, 1.0, 1.0]
        self.background_color = [0.0, 0.0, 0.0]
        
    # # 初始化绘图的模板
    # def setting_show_window(self):
    #     print("aaa")
    #     vis = o3d.visualization.VisualizerWithKeyCallback()
    #     print("bbb")
    #     vis.create_window(width=self.window_w, height=self.window_h, visible=False)
    #     print("ccc")
    #     render_option = vis.get_render_option()
    #     print("ddd")
    #     render_option.background_color = np.array(self.background_color)
    #     print("eee")
    #     return vis
    
    # 初始化绘图的模板
    def setting_show_window(self):
        print("aaa")
        # 用普通 Visualizer，避免 GUI 回调模式
        vis = o3d.visualization.Visualizer()
        print("bbb")
        vis.create_window(
            width=self.window_w,
            height=self.window_h,
            visible=False  # 离屏渲染
        )
        print("ccc")
        render_option = vis.get_render_option()
        print("ddd")
        render_option.background_color = np.array(self.background_color, dtype=np.float32)
        print("eee")
        return vis

    # 保存并关闭窗口（新增）
    def save_and_close(self, vis_window, save_path):
        vis_window.poll_events()
        vis_window.update_renderer()
        # 保存屏幕截图
        o3d.io.write_image(save_path, vis_window.capture_screen_float_buffer(False))
        vis_window.destroy_window()
        print(f"保存可视化结果到: {save_path}")

    # c2w_rt:[N, 3, 4]或者[N, 4, 4]
    # 把相机可视化出来
    def draw_c2w_in_space(self, c2w_rt, intrs=None, scale=0.01, cam_color=[0, 1, 0], vis_window=None):
        # 初始化显示界面，如果不指定，默认不共用，独立创建
        if vis_window is None:
            print("111")
            vis_window = self.setting_show_window()
        else:
            print("222")
            vis_window = vis_window
        # tensor 类型转换
        print("333")
        if isinstance(c2w_rt, torch.Tensor):
            c2w_rt = c2w_rt.detach().cpu().numpy()
        for ii, c2w in enumerate(c2w_rt):
            if c2w.shape[0] == 3:
                c2w = np.vstack([c2w, np.array([0, 0, 0, 1])])
            assert c2w.shape == (4, 4), "RT matrix must be 4x4"

            # 获取相机模型几何体
            if intrs is None:
                geoms = self.draw_camera(c2w, color=cam_color, show_axis=True)
            else:
                geoms = self.draw_camera(
                    c2w,
                    intrs[ii][0, 2] * 2 * scale,
                    intrs[ii][1, 2] * 2 * scale,
                    intrs[ii][0, 0] * scale,
                    color=cam_color,
                    show_axis=True
                )
            # 添加到窗口
            for g in geoms:
                vis_window.add_geometry(g)

        print("555")
        return vis_window

    # # c2w_rt:[N, 3, 4]或者[N, 4, 4]
    # # 把相机可视化出来
    # def draw_c2w_in_space(self, c2w_rt, intrs=None, scale=0.01, cam_color=[0, 1, 0], vis_window=None):
    #     # 初始化显示界面，如果不指定，默认不共用，独立创建
    #     if vis_window is None:
    #         print("111")
    #         vis_window = self.setting_show_window()
    #     else:
    #         print("222")
    #         vis_window = vis_window
    #     # tensor 类型转换
    #     print("333")
    #     if isinstance(c2w_rt, torch.Tensor):
    #         c2w_rt = c2w_rt.detach().cpu().numpy()
    #     for ii, c2w in enumerate(c2w_rt):
    #         # print("444")
    #         # 保证 4x4
    #         if c2w.shape[0] == 3:
    #             c2w = np.vstack([c2w, np.array([0, 0, 0, 1])])
    #         assert c2w.shape == (4, 4), "RT matrix must be 4x4"

    #         # 获取相机模型几何体
    #         if intrs is None:
    #             geoms = self.draw_camera(c2w, color=cam_color, show_axis=True)
    #         else:
    #             geoms = self.draw_camera(
    #                 c2w,
    #                 intrs[ii][0, 2] * 2 * scale,
    #                 intrs[ii][1, 2] * 2 * scale,
    #                 intrs[ii][0, 0] * scale,
    #                 color=cam_color,
    #                 show_axis=True
    #             )
    #         # 添加到窗口
    #         for g in geoms:
    #             vis_window.add_geometry(g)

    #     print("555")
    #     return vis_window
        # # 绘图过程
        # geometries = []
        # for ii, c2w in enumerate(c2w_rt):
        #     print("444")
        #     # 保证4x4矩阵
        #     if c2w.shape[0] == 3:                
        #         c2w = np.vstack([c2w, np.array([0, 0, 0, 1])])
        #     assert c2w.shape == (4, 4), "RT matrix must be 4x4"
            
        #     # 获取绘制的相机和坐标轴
        #     if intrs is None:
        #         geometries.extend(self.draw_camera(c2w,
        #                                            color=cam_color, 
        #                                            show_axis=True))
        #     else:
        #         geometries.extend(self.draw_camera(c2w,
        #                                            intrs[ii][0, 2]*2*scale,
        #                                            intrs[ii][1, 2]*2*scale,
        #                                            intrs[ii][0, 0]*scale,
        #                                            color=cam_color,
        #                                            show_axis=True))
        # print("555")
        # # 在 Open3D 窗口中添加几何体
        # for geometry in geometries:
        #     vis_window.add_geometry(geometry)

        # return vis_window
    
    # 绘制相机外形
    def draw_camera(self, c2w, cam_width=0.32/2, cam_height=0.24/2, f=0.10, color=[0, 1, 0], show_axis=True):
        points = [[0, 0, 0], [-cam_width, -cam_height, f], [cam_width, -cam_height, f],
                [cam_width, cam_height, f], [-cam_width, cam_height, f]]
        lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
        colors = [color for i in range(len(lines))]

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        line_set.transform(c2w)

        if show_axis:
            axis = o3d.geometry.TriangleMesh.create_coordinate_frame()
            axis.scale(min(cam_width, cam_height), np.array([0., 0., 0.]))
            axis.transform(c2w)
            return [line_set, axis]
        else:
            return [line_set]

    # 在空间中绘制三维点，默认白色
    # pts: [B, numb, 3]
    # color: [B, numb, 3]
    # mask: [B, numb]
    # sample_down: 下采样倍数
    def draw_pts_in_space(self, pts, color=None, mask=None, sample_down=None, vis_window=None):

        # 初始化显示窗口（如果未提供现有窗口）
        if vis_window is None:
            vis_window = self.setting_show_window()        
        
        # 默认是红色点
        if color is None:
            color = np.zeros_like(pts)
            color[..., 0] = 1.0

        # 默认所有点都有效
        if mask is None:
            mask = np.ones_like(pts)[..., 0].astype(bool)

        if sample_down is not None:
            pts = pts[:, ::sample_down, :]
            color = color[:, ::sample_down, :]
            mask = mask[:, ::sample_down]
        
        final_pts = pts.reshape(-1, 3)[mask.reshape(-1)]
        final_colors = color.reshape(-1, 3)[mask.reshape(-1)]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(final_pts)
        pcd.colors = o3d.utility.Vector3dVector(final_colors)
        
        # 添加点云到可视化窗口
        vis_window.add_geometry(pcd)

        return vis_window

    # 在空间中绘制方向向量，默认白色
    # vec_d: [B, numb, 3]
    # vec_o: [B, numb, 3]
    # mask: [B, numb]
    # sample_down: 下采样倍数    
    def draw_vec_in_space(self, vec_d, vec_o, length=0.1, color=None, mask=None, sample_down=None, vis_window=None):
        # 初始化显示窗口（如果未提供现有窗口）
        if vis_window is None:
            vis_window = self.setting_show_window()        

        # 默认是红色
        if color is None:
            color = np.zeros_like(vec_d)
            color[..., 0] = 1.0

        # 默认所有点都有效
        if mask is None:
            mask = np.ones_like(vec_d)[..., 0].astype(bool)

        if sample_down is not None:
            vec_d = vec_d[:, ::sample_down, :]
            vec_o = vec_o[:, ::sample_down, :]
            color = color[:, ::sample_down, :]
            mask = mask[:, ::sample_down]
        
        # [N, 3]
        vec_d = vec_d.reshape(-1, 3)[mask.reshape(-1)]
        vec_o = vec_o.reshape(-1, 3)[mask.reshape(-1)]
        colors = color.reshape(-1, 3)[mask.reshape(-1)]

        # vec_d = vec_d.reshape(-1, 3)[mask.reshape(-1)][:5]
        # vec_o = vec_o.reshape(-1, 3)[mask.reshape(-1)][:5]
        # colors = color.reshape(-1, 3)[mask.reshape(-1)][:5]
        
        # 终点计算
        start_pts = vec_o
        end_pts = start_pts + length * vec_d

        # 组合所有点
        pts = np.vstack((start_pts, end_pts))  # (2N, 3)

        # 生成线索引，每条线连接 vec_o[i] -> end_pts[i]
        lines = [[i, i + len(start_pts)] for i in range(len(start_pts))]

        # 创建 LineSet
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(pts)  # 设置点
        line_set.lines = o3d.utility.Vector2iVector(lines)    # 设置连线
        line_set.colors = o3d.utility.Vector3dVector(colors)  # 颜色

        vis_window.add_geometry(line_set)


        return vis_window

    # 显示所有绘制结果
    def show_window(self, vis_window=None):
        if vis_window is None:
            # 开始渲染
            self.vis_window.run()
            self.vis_window.destroy_window()
        else:
            vis_window.run()
            vis_window.destroy_window()

    # 计算两个矩阵间的转换矩阵(用第一个相机参数)
    def c2w_alignment_single(self, fix_c2w, trans_c2w):
        # 提取旋转矩阵和位移向量
        R_fix, T_fix = fix_c2w[:3, :3], fix_c2w[:3, 3]
        R_trans, T_trans = trans_c2w[:3, :3], trans_c2w[:3, 3]
        # 计算从B到A的转换矩阵
        # R_B_inv = R_trans.T  # 旋转矩阵的逆是转置
        R_B_inv = torch.linalg.inv(R_trans)
        T_B_inv = -R_B_inv @ T_trans  # 计算平移部分
        # 组合转换矩阵
        RT_B_to_A = torch.eye(4)
        RT_B_to_A[:3, :3] = R_fix @ R_B_inv
        RT_B_to_A[:3, 3] = R_fix @ T_B_inv + T_fix
        
        return RT_B_to_A        

    # 计算两个矩阵间的转换矩阵(用多个相机参数)
    def c2w_alignment(self, fix_c2ws, trans_c2ws):
        # 提取位移向量 [N, 3]
        T_fixs, T_trans = fix_c2ws[:, :3, -1], trans_c2ws[:, :3, -1]
        cam_num = T_fixs.shape[0]
        # 计算协方差矩阵
        C = 1.0 / cam_num * (T_trans.T @ T_fixs)
        # SVD分解
        U_svd, D_svd, V_svd_T = torch.linalg.svd(C)
        # 求解S矩阵
        S = torch.eye(3)
        if(torch.det(U_svd)*torch.det(V_svd_T) < 0):
            S[2, 2] = -1
        # 求解转换矩阵
        R = U_svd @ S @ V_svd_T
        # 求解平移矩阵
        mu_trans = T_trans.mean(0)
        mu_fixs = T_fixs.mean(0)
        # print("mu_trans, mu_fixs: ", mu_trans, mu_fixs)
        t = mu_trans - R @ mu_fixs
        # 组合转换矩阵
        RT_B_to_A = torch.eye(4)
        RT_B_to_A[:3, :3] = R
        RT_B_to_A[:3, 3] = t
        return RT_B_to_A

    # 使用转换矩阵，对另一组RT进行变换
    def c2w_transform(self, trans_mat, trans_c2w):
        trans_mat_expd = trans_mat[None].repeat(trans_c2w.shape[0], 1, 1)
        target_c2w = trans_mat_expd @ trans_c2w
        return target_c2w

    # 绘制相机间的对应关系
    def show_camera_correspondences(self, rt1, rt2, vis_window=None, color=[1, 0, 0]):

        # 确保两组RT的形状一致
        assert rt1.shape == rt2.shape, "两组RT矩阵的形状必须一致"
        N = rt1.shape[0]

        # 初始化可视化窗口（如果未提供）
        if vis_window is None:
            vis_window = self.setting_show_window()

        # 转换为 NumPy 数组
        if isinstance(rt1, torch.Tensor):
            rt1 = rt1.detach().cpu().numpy()
        if isinstance(rt2, torch.Tensor):
            rt2 = rt2.detach().cpu().numpy()

        # 提取相机位置（中心点）
        centers1 = rt1[:, :3, 3]  # 第一组相机位置 [N, 3]
        centers2 = rt2[:, :3, 3]  # 第二组相机位置 [N, 3]

        # 绘制每对相机的位置连线
        line_points = []
        line_indices = []
        for i in range(N):
            line_points.append(centers1[i])  # 真值位置
            line_points.append(centers2[i])  # 预测位置
            line_indices.append([2 * i, 2 * i + 1])  # 每对的索引

        # 转换为 Open3D 的几何体
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(line_points)  # 设置点坐标
        line_set.lines = o3d.utility.Vector2iVector(line_indices)  # 设置线索引
        line_set.colors = o3d.utility.Vector3dVector([color] * len(line_indices))  # 设置线颜色

        # 添加到可视化窗口
        vis_window.add_geometry(line_set)

        return vis_window
