import argparse
import math
import os
import uuid
import setproctitle
import logging
import sys
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from random import randint
from config import Load_config
from dataload import Scene
from model import GaussianModel
from utils.general_utils import get_expon_lr_func, safe_state
from utils.sh_utils import eval_sh
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

try:
    from fused_ssim import fused_ssim # type: ignore
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

def train(sys_param, gaussian_model: GaussianModel, scene: Scene):
    # 检查是否安装了sparse_adam的库，如果没有安装且优化器要求，退出并给出提示
    if not SPARSE_ADAM_AVAILABLE and sys_param['optimizer_type'] == 'sparse_adam':
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")
    # 记录训练起始迭代数
    first_iter = 0
    # 准备输出路径和日志记录器
    tb_writer = prepare_output_and_logger(sys_param)
    # 初始化模型训练的优化器以及学习率等
    gaussian_model.training_setup(sys_param)
    # 在模型恢复时加载检查点，加载存储在检查点的模型参数
    if sys_param['start_checkpoint']:
        print("训练模块！！！！-----从某个检查点恢复模型数据")
        (model_params, first_iter) = torch.load(sys_param['start_checkpoint'])
        gaussian_model.restore(model_params, sys_param)
    else:
        print("训练模块！！！！-----从头训练模型！")
    # 设置背景色并移动至GPU上
    bg_color = [1, 1, 1] if sys_param['white_background'] else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device='cuda')
    # print("背景色background: ", background)
    # 创建两个 CUDA 事件 iter_start、iter_end，用于记录训练迭代的开始时间和结束时间，允许使用时间跟踪
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    use_sparse_adam = sys_param['optimizer_type'] == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    # 生成一个学习率调整函数，具体控制深度 L1 损失的权重（depth_l1_weight）随训练迭代的变化
    depth_l1_weight = get_expon_lr_func(lr_init=sys_param['depth_l1_weight_init'],
                                        lr_final=sys_param['depth_l1_weight_final'],
                                        max_steps=sys_param['iterations_MAX'])
    # 获取训练摄像机（视点）列表，并将其复制到 viewpoint_stack 中
    viewpoint_stack = scene.getTrainCameras().copy()
    print("训练模块！！！！-----训练摄像机列表: ", len(viewpoint_stack))
    # 创建一个包含所有视点索引的列表
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    # 初始化损失记录列表
    loss_history = []

    # 创建实时绘图函数
    def update_loss_plot(iteration, loss, save_path=None):
        loss_history.append(loss)
        if iteration % 10000 == 0:
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, iteration + 1), loss_history, label="Loss")
            plt.xlabel("Iteration")
            plt.ylabel("Loss")
            plt.title("Training Loss Curve")
            plt.legend()
            plt.grid()
            if save_path:
                plt.savefig(save_path)
            plt.close()

    # 训练进度条
    progress_bar = tqdm(range(first_iter, sys_param['iterations_MAX']), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, sys_param['iterations_MAX'] + 1):
        # 记录迭代开始时间
        iter_start.record()
        # 更新学习率
        gaussian_model.update_learning_rate(iteration)
        # 每1000次迭代增加一次球谐函数阶数
        if iteration % 1000 == 0:
            gaussian_model.oneupSHdegree()
        
        ######################## 随机选择一个相机视角 ########################
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        # 随机索引一个相机
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        ######################## 选中相机视角渲染 ########################
        if (iteration - 1) == sys_param['debug_from']:
            sys_param['debug'] = True
        # 设置背景色
        bg = torch.rand((3), device='cuda') if sys_param['random_background'] else background
        # 调用render函数对当前相机视角进行渲染
        render_pkg = render(viewpoint_camera=viewpoint_cam, pc=gaussian_model, sys_param=sys_param, bg_color=bg,
                            separate_sh=SPARSE_ADAM_AVAILABLE, use_trained_exp=sys_param['train_test_exp'])
        # 提取渲染结果从render_pkg中
        # image：渲染的图像（通常是RGB图像）。
        # viewspace_point_tensor：视空间中的点，可能用于后续计算或分析。
        # visibility_filter：可见性过滤器，用于筛选可见的点或区域。
        # radii：可能表示物体或场景中物体的半径，或者是渲染时的某些尺度信息。
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg['render'], render_pkg['viewspace_points'], render_pkg['visibility_filter'], render_pkg['radii']
        # 应用透明度掩码
        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        ######################## 计算RGB图像损失 ########################
        # 获取该相机视角真实图片
        gt_image = viewpoint_cam.original_image.cuda()
        # 计算L1损失，像素值的绝对差的平均值
        Ll1 = l1_loss(image, gt_image)
        # 计算SSIM损失
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)
        # 加权损失
        loss = (1.0 - sys_param['lambda_dssim']) * Ll1 + sys_param['lambda_dssim'] * (1.0 - ssim_value)
        # 实时更新损失图像
        
        update_loss_plot(iteration, loss.item(), save_path=f"loss_curve_{iteration}.png")
        ######################## 计算深度图损失 ########################
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg['depth']
            mono_invDepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()
            Ll1depth_pure = torch.abs((invDepth - mono_invDepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0
        
        ######################## 反向传播损失 ########################
        loss.backward()
        # 记录迭代结束时间
        iter_end.record()

        # 该代码块运行在不需要梯度计算的上下文中，用于节约显存和计算资源，表明以下代码不涉及模型的反向传播和梯度计算。
        with torch.no_grad():
            # 平滑损失值，减少显示的损失值的波动，提供更稳定的损失趋势
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log
            # 每隔10次迭代，更新进度条，显示平滑后的损失值和深度损失值
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            # 到最大迭代次数后，关闭进度条
            if iteration == sys_param['iterations_MAX']:
                progress_bar.close()

            ######################## 日志记录和模型保存 ########################
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), sys_param['test_iterations'], scene, render, (sys_param, background, sys_param['train_test_exp'], SPARSE_ADAM_AVAILABLE), sys_param['train_test_exp'])
            if (iteration in sys_param['save_iterations']):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
            
            ######################## 高斯模型稠密化处理 ########################
            if iteration < sys_param['densify_until_iter']:
                # 更新高斯模型中一些点的最大半径
                gaussian_model.max_radii2D[visibility_filter] = torch.max(gaussian_model.max_radii2D[visibility_filter], radii[visibility_filter])
                # 统计稠密化信息
                gaussian_model.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > sys_param['densify_from_iter'] and iteration % sys_param['densification_interval'] == 0:
                    size_threshold = 20 if iteration > sys_param['opacity_reset_interval'] else None
                    gaussian_model.densify_and_prune(sys_param['densify_grad_threshold'], 0.005, scene.cameras_extent, size_threshold, radii)
                
                if iteration % sys_param['opacity_reset_interval'] == 0 or (sys_param['white_background'] and iteration == sys_param['densify_from_iter']):
                    gaussian_model.reset_opacity()
            
            ######################## 对优化器进行一次参数更新 ########################
            if iteration < sys_param['iterations_MAX']:
                # 曝光参数优化 执行一次梯度更新；梯度清零，准备下一轮迭代 将梯度显式设置为 None，可以节省显存，提升性能
                gaussian_model.exposure_optimizer.step()
                gaussian_model.exposure_optimizer.zero_grad(set_to_none = True)
                # 稀疏优化 专门处理当前可见（visible）点云的优化
                if use_sparse_adam:
                    visible = radii > 0
                    gaussian_model.optimizer.step(visible, radii.shape[0])
                    gaussian_model.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussian_model.optimizer.step()
                    gaussian_model.optimizer.zero_grad(set_to_none = True)
            
            ######################## 训练过程中的中间检查点记录 ########################
            if (iteration in sys_param['checkpoint_iterations']):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussian_model.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(sys_param):
    if not sys_param['model_path']:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        sys_param['model_path'] = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(sys_param['model_path']))
    os.makedirs(sys_param['model_path'], exist_ok = True)
    # with open(os.path.join(sys_param['model_path'], "cfg_args"), 'w') as cfg_log_f:
    #     cfg_log_f.write(str(argparse.Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(sys_param['model_path'])
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, sys_param, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussian_model, *sys_param)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

# 渲染一个场景中的高斯模型
# viewpoint_camera：一个摄像机对象，包含了视角和投影矩阵等信息。
# pc：高斯模型对象（GaussianModel），包含了需要渲染的3D高斯点云。
# pipe：渲染管线的相关配置，包含了调试模式和抗锯齿等参数。
# bg_color：背景色的张量，必须在GPU上。
# scaling_modifier：可选的缩放因子，用于调整渲染的大小。
# separate_sh：是否分离SH（球面谐波）特征，影响渲染过程中SH颜色的处理方式。
# override_color：如果提供了此参数，将使用自定义颜色，否则会计算或使用默认颜色。
# use_trained_exp：是否使用训练中获取的曝光信息来调整渲染图像。
def render(viewpoint_camera, pc:GaussianModel, sys_param, bg_color:torch.Tensor, scaling_modifier=1.0, separate_sh=False, override_color=None, use_trained_exp=False):
    # 创建一个零向量，用于保存渲染过程中的2D屏幕空间点，并为其启用梯度计算
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    # retain_grad()一般用于非叶子节点（非模型参数）的张量，以便在反向传播后仍然可以访问梯度。
    try:
        screenspace_points.retain_grad()
    except:
        pass
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    # 设置光栅化配置
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        # 摄像机的中心位置
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=sys_param['debug'],
        # 是否开启抗锯齿模式
        antialiasing=sys_param['antialiasing']
    )
    # 初始化光栅化器，负责将3D数据渲染2D图像
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    means3D = pc.get_xyz
    # means2D 是一个形状相同但在屏幕空间中的点 (screenspace_points)，这些点将用来计算最终渲染图像上的每个像素位置。
    means2D = screenspace_points
    # 获取透明度信息
    opacity = pc.get_opacity

    scales = None
    rotations = None
    cov3D_precomp = None
    if sys_param['compute_cov3D_python']:
        # 计算协方差矩阵
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
    
    # 处理了基于球面谐波（Spherical Harmonics, SHs）和RGB颜色之间的转换过程
    shs = None
    colors_precomp = None
    # 检查是否提供有颜色
    if override_color is None:
        if sys_param['convert_SHs_python']:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            if separate_sh:
                dc, shs = pc.get_features_dc, pc.get_features_rest
            else:
                shs = pc.get_features
    else:
        colors_precomp = override_color

    # 如果 separate_sh 为 True，则代码会处理 dc（直接颜色特征）和 shs（球面谐波特征）作为输入，进行渲染。 dc 和 shs 是高斯点的颜色和光照特征，rasterizer 函数需要这些特征来渲染图像。
    # 如果 separate_sh 为 False，则只传入 shs，意味着直接使用球面谐波特征来渲染。
    if separate_sh:
        # rendered_image：渲染后的图像，通常是一个二维图像，显示了高斯点的投影结果。
        # radii：在屏幕上的半径，表示每个高斯点在图像中占据的像素区域大小。
        # depth_image：深度图像，记录每个点到相机的距离，通常用于3D重建或可视化。
        rendered_image, radii, depth_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            dc = dc,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp
        )
    else:
        rendered_image, radii, depth_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp
        )

    # 应用曝光信息渲染图像（仅训练）
    if use_trained_exp:
        # pc.get_exposure_from_name 是一个函数，可能是从点云模型（pc）中获取某个特定视角的曝光信息。viewpoint_camera.image_name 传递的是当前相机的图像名称，用于索引对应的曝光参数。
        # exposure[:3, :3] 取的是一个3x3矩阵，通常表示相机的色彩校正矩阵或RGB通道的线性变换。
        # exposure[:3, 3, None, None] 取的是曝光的偏移量，通常表示相机图像的亮度调整。
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3,   None, None]
    rendered_image = rendered_image.clamp(0, 1)
    # torch.cuda.is_available()

    # screenspace_points = screenspace_points.cpu()
    # print(f"Has NaNs: {torch.isnan(screenspace_points).any()}")

    # render: 渲染后的图像结果
    # viewspace_points: 相机变换后的视图空间坐标
    # visibility_filter: 计算可见点的索引，用于筛选出在屏幕上可见的高斯点。 [N, d]:N表示可见点数量，d表示点的维度
    out = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": (radii > 0).nonzero(),
        "radii": radii,
        "depth": depth_image
    }
    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 配置文件路径
    parser.add_argument('--config', type=str, default='./config')
    # 网络配置
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    # 是否启用检测程序,默认不启用
    parser.add_argument("--detect_anomaly", action='store_true', default=False)
    # 是否开始安静模式，不输出日志信息
    parser.add_argument('--quiet', action='store_true')
    # 禁用训练过程中的可视化查看器
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    # 用于指定从哪个迭代步骤开始启动调试,默认为-1，不启动调试
    parser.add_argument("--debug_from", type=int, default=-1)
    # 指定测试的迭代步骤, nargs="+"表示可以接受一个或多个值
    parser.add_argument('--test_iterations', nargs="+", type=int, default=[7_000, 30_000])
    # 指定保存模型检查点的迭代步骤,通常用于保存模型的关键进展点
    parser.add_argument('--save_iterations', nargs="+", type=int, default=[7_000, 30_000])
    # 通常用于更频繁地保存训练检查点，以便在训练过程中出现中断时可以快速恢复
    parser.add_argument('--checkpoint_iterations', nargs="+", type=int, default=[])
    # 指定从哪个检查点开始训练， 默认为None，表示从头开始训练
    parser.add_argument('--start_checkpoint', type=str, default=None)
    args = parser.parse_args()

    setproctitle.setproctitle("3dgs-slt-train")

    #################################### 读取配置信息 ####################################
    logging.info("读取系统信息中...")
    config_info = Load_config(args, mode='train')
    sys_param = config_info.system_info
    # print("系统参数: ", sys_param)
    safe_state(sys_param['quiet'])

    ################## 例化高斯网络模型，初始化数据 ##################
    print("！！！！训练模式！！！！")
    gaussian_model = GaussianModel(sys_param['sh_degree'], sys_param['optimizer_type'])
    # 输出模型的所有参数
    print("初始化Gaussian模型后的所有参数:")
    for attr, value in vars(gaussian_model).items():
        print(f"{attr}: {value}")
    scene = Scene(sys_param, gaussian_model)
    train(sys_param, gaussian_model, scene)
    print("！！！！训练结束！！！！")