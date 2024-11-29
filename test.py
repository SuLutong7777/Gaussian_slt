import argparse
import logging
import setproctitle
import torch
import os
import torchvision
from tqdm import tqdm
from model import GaussianModel
from dataload import Scene
from config.config_read import Load_config
from utils.system_utils import makedirs
from utils.general_utils import safe_state
from train import render
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def render_set(sys_param, model_path, name, iteration, views, gaussians_model, background, separate_sh):
    # 渲染图片存储路径
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    # 标准图片存储路径
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    makedirs(render_path)
    makedirs(gts_path)
    # 测试进度条开启
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        # 渲染图像
        render_image = render(view, gaussian_model, sys_param, background, separate_sh=separate_sh, use_trained_exp=sys_param['train_test_exp'])['render']
        # 标准图像rgb
        gt_image = view.original_image[0:3, :, :]
        # 分割图像，只保留右半部分
        if sys_param['train_test_exp']:
            render_image = render_image[..., render_image.shape[-1] // 2:]
            gt_image = gt_image[..., gt_image.shape[-1] // 2]
        # 保存图像
        torchvision.utils.save_image(render_image, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt_image, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        
def render_sets(sys_param, gaussian_model: GaussianModel, scene:Scene, separate_sh: bool):
    with torch.no_grad():
        bg_color = [1, 1, 1] if sys_param['white_background'] else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device='cuda')
        if not sys_param['skip_train']:
            render_set(sys_param, sys_param['model_path'], "train", scene.loaded_iter, scene.getTrainCameras(), gaussian_model, background, separate_sh)
        if not sys_param['skip_test']:
            render_set(sys_param, sys_param['model_path'], "test", scene.loaded_iter, scene.getTestCameras(), gaussian_model, background, separate_sh)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 配置文件路径
    parser.add_argument('--config', type=str, default='./config')
    # 指定加载那一轮的模型进行训练
    parser.add_argument('--iteration', default=-1, type=int)
    # 跳过训练或者测试
    parser.add_argument('--skip_train', action='store_true')
    parser.add_argument('--skip_test', action='store_true')
    # 安静模式
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()

    setproctitle.setproctitle("3dgs-slt-test")
    logging.info("读取系统信息中...")
    config_info = Load_config(args, mode='test')
    sys_param = config_info.system_info
    safe_state(sys_param['quiet'])
    ########################## 例化模型和数据集 #########################
    print("！！！！测试模式！！！！")
    gaussian_model = GaussianModel(sys_param['sh_degree'])
    scene = Scene(sys_param, gaussian_model, load_iteration=sys_param['iteration'], shuffle=False)
    render_sets(sys_param, gaussian_model, scene, SPARSE_ADAM_AVAILABLE)
    print("！！！！测试结束！！！！")