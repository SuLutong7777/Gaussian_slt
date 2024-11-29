from pathlib import Path
import yaml
import os

# 加载配置参数类
class Load_config():
    def __init__(self, args, mode='train'):
        # 读取配置文件根目录
        self.root_config = args.config
        # 读取终端输入的信息
        self.system_info = self.load_terminal(args, mode)
        # 读取yaml文件中的信息
        self.load_yaml()
        if mode == 'train':
            # 将最大迭代次数加到保存迭代次数中
            self.system_info['save_iterations'].append(self.system_info['iterations_MAX'])

    # 读取终端信息函数
    def load_terminal(self, args, mode='train'):
        # 定义系统字典
        system_info={}
        ############### 训练参数 ###############
        if mode == 'train':
            system_info['ip'] = args.ip
            system_info['port'] = args.port
            system_info['detect_anomaly'] = args.detect_anomaly
            system_info['quiet'] = args.quiet
            system_info['disable_viewer'] = args.disable_viewer
            system_info['debug_from'] = args.debug_from
            system_info['test_iterations'] = args.test_iterations
            system_info['save_iterations'] = args.save_iterations
            system_info['checkpoint_iterations'] = args.checkpoint_iterations
            system_info['start_checkpoint'] = args.start_checkpoint
        ############### 测试参数 ###############
        elif mode == 'test':
            system_info['iteration'] = args.iteration
            system_info['skip_train'] = args.skip_train
            system_info['skip_test'] = args.skip_test
            system_info['quiet'] = args.quiet

        return system_info
    
    # 读取yaml文件信息
    def load_yaml(self):
        # 获取yaml配置文件的路径
        yaml_path = os.path.join(Path(self.root_config), Path("config.yaml"))
        # 读取yaml文件
        with open(yaml_path, 'r', encoding='utf-8') as f:
            cfg_info = yaml.load(f, Loader=yaml.FullLoader)
           
            ######################## 模型参数 ########################
            self.system_info['sh_degree'] = cfg_info['model']['sh_degree']
            self.system_info['source_path'] = cfg_info['model']['source_path']
            self.system_info['images_path'] = cfg_info['model']['images_path']
            self.system_info['model_path'] = cfg_info['model']['model_path']
            self.system_info['depths_path'] = cfg_info['model']['depths_path']
            self.system_info['resolution'] = cfg_info['model']['resolution']
            self.system_info['white_background'] = cfg_info['model']['white_background']
            self.system_info['train_test_exp'] = cfg_info['model']['train_test_exp']
            self.system_info['eval_mode'] = cfg_info['model']['eval_mode']
            self.system_info['data_device'] = cfg_info['model']['data_device']
            ######################## 优化器参数 ########################
            self.system_info['iterations_MAX'] = cfg_info['optimization']['iterations_MAX']
            self.system_info['position_lr_init'] = cfg_info['optimization']['position_lr_init']
            self.system_info['position_lr_final'] = cfg_info['optimization']['position_lr_final']
            self.system_info['position_lr_delay_mult'] = cfg_info['optimization']['position_lr_delay_mult']
            self.system_info['position_lr_max_steps'] = cfg_info['optimization']['position_lr_max_steps']
            self.system_info['feature_lr'] = cfg_info['optimization']['feature_lr']
            self.system_info['opacity_lr'] = cfg_info['optimization']['opacity_lr']
            self.system_info['scaling_lr'] = cfg_info['optimization']['scaling_lr']
            self.system_info['rotation_lr'] = cfg_info['optimization']['rotation_lr']
            self.system_info['exposure_lr_init'] = cfg_info['optimization']['exposure_lr_init']
            self.system_info['exposure_lr_final'] = cfg_info['optimization']['exposure_lr_final']
            self.system_info['exposure_lr_delay_steps'] = cfg_info['optimization']['exposure_lr_delay_steps']
            self.system_info['exposure_lr_delay_mult'] = cfg_info['optimization']['exposure_lr_delay_mult']
            self.system_info['lambda_dssim'] = cfg_info['optimization']['lambda_dssim']
            self.system_info['opacity_reset_interval'] = cfg_info['optimization']['opacity_reset_interval']
            self.system_info['percent_dense'] = cfg_info['optimization']['percent_dense']
            self.system_info['densification_interval'] = cfg_info['optimization']['densification_interval']
            self.system_info['densify_from_iter'] = cfg_info['optimization']['densify_from_iter']
            self.system_info['densify_until_iter'] = cfg_info['optimization']['densify_until_iter']
            self.system_info['densify_grad_threshold'] = cfg_info['optimization']['densify_grad_threshold']
            self.system_info['depth_l1_weight_init'] = cfg_info['optimization']['depth_l1_weight_init']
            self.system_info['depth_l1_weight_final'] = cfg_info['optimization']['depth_l1_weight_final']
            self.system_info['random_background'] = cfg_info['optimization']['random_background']
            self.system_info['optimizer_type'] = cfg_info['optimization']['optimizer_type']
            ######################## 管道配置参数 ########################
            self.system_info['convert_SHs_python'] = cfg_info['pipeline']['convert_SHs_python']
            self.system_info['compute_cov3D_python'] = cfg_info['pipeline']['compute_cov3D_python']
            self.system_info['debug'] = cfg_info['pipeline']['debug']
            self.system_info['antialiasing'] = cfg_info['pipeline']['antialiasing']
            
