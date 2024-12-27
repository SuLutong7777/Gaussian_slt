# Gaussian_slt
This is a replicated set of code for personal learning purposes.
## Set up
This code is developed with `python3.9.21`. PyTorch 2.5.1 and cuda 12.2 are required.  
It is recommended use `Anaconda` to set up the environment. Install the dependencies and activate the environment `3dgs-env` with
```
conda env create --file requirements.yaml
conda activate 3dgs-env
```
## Running
To train the model:
```
python train.py 
```
To render the model:
```
python test.py 
```
`source_path` and `model_path` can be configured in the `config.yaml` file.
## Some small episodes
1.pytorch和cuda版本对应关系以及安装命令:`https://pytorch.org/`

2.`pip install submodules/diff-gaussian-rasterization`失败，最后发现原因是`submodules/diff-gaussian-rasterization/third_party/glm`是空的，把`glm`源码下载过来就可以了

3.`Traceback (most recent call last):
  File "/home/seven/seven/gaussian-splatting-main/train.py", line 282, in <module>
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)
  File "/home/seven/seven/gaussian-splatting-main/train.py", line 111, in training
    render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
  File "/home/seven/seven/gaussian-splatting-main/gaussian_renderer/__init__.py", line 36, in render
    raster_settings = GaussianRasterizationSettings(
TypeError: <lambda>() got an unexpected keyword argument 'antialiasing'`报错，但是进入`GaussianRasterizationSettings`发现是有这个参数的，最后发现是因为最新版的有，更新一下`diff-gaussian-rasterization`就好了






## 代码架构
![代码架构](utils/gs_slt.png)
## 计算机图形学
![参考网站](https://chuquan.me/2024/03/23/foundation-of-computer-graphic-03/)
## 高斯方法架构
![参考网站](https://www.bilibili.com/video/BV1zi421v7Dr/?spm_id_from=333.337.search-card.all.click&vd_source=b8e7bf3a9fa3baa07f9407580fe339f2)
