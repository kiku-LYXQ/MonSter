import os
import hydra
import torch
from tqdm import tqdm
import torch.optim as optim
# from util import InputPadder
from core.utils.utils import InputPadder
from core.monster import Monster 
from omegaconf import OmegaConf # OmegaConf 是一个专为层次化配置管理 设计的 Python 库，由 Facebook（现 Meta）开发，主要用于简化复杂应用（如机器学习实验）的配置管理。它是 Hydra 框架的核心依赖，支持 YAML 文件解析、动态配置合并和类型安全访问。
import torch.nn.functional as F
from accelerate import Accelerator
import core.stereo_datasets as datasets
from accelerate.utils import set_seed
from accelerate.logging import get_logger
from accelerate import DataLoaderConfiguration
from accelerate.utils import DistributedDataParallelKwargs


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import wandb # 专为机器学习研发设计的实验跟踪、数据可视化与协作平台。它帮助开发者和团队高效管理训练过程、记录超参数、监控模型性能，并支持结果的可视化分析与共享。
from pathlib import Path

def check_nan(layer, input, output):
    if isinstance(output, tuple):  # 检查是否为元组
        output = output[1][-1]
    if torch.isnan(output).any():
        print(f"NaN detected in {layer.__class__.__name__}")

def check_nan_grad(layer, grad_input, grad_output):
    if isinstance(grad_input, tuple):  # 检查是否为元组
        grad_input = grad_input[0]
    if torch.isnan(grad_input).any():
        print(f"NaN detected in gradient of {layer.__class__.__name__}")



def gray_2_colormap_np(img, cmap = 'rainbow', max = None):
    img = img.cpu().detach().numpy().squeeze()
    assert img.ndim == 2
    img[img<0] = 0
    mask_invalid = img < 1e-10
    if max == None:
        img = img / (img.max() + 1e-8)
    else:
        img = img/(max + 1e-8)

    norm = matplotlib.colors.Normalize(vmin=0, vmax=1.1)
    cmap_m = matplotlib.cm.get_cmap(cmap)
    map = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap_m)
    colormap = (map.to_rgba(img)[:,:,:3]*255).astype(np.uint8)
    colormap[mask_invalid] = 0

    return colormap

def sequence_loss(disp_preds, disp_init_pred, disp_gt, valid, loss_gamma=0.9, max_disp=192):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(disp_preds)
    assert n_predictions >= 1
    disp_loss = 0.0
    mag = torch.sum(disp_gt**2, dim=1).sqrt()
    valid = ((valid >= 0.5) & (mag < max_disp)).unsqueeze(1)
    assert valid.shape == disp_gt.shape, [valid.shape, disp_gt.shape]
    assert not torch.isinf(disp_gt[valid.bool()]).any()

    # quantile = torch.quantile((disp_init_pred - disp_gt).abs(), 0.9)
    init_valid = valid.bool() & ~torch.isnan(disp_init_pred)#  & ((disp_init_pred - disp_gt).abs() < quantile)
    disp_loss += 1.0 * F.smooth_l1_loss(disp_init_pred[init_valid], disp_gt[init_valid], reduction='mean')
    for i in range(n_predictions):
        adjusted_loss_gamma = loss_gamma**(15/(n_predictions - 1))
        i_weight = adjusted_loss_gamma**(n_predictions - i - 1)
        i_loss = (disp_preds[i] - disp_gt).abs()
        # quantile = torch.quantile(i_loss, 0.9)
        assert i_loss.shape == valid.shape, [i_loss.shape, valid.shape, disp_gt.shape, disp_preds[i].shape]
        disp_loss += i_weight * i_loss[valid.bool() & ~torch.isnan(i_loss)].mean()

    epe = torch.sum((disp_preds[-1] - disp_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    if valid.bool().sum() == 0:
        epe = torch.Tensor([0.0]).cuda()

    metrics = {
        'train/epe': epe.mean(),
        'train/1px': (epe < 1).float().mean(),
        'train/3px': (epe < 3).float().mean(),
        'train/5px': (epe < 5).float().mean(),
    }
    return disp_loss, metrics

def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """

    # 定位特征解码器模块的参数标识（用于参数分组）
    # list(map(id, ...)) 获取 feat_decoder 所有参数的唯一内存标识
    DPT_params = list(map(id, model.feat_decoder.parameters()))

    # 创建其他参数的过滤器：
    # 1. 过滤掉属于特征解码器的参数（通过内存标识比对）
    # 2. 只选择需要梯度更新的参数（requires_grad=True）
    rest_params = filter(lambda x:id(x) not in DPT_params and x.requires_grad, model.parameters())

    # 参数分组配置（不同模块设置不同学习率）
    params_dict = [
        # 特征解码器参数组：
        # 使用基础学习率的 1/2（通常用于预训练模块的微调）
        {'params': model.feat_decoder.parameters(), 'lr': args.lr/2.0},
        # 其他参数组：
        # 使用完整基础学习率（主网络部分的常规学习率）
        {'params': rest_params, 'lr': args.lr}, ]

    # 初始化 AdamW 优化器
    optimizer = optim.AdamW(params_dict, lr=args.lr, weight_decay=args.wdecay, eps=1e-8)

    # 每个参数组的最大学习率（需与params_dict顺序对应）：
    # - 特征解码器最大学习率：args.lr/2.0
    # - 其他参数最大学习率：args.lr
    # 总训练步数（增加100步作为缓冲区防止溢出）
    # 学习率上升阶段占总训练周期的比例（1%预热）
    # 禁用动量循环（与Adam优化器兼容性更好）
    # 退火策略采用线性变化
    # OneCycle调度策略：
    # 包含学习率预热（1 % 步数）：从初始值线性上升到最大值
    # 主训练阶段：线性下降回到初始学习率
    # 优势：快速收敛 + 逃离局部最优，尤其适合大batch size训练
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, [args.lr/2.0, args.lr], args.total_step+100,
            pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')


    return optimizer, scheduler

# 装饰器hydra.main 是 Hydra 配置管理框架的核心装饰器，用于将 Python 函数转换为 Hydra 应用的入口。以下是该装饰器参数及其作用的详细说明：
# version_base=None 禁用 Hydra 的版本检查默认行为
# config_path='config' 指定配置文件目录路径
# config_name='train_eth3d' 指定默认加载的配置文件
@hydra.main(version_base=None, config_path='config', config_name='train_eth3d')
def main(cfg):
    # 初始化基础设置
    set_seed(cfg.seed)
    logger = get_logger(__name__) # 获取日志记录器
    Path(cfg.save_path).mkdir(exist_ok=True, parents=True) # 创建模型保存目录

    # 分布式训练设置
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True) # 允许未使用参数（应对复杂模型结构）
    # 初始化加速器（支持多GPU/TPU/混合精度）
    # mixed_precision='bf16',  # 使用bfloat16混合精度训练
    # dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True),  # 可复现的数据采样
    # log_with='wandb',  # 集成Weights & Biases日志
    # kwargs_handlers=[kwargs],  # 分布式参数
    # step_scheduler_with_optimizer=False  # 手动控制学习率调度
    accelerator = Accelerator(mixed_precision='bf16', dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True), log_with='wandb', kwargs_handlers=[kwargs], step_scheduler_with_optimizer=False)

    # 初始化W&B追踪器
    # config=OmegaConf.to_container(cfg, resolve=True),  # 转换OmegaConf配置为字典
    accelerator.init_trackers(project_name=cfg.project_name, config=OmegaConf.to_container(cfg, resolve=True), init_kwargs={'wandb': cfg.wandb})

    # 数据准备阶段
    # 原始数据 → Dataset(单样本处理) → DataLoader(批量加载) → 模型训练
    #              ↳ 数据增强、格式转换    ↳ 打乱、并行化、内存优化
    train_dataset = datasets.fetch_dataloader(cfg) # 获取训练数据集
    # batch_size=cfg.batch_size//cfg.num_gpu,  # 单卡实际batch大小
    # pin_memory=True,  # 启用内存锁页加速数据传输
    # shuffle=True,  # 训练数据打乱
    # shuffle=True,  # 训练数据打乱
    # drop_last=True  # 丢弃不完整批次
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size//cfg.num_gpu,
        pin_memory=True, shuffle=True, num_workers=int(4), drop_last=True)

    # todo: why is aug_params empty?
    # 验证数据集配置
    aug_params = {}
    val_dataset = datasets.ETH3D(aug_params)
    # batch_size=1,  # 逐样本验证
    # shuffle=False,  # 验证数据顺序固定
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=int(1),
        pin_memory=True, shuffle=False, num_workers=int(4), drop_last=False)

    # 模型初始化
    model = Monster(cfg) # 核心立体匹配模型
    # 加载预训练权重
    if cfg.restore_ckpt is not None:
        assert cfg.restore_ckpt.endswith(".pth")
        print(f"Loading checkpoint from {cfg.restore_ckpt}")
        assert os.path.exists(cfg.restore_ckpt)
        checkpoint = torch.load(cfg.restore_ckpt, map_location='cpu')

        # 处理不同格式的checkpoint
        ckpt = dict()
        if 'state_dict' in checkpoint.keys():
            checkpoint = checkpoint['state_dict']
        for key in checkpoint:
            ckpt[key.replace('module.', '')] = checkpoint[key]

        # 严格匹配参数
        model.load_state_dict(ckpt, strict=True)
        print(f"Loaded checkpoint from {cfg.restore_ckpt} successfully")
    del ckpt, checkpoint

    # 优化器和学习率调度
    optimizer, lr_scheduler = fetch_optimizer(cfg, model)

    # 加速器封装（自动处理分布式训练）
    train_loader, model, optimizer, lr_scheduler, val_loader = accelerator.prepare(train_loader, model, optimizer, lr_scheduler, val_loader)
    model.to(accelerator.device) # 显式指定设备

    # 训练循环初始化
    total_step = 0
    should_keep_training = True
    # 主训练循环
    while should_keep_training:
        active_train_loader = train_loader

        model.train()
        #为什么要freeze BN层
        # BN层在CNN网络中大量使用，可以看上面bn层的操作，
        # 第一步是计算当前batch的均值和方差，也就是bn依赖于均值和方差，
        # 如果batch_size太小，计算一个小batch_size的均值和方差，肯定没有计算大的batch_size的均值和方差稳定和有意义，
        # 这个时候，还不如不使用bn层，因此可以将bn层冻结。
        # 另外，我们使用的网络，几乎都是在imagenet上pre-trained，完全可以使用在imagenet上学习到的参数。
        model.module.freeze_bn() # 冻结BatchNorm统计量（常用技巧）

        # 批次迭代
        for data in tqdm(active_train_loader, dynamic_ncols=True, disable=not accelerator.is_main_process):
            # 数据解包
            _, left, right, disp_gt, valid = [x for x in data]  # 左右视图 + 真实视差

            # 混合精度前向
            with accelerator.autocast():
                disp_init_pred, disp_preds, depth_mono = model(left, right, iters=cfg.train_iters)

            # 损失计算
            # 多尺度序列损失
            loss, metrics = sequence_loss(disp_preds, disp_init_pred, disp_gt, valid, max_disp=cfg.max_disp)
            # 反向传播
            accelerator.backward(loss) # 自动缩放梯度
            accelerator.clip_grad_norm_(model.parameters(), 1.0) # 梯度裁剪防爆炸
            optimizer.step()
            lr_scheduler.step() # 手动调度学习率
            optimizer.zero_grad()

            # 日志记录
            total_step += 1
            loss = accelerator.reduce(loss.detach(), reduction='mean') # 跨设备聚合损失
            metrics = accelerator.reduce(metrics, reduction='mean')
            accelerator.log({'train/loss': loss, 'train/learning_rate': optimizer.param_groups[0]['lr']}, total_step)
            accelerator.log(metrics, total_step)

            ####visualize the depth_mono and disp_preds
            # 可视化记录（每20步）
            if total_step % 20 == 0 and accelerator.is_main_process:
                # 图像归一化处理
                image1_np = left[0].squeeze().cpu().numpy()
                image1_np = (image1_np - image1_np.min()) / (image1_np.max() - image1_np.min()) * 255.0
                image1_np = image1_np.astype(np.uint8)
                image1_np = np.transpose(image1_np, (1, 2, 0))

                image2_np = right[0].squeeze().cpu().numpy()
                image2_np = (image2_np - image2_np.min()) / (image2_np.max() - image2_np.min()) * 255.0
                image2_np = image2_np.astype(np.uint8)
                image2_np = np.transpose(image2_np, (1, 2, 0))

                # 视差图着色
                depth_mono_np = gray_2_colormap_np(depth_mono[0].squeeze())
                disp_preds_np = gray_2_colormap_np(disp_preds[-1][0].squeeze())
                disp_gt_np = gray_2_colormap_np(disp_gt[0].squeeze())

                # 上传W&B
                accelerator.log({"disp_pred": wandb.Image(disp_preds_np, caption="step:{}".format(total_step))}, total_step)
                accelerator.log({"disp_gt": wandb.Image(disp_gt_np, caption="step:{}".format(total_step))}, total_step)
                accelerator.log({"depth_mono": wandb.Image(depth_mono_np, caption="step:{}".format(total_step))}, total_step)

            # 模型保存（每2500步）
            if (total_step > 0) and (total_step % cfg.save_frequency == 0):
                if accelerator.is_main_process:
                    save_path = Path(cfg.save_path + '/%d.pth' % (total_step))
                    model_save = accelerator.unwrap_model(model)
                    torch.save(model_save.state_dict(), save_path)
                    del model_save

            # 验证阶段（每2500步）
            if (total_step > 0) and (total_step % cfg.val_frequency == 0):
                torch.cuda.empty_cache() # 清理显存
                model.eval()
                elem_num, total_epe, total_out = 0, 0, 0
                for data in tqdm(val_loader, dynamic_ncols=True, disable=not accelerator.is_main_process):
                    _, left, right, disp_gt, valid = [x for x in data]
                    padder = InputPadder(left.shape, divis_by=32) # 输入对齐（保证尺寸可被32整除）
                    left, right = padder.pad(left, right)
                    with torch.no_grad():
                        disp_pred = model(left, right, iters=cfg.valid_iters, test_mode=True)
                    disp_pred = padder.unpad(disp_pred)
                    assert disp_pred.shape == disp_gt.shape, (disp_pred.shape, disp_gt.shape)

                    # 计算指标
                    epe = torch.abs(disp_pred - disp_gt) # 端点误差
                    out = (epe > 1.0).float() # 错误像素率
                    epe = torch.squeeze(epe, dim=1)
                    out = torch.squeeze(out, dim=1)
                    # 跨设备聚合指标
                    epe, out = accelerator.gather_for_metrics((epe[valid >= 0.5].mean(), out[valid >= 0.5].mean()))

                    # 累计统计
                    elem_num += epe.shape[0]
                    for i in range(epe.shape[0]):
                        total_epe += epe[i]
                        total_out += out[i]
                    # 记录验证指标
                    # 'val/epe': total_epe / elem_num,  # 平均端点误差
                    # 'val/d1': 100 * total_out / elem_num  # 错误率百分比
                    accelerator.log({'val/epe': total_epe / elem_num, 'val/d1': 100 * total_out / elem_num}, total_step)

                model.train()
                model.module.freeze_bn()

            # 终止条件
            if total_step == cfg.total_step:
                should_keep_training = False
                break

    # 训练结束处理
    if accelerator.is_main_process:
        save_path = Path(cfg.save_path + '/final.pth')
        model_save = accelerator.unwrap_model(model)
        torch.save(model_save.state_dict(), save_path) # 保存最终模型
        del model_save
    
    accelerator.end_training() # 清理分布式资源

if __name__ == '__main__':
    main()