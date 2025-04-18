from __future__ import print_function, division
import sys
sys.path.append('core')
import os
import argparse
import time
import logging
import numpy as np
import torch
from tqdm import tqdm
# add prefix core
from core.monster import Monster, autocast

import core.stereo_datasets as datasets
from core.utils.utils import InputPadder
from PIL import Image
import torch.nn.functional as F

class NormalizeTensor(object):
    """Normalize a tensor by given mean and std."""
    
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
    
    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
            
        Returns:
            Tensor: Normalized Tensor image.
        """
        # Ensure mean and std have the same number of channels as the input tensor
        Device = tensor.device
        self.mean = self.mean.to(Device)
        self.std = self.std.to(Device)

        # Normalize the tensor
        if self.mean.ndimension() == 1:
            self.mean = self.mean[:, None, None]
        if self.std.ndimension() == 1:
            self.std = self.std[:, None, None]

        return (tensor - self.mean) / self.std
    

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@torch.no_grad()
def validate_eth3d(model, iters=32, mixed_prec=False):
    """ Peform validation using the ETH3D (train) split """
    """ 
        使用ETH3D训练分割进行验证（注意：这里可能实际使用验证集）

        参数：
            model: 待验证的立体匹配模型
            iters: 迭代优化次数（影响模型推理精度）
            mixed_prec: 是否启用混合精度推理（节省显存加速计算）
        返回：
            包含EPE和D1指标的字典
    """
    # 模型切换到评估模式（关闭Dropout/BatchNorm等训练专用层）
    model.eval()
    aug_params = {} # 初始化数据增强参数（此处为空表示验证时不使用数据增强）
    val_dataset = datasets.ETH3D(aug_params) # 加载ETH3D验证数据集

    # 初始化指标存储列表
    out_list, epe_list = [], []

    # 遍历验证集中的每个样本
    for val_id in range(len(val_dataset)):
        # 获取当前样本数据：文件路径、左右视图、真值视差、有效区域掩码
        # imageL_file, imageR_file, GT_file are file path
        # image1, image2 预处理后的图片
        # flow_gt 真值视差图（Disparity Ground Truth）
        # valid_gt 有效区域掩码（Valid Mask） 值为0或1，用于评估数据集该像素点的视差真实值是否可靠
        (imageL_file, imageR_file, GT_file), image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        # 将图像数据送入GPU并增加批次维度（batch_size=1）
        image1 = image1[None].cuda() # 实际[1, 3, 489, 942]
        image2 = image2[None].cuda()

        # 图像填充对齐（确保尺寸能被32整除，适配模型下采样次数）
        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2) # [1, 3, 489, 942] -> [1, 3, 512, 960]

        # 模型推理阶段（禁用梯度计算）
        with torch.no_grad():
            # 混合精度上下文（自动选择float32/float16计算）
            with autocast(enabled=mixed_prec):
                # 模型前向传播，得到预测视差图（flow_pr）
                flow_pr = model(image1, image2, iters=iters, test_mode=True)

        # 后处理：移除填充区域 + 转CPU + 去除批次维度
        flow_pr = padder.unpad(flow_pr.float()).cpu().squeeze(0)
        # 验证预测与真值尺寸一致性
        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        # 计算端点误差EPE（End-Point Error）
        epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()
        epe_flattened = epe.flatten() # 展平为1维向量

        # 加载遮挡区域掩码（无效区域过滤）
        # 掩码文件路径替换逻辑：disp0GT.pfm → mask0nocc.png
        occ_mask = Image.open(GT_file.replace('disp0GT.pfm', 'mask0nocc.png'))
        occ_mask = np.ascontiguousarray(occ_mask).flatten() # 转为连续数组并展平

        # 构建有效区域掩码（valid_gt >=0.5 且 非遮挡区域）
        val = (valid_gt.flatten() >= 0.5) & (occ_mask == 255)
        # 若不使用遮挡掩码的版本（注释备用）:
        # val = (valid_gt.flatten() >= 0.5)

        # 计算D1指标（误差超过1像素视为错误）
        out = (epe_flattened > 1.0)
        image_out = out[val].float().mean().item() # 错误像素占比
        image_epe = epe_flattened[val].mean().item() # 平均EPE

        # 记录当前样本指标
        logging.info(f"ETH3D {val_id+1} out of {len(val_dataset)}. EPE {round(image_epe,4)} D1 {round(image_out,4)}")
        epe_list.append(image_epe)
        out_list.append(image_out)

    # 计算整体指标
    epe_list = np.array(epe_list)
    out_list = np.array(out_list)

    epe = np.mean(epe_list) # 平均端点误差
    d1 = 100 * np.mean(out_list) # D1错误率（百分比形式）

    # 打印汇总结果
    print("Validation ETH3D: EPE %f, D1 %f" % (epe, d1))
    # 返回指标字典（可用于超参优化或模型比较）
    return {'eth3d-epe': epe, 'eth3d-d1': d1}


@torch.no_grad()
def validate_kitti(model, iters=32, mixed_prec=False):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    # aug_params = {'crop_size': list([540, 960])}
    aug_params = {}
    val_dataset = datasets.KITTI(aug_params, image_set='training')
    torch.backends.cudnn.benchmark = True

    out_list, epe_list, elapsed_list = [], [], []
    for val_id in range(len(val_dataset)):
        (imageL_file, _, _), image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()
    
        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with torch.no_grad():
            with autocast(enabled=mixed_prec):
                start = time.time()
                flow_pr = model(image1, image2, iters=iters, test_mode=True)
                end = time.time()

        if val_id > 50:
            elapsed_list.append(end-start)
        flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)

        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()

        epe_flattened = epe.flatten()
        val = (valid_gt.flatten() >= 0.5) & (flow_gt.abs().flatten() < 192)
        # val = valid_gt.flatten() >= 0.5

        out = (epe_flattened > 3.0)
        image_out = out[val].float().mean().item()
        image_epe = epe_flattened[val].mean().item()
        if val_id < 9 or (val_id+1)%10 == 0:
            logging.info(f"KITTI Iter {val_id+1} out of {len(val_dataset)}. EPE {round(image_epe,4)} D1 {round(image_out,4)}. Runtime: {format(end-start, '.3f')}s ({format(1/(end-start), '.2f')}-FPS)")
        epe_list.append(epe_flattened[val].mean().item())
        out_list.append(out[val].cpu().numpy())

        # if val_id > 20:
        #     break

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    avg_runtime = np.mean(elapsed_list)

    print(f"Validation KITTI: EPE {epe}, D1 {d1}, {format(1/avg_runtime, '.2f')}-FPS ({format(avg_runtime, '.3f')}s)")
    return {'kitti-epe': epe, 'kitti-d1': d1}


@torch.no_grad()
def validate_vkitti(model, iters=32, mixed_prec=False):
    """ Peform validation using the vkitti (train) split """
    model.eval()
    aug_params = {}
    val_dataset = datasets.VKITTI2(aug_params)
    torch.backends.cudnn.benchmark = True

    out_list, epe_list, elapsed_list = [], [], []
    for val_id in range(len(val_dataset)):
        _, image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            start = time.time()
            flow_pr = model(image1, image2, iters=iters, test_mode=True)
            end = time.time()

        if val_id > 50:
            elapsed_list.append(end - start)
        flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)

        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        epe = torch.sum((flow_pr - flow_gt) ** 2, dim=0).sqrt()

        epe_flattened = epe.flatten()
        val = (valid_gt.flatten() >= 0.5) & (flow_gt.abs().flatten() < 192)
        # val = valid_gt.flatten() >= 0.5

        out = (epe_flattened > 3.0)
        image_out = out[val].float().mean().item()
        image_epe = epe_flattened[val].mean().item()
        if val_id < 9 or (val_id + 1) % 10 == 0:
            logging.info(
                f"VKITTI Iter {val_id + 1} out of {len(val_dataset)}. EPE {round(image_epe, 4)} D1 {round(image_out, 4)}. Runtime: {format(end - start, '.3f')}s ({format(1 / (end - start), '.2f')}-FPS)")
        epe_list.append(epe_flattened[val].mean().item())
        out_list.append(out[val].cpu().numpy())

        # if val_id > 20:
        #     break

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    avg_runtime = np.mean(elapsed_list)

    print(f"Validation VKITTI: EPE {epe}, D1 {d1}, {format(1 / avg_runtime, '.2f')}-FPS ({format(avg_runtime, '.3f')}s)")
    return {'vkitti-epe': epe, 'vkitti-d1': d1}



@torch.no_grad()
def validate_sceneflow(model, iters=32, mixed_prec=False):
    """ Peform validation using the Scene Flow (TEST) split """
    model.eval()
    val_dataset = datasets.SceneFlowDatasets(dstype='frames_finalpass', things_test=True)
    torch.backends.cudnn.benchmark = True

    out_list, epe_list, elapsed_list = [], [], []
    for val_id in tqdm(range(len(val_dataset))):
        _, image1, image2, flow_gt, valid_gt = val_dataset[val_id]

        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            start = time.time()
            flow_pr = model(image1, image2, iters=iters, test_mode=True)
            end = time.time()
        # print(torch.cuda.memory_summary(device=None, abbreviated=False))
        if val_id > 50:
            elapsed_list.append(end-start)

        flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)
        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)

        # epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()
        epe = torch.abs(flow_pr - flow_gt)

        epe = epe.flatten()
        val = (valid_gt.flatten() >= 0.5) & (flow_gt.abs().flatten() < 192)

        if(np.isnan(epe[val].mean().item())):
            continue

        out = (epe > 3.0)
        image_out = out[val].float().mean().item()
        image_epe = epe[val].mean().item()
        if val_id < 9 or (val_id + 1) % 10 == 0:
            logging.info(
                f"Scene Flow Iter {val_id + 1} out of {len(val_dataset)}. EPE {round(image_epe, 4)} D1 {round(image_out, 4)}. Runtime: {format(end - start, '.3f')}s ({format(1 / (end - start), '.2f')}-FPS)")

        print('epe', epe[val].mean().item())
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    avg_runtime = np.mean(elapsed_list)
    # f = open('test.txt', 'a')
    # f.write("Validation Scene Flow: %f, %f\n" % (epe, d1))

    print(f"Validation Scene Flow: EPE {epe}, D1 {d1}, {format(1/avg_runtime, '.2f')}-FPS ({format(avg_runtime, '.3f')}s)" )
    return {'scene-disp-epe': epe, 'scene-disp-d1': d1}

@torch.no_grad()
def validate_driving(model, iters=32, mixed_prec=False):
    """ Peform validation using the DrivingStereo (test) split """
    model.eval()
    aug_params = {}
    # val_dataset = datasets.DrivingStereo(aug_params, image_set='test')
    val_dataset = datasets.DrivingStereo(aug_params, image_set='cloudy')
    print(len(val_dataset))
    torch.backends.cudnn.benchmark = True

    out_list, epe_list, elapsed_list = [], [], []
    out1_list, out2_list = [], []
    for val_id in range(len(val_dataset)):
        _, image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with torch.autocast(device_type='cuda', enabled=mixed_prec):
            start = time.time()
            flow_pr = model(image1, image2, iters=iters, test_mode=True)
            end = time.time()

        if val_id > 50:
            elapsed_list.append(end-start)
        flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)

        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()

        epe_flattened = epe.flatten()
        val = (valid_gt.flatten() >= 0.5) & (flow_gt.abs().flatten() < 192)
        # val = valid_gt.flatten() >= 0.5

        out = (epe_flattened > 3.0)
        out1 = (epe_flattened > 1.0)
        out2 = (epe_flattened > 2.0)
        image_out = out[val].float().mean().item()
        image_epe = epe_flattened[val].mean().item()
        if val_id < 9 or (val_id+1)%10 == 0:
            logging.info(f"Driving Iter {val_id+1} out of {len(val_dataset)}. EPE {round(image_epe,4)} D1 {round(image_out,4)}. Runtime: {format(end-start, '.3f')}s ({format(1/(end-start), '.2f')}-FPS)")
        epe_list.append(epe_flattened[val].mean().item())
        out_list.append(out[val].cpu().numpy())
        out1_list.append(out1[val].cpu().numpy())
        out2_list.append(out2[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)
    out1_list = np.concatenate(out1_list)
    out2_list = np.concatenate(out2_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)
    bad_2 = 100 * np.mean(out2_list)
    bad_1 = 100 * np.mean(out1_list)
    avg_runtime = np.mean(elapsed_list)

    print(f"Validation DrivingStereo: EPE {epe}, bad1 {bad_1}, bad2 {bad_2}, bad3 {d1}, {format(1/avg_runtime, '.2f')}-FPS ({format(avg_runtime, '.3f')}s)")
    return {'driving-epe': epe, 'driving-d1': d1}


@torch.no_grad()
def validate_middlebury(model, iters=32, split='F', mixed_prec=False):
    """ Peform validation using the Middlebury-V3 dataset """
    model.eval()
    aug_params = {}
    val_dataset = datasets.Middlebury(aug_params, split=split)

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        (imageL_file, _, _), image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)
        a = input('input something')
        print(a)

        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()

        epe_flattened = epe.flatten()

        occ_mask = Image.open(imageL_file.replace('im0.png', 'mask0nocc.png')).convert('L')
        occ_mask = np.ascontiguousarray(occ_mask, dtype=np.float32).flatten()

        val = (valid_gt.reshape(-1) >= 0.5) & (flow_gt[0].reshape(-1) < 192) & (occ_mask==255)
        out = (epe_flattened > 2.0)
        image_out = out[val].float().mean().item()
        image_epe = epe_flattened[val].mean().item()
        logging.info(f"Middlebury Iter {val_id+1} out of {len(val_dataset)}. EPE {round(image_epe,4)} D1 {round(image_out,4)}")
        epe_list.append(image_epe)
        out_list.append(image_out)

    epe_list = np.array(epe_list)
    out_list = np.array(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    print(f"Validation Middlebury{split}: EPE {epe}, D1 {d1}")
    return {f'middlebury{split}-epe': epe, f'middlebury{split}-d1': d1}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default="/data2/cjd/mono_fusion/checkpoints/sceneflow.pth")

    parser.add_argument('--dataset', help="dataset for evaluation", default='sceneflow', choices=["eth3d", "kitti", "sceneflow", "vkitti", "driving"] + [f"middlebury_{s}" for s in 'FHQ'])
    parser.add_argument('--mixed_precision', default=False, action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')

    # Architecure choices
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--max_disp', type=int, default=192, help="max disp of geometry encoding volume")
    args = parser.parse_args()

    model = torch.nn.DataParallel(Monster(args), device_ids=[0])

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Total number of parameters: {total_params:.2f}M")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Total number of trainable parameters: {trainable_params:.2f}M")

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth")
        logging.info("Loading checkpoint...")
        logging.info(args.restore_ckpt)
        assert os.path.exists(args.restore_ckpt)
        checkpoint = torch.load(args.restore_ckpt)
        ckpt = dict()
        if 'state_dict' in checkpoint.keys():
            checkpoint = checkpoint['state_dict']
        for key in checkpoint:
            # ckpt['module.' + key] = checkpoint[key]
            if key.startswith("module."):
                ckpt[key] = checkpoint[key]  # 保持原样
            else:
                ckpt["module." + key] = checkpoint[key]  # 添加 "module."

        model.load_state_dict(ckpt, strict=True)

        logging.info(f"Done loading checkpoint")

    model.cuda()
    model.eval()

    print(f"The model has {format(count_parameters(model)/1e6, '.2f')}M learnable parameters.")
    use_mixed_precision = args.corr_implementation.endswith("_cuda")

    if args.dataset == 'eth3d':
        validate_eth3d(model, iters=args.valid_iters, mixed_prec=use_mixed_precision)

    elif args.dataset == 'kitti':
        validate_kitti(model, iters=args.valid_iters, mixed_prec=use_mixed_precision)

    elif args.dataset in [f"middlebury_{s}" for s in 'FHQ']:
        validate_middlebury(model, iters=args.valid_iters, split=args.dataset[-1], mixed_prec=use_mixed_precision)

    elif args.dataset == 'sceneflow':
        validate_sceneflow(model, iters=args.valid_iters, mixed_prec=use_mixed_precision)

    elif args.dataset == 'vkitti':
        validate_vkitti(model, iters=args.valid_iters, mixed_prec=use_mixed_precision)

    elif args.dataset == 'driving':
        validate_driving(model, iters=args.valid_iters, mixed_prec=use_mixed_precision)
