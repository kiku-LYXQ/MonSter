import torch
import torch.nn.functional as F


def normalize_coords(grid):
    """Normalize coordinates of image scale to [-1, 1]
    Args:
        grid: [B, 2, H, W]

    将图像尺度坐标归一化到 [-1, 1] 范围
        Args:
            grid: [B, 2, H, W], 输入坐标网格，其中 grid[:,0] 是x坐标，grid[:,1] 是y坐标
        Returns:
            grid: [B, H, W, 2], 归一化后的坐标网格，范围 [-1, 1]
    """
    assert grid.size(1) == 2
    h, w = grid.size()[2:]
    # x坐标归一化公式: x_normalized = 2*(x_pixel / (w-1)) - 1
    grid[:, 0, :, :] = 2 * (grid[:, 0, :, :].clone() / (w - 1)) - 1  # x: [-1, 1]
    # y坐标归一化公式: y_normalized = 2*(y_pixel / (h-1)) - 1
    grid[:, 1, :, :] = 2 * (grid[:, 1, :, :].clone() / (h - 1)) - 1  # y: [-1, 1]
    # 调整维度顺序以适配 PyTorch 的 grid_sample 输入格式
    grid = grid.permute((0, 2, 3, 1))  # [B, H, W, 2]
    return grid


def meshgrid(img, homogeneous=False):
    """Generate meshgrid in image scale
    Args:
        img: [B, _, H, W]
        homogeneous: whether to return homogeneous coordinates
    Return:
        grid: [B, 2, H, W]

    生成图像尺度的坐标网格
    Args:
        img: [B, C, H, W], 输入图像（用于确定形状）
        homogeneous: 是否返回齐次坐标（添加第三维为1）
    Returns:
        grid: [B, 2, H, W] 或 [B, 3, H, W]（若齐次）
    """
    b, _, h, w = img.size()

    # 生成x坐标网格（行向量扩展为矩阵）
    x_range = torch.arange(0, w).view(1, 1, w).expand(1, h, w).type_as(img)  # [1, H, W]
    # 生成y坐标网格（列向量扩展为矩阵）
    y_range = torch.arange(0, h).view(1, h, 1).expand(1, h, w).type_as(img)

    # 拼接x和y坐标，形成网格
    grid = torch.cat((x_range, y_range), dim=0)  # [2, H, W], grid[:, i, j] = [j, i]
    grid = grid.unsqueeze(0).expand(b, 2, h, w)  # [B, 2, H, W]

    if homogeneous:
        # 添加齐次坐标（第三维数值为1） 在monster中没有用到
        ones = torch.ones_like(x_range).unsqueeze(0).expand(b, 1, h, w)  # [B, 1, H, W]
        grid = torch.cat((grid, ones), dim=1)  # [B, 3, H, W]
        assert grid.size(1) == 3
    return grid

def interp(x, sample_grid, padding_mode):
    """双线性插值封装函数，处理混合精度训练
        Args:
            x: 输入图像/特征图 [B, C, H, W]
            sample_grid: 归一化后的采样网格 [B, H, W, 2]
            padding_mode: 填充模式（'zeros'或'border'）
        Returns:
            output: 变形后的图像/特征图 [B, C, H, W]
    """
    original_dtype = x.dtype
    x_fp32 = x.float()
    sample_grid_fp32 = sample_grid.float()
    with torch.cuda.amp.autocast(enabled=False):
        # 使用PyTorch的grid_sample进行双线性插值
        output_fp32 = F.grid_sample(x_fp32, sample_grid_fp32, mode='bilinear', padding_mode=padding_mode)
    if original_dtype != torch.float32: # 恢复原始数据类型，避免推理报错
        output = output_fp32.to(original_dtype)
    else:
        output = output_fp32
    return output


def disp_warp(img, disp, padding_mode='border'):
    """Warping by disparity
    Args:
        img: [B, 3, H, W]   在 monster 传入的img是1/4原分辨率的深度特征图，是右图
        disp: [B, 1, H, W], positive
        padding_mode: 'zeros' or 'border'
    Returns:
        warped_img: [B, 3, H, W]
        valid_mask: [B, 3, H, W]
    """
    # assert disp.min() >= 0

    grid = meshgrid(img)  # [B, 2, H, W] in image scale 存的是不同像素点位置的的坐标 # grid[b, 0, i, j] = j（x坐标）, grid[b, 1, i, j] = i（y坐标）
    # Note that -disp here
    offset = torch.cat((-disp, torch.zeros_like(disp)), dim=1)  # [B, 2, H, W] 因为假设y轴不变化，只是x轴有视差所以zero_like
    sample_grid = grid + offset
    sample_grid = normalize_coords(sample_grid)  # [B, H, W, 2] in [-1, 1]
    # warped_img = F.grid_sample(img, sample_grid, mode='bilinear', padding_mode=padding_mode)
    warped_img = interp(img, sample_grid, padding_mode)


    mask = torch.ones_like(img)
    # valid_mask = F.grid_sample(mask, sample_grid, mode='bilinear', padding_mode='zeros')
    valid_mask = interp(mask, sample_grid, padding_mode)
    valid_mask[valid_mask < 0.9999] = 0
    valid_mask[valid_mask > 0] = 1
    return warped_img, valid_mask # valid_mask monster这里没用到