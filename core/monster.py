import torch
import torch.nn as nn
import torch.nn.functional as F
from core.update import BasicMultiUpdateBlock, BasicMultiUpdateBlock_mix2
from core.geometry import Combined_Geo_Encoding_Volume
from core.submodule import *
from core.refinement import REMP
from core.warp import disp_warp
import matplotlib.pyplot as plt

try:
    autocast = torch.cuda.amp.autocast
except:
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass
import sys
# todo: correct path name
# correct Depth-Anything-V2-list3 to Depth_Anything_V2_list3
sys.path.append('./Depth_Anything_V2_list3')
from Depth_Anything_V2_list3.depth_anything_v2.dpt import DepthAnythingV2, DepthAnythingV2_decoder

    
def compute_scale_shift(monocular_depth, gt_depth, mask=None):
    """
    计算 monocular depth 和 ground truth depth 之间的 scale 和 shift.
    
    参数:
    monocular_depth (torch.Tensor): 单目深度图，形状为 (H, W) 或 (N, H, W)
    gt_depth (torch.Tensor): ground truth 深度图，形状为 (H, W) 或 (N, H, W)
    mask (torch.Tensor, optional): 有效区域的掩码，形状为 (H, W) 或 (N, H, W)
    
    返回:
    scale (float): 计算得到的 scale
    shift (float): 计算得到的 shift

    计算单目深度图（monocular_depth）与真实深度图（gt_depth）之间的全局缩放（scale）和平移（shift）。

    参数:
        monocular_depth (torch.Tensor): 单目深度估计图，形状为 (H, W) 或 (N, H, W)
        gt_depth (torch.Tensor): 真实深度图（Ground Truth），形状与 monocular_depth 相同，  预测的深度图也用这个
        mask (torch.Tensor, optional): 有效区域掩码（True 表示有效区域），形状与 monocular_depth 相同

    返回:
        scale (float): 全局缩放系数，满足 gt_depth ≈ scale * monocular_depth + shift
        shift (float): 全局平移量

    数学原理:
        通过最小二乘法优化以下目标函数：
            sG, tG = argmin_{s,t} Σ (s·m_i + t - d_i)^2,
        其中 m_i 是单目深度值，d_i 是真实深度值，i ∈ 有效区域 Ω。
    """

    # 展平单目深度图以进行排序
    flattened_depth_maps = monocular_depth.clone().view(-1).contiguous()

    # 对单目深度值进行排序，用于后续阈值处理
    sorted_depth_maps, _ = torch.sort(flattened_depth_maps) # 升序排列
    percentile_10_index = int(0.2 * len(sorted_depth_maps)) # 取前20%的索引（可能是笔误，实际为20%分位数）
    threshold_10_percent = sorted_depth_maps[percentile_10_index] # 阈值：单目深度值的前20%分位

    # 生成有效区域掩码（若未提供）
    if mask is None:
        # 条件1：真实深度 > 0（排除无效值）
        # 条件2：单目深度 > 1e-2（排除数值不稳定区域）
        # 条件3：单目深度 > 阈值（过滤低置信度的深度估计）
        mask = (gt_depth > 0) & (monocular_depth > 1e-2) & (monocular_depth > threshold_10_percent)

    # 提取有效区域的单目深度和真实深度
    monocular_depth_flat = monocular_depth[mask] # [M], M为有效点数
    gt_depth_flat = gt_depth[mask]

    # 构建最小二乘问题的输入矩阵 X 和标签 y
    # 模型假设：d_i = s * m_i + t → 优化参数 s (scale) 和 t (shift)
    X = torch.stack([
        monocular_depth_flat, # 第一列：单目深度值 m_i
        torch.ones_like(monocular_depth_flat) # 第一列：单目深度值 m_i
    ], dim=1)
    y = gt_depth_flat
    
    # 使用最小二乘法计算 [scale, shift]
    # 解正规方程 X^T X [s; t] = X^T y
    # 为了避免矩阵奇异，添加正则化项 λI（λ=1e-6）
    A = torch.matmul(X.t(), X) + 1e-6 * torch.eye(2, device=X.device)
    b = torch.matmul(X.t(), y)
    params = torch.linalg.solve(A, b) # 解方程 A·params = b → params = [s, t]
    
    scale, shift = params[0].item(), params[1].item() # 转换为 Python float
    
    return scale, shift


class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(BasicConv(in_channels, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))
                                    
        self.conv2 = nn.Sequential(BasicConv(in_channels*2, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))                             

        self.conv3 = nn.Sequential(BasicConv(in_channels*4, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*6, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1)) 


        self.conv3_up = BasicConv(in_channels*6, in_channels*4, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv2_up = BasicConv(in_channels*4, in_channels*2, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv1_up = BasicConv(in_channels*2, 8, deconv=True, is_3d=True, bn=False,
                                  relu=False, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.agg_0 = nn.Sequential(BasicConv(in_channels*8, in_channels*4, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1),)

        self.agg_1 = nn.Sequential(BasicConv(in_channels*4, in_channels*2, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1))



        self.feature_att_8 = FeatureAtt(in_channels*2, 64)
        self.feature_att_16 = FeatureAtt(in_channels*4, 192)
        self.feature_att_32 = FeatureAtt(in_channels*6, 160)
        self.feature_att_up_16 = FeatureAtt(in_channels*4, 192)
        self.feature_att_up_8 = FeatureAtt(in_channels*2, 64)

    def forward(self, x, features):
        conv1 = self.conv1(x)
        conv1 = self.feature_att_8(conv1, features[1])

        conv2 = self.conv2(conv1)
        conv2 = self.feature_att_16(conv2, features[2])

        conv3 = self.conv3(conv2)
        conv3 = self.feature_att_32(conv3, features[3])

        conv3_up = self.conv3_up(conv3)
        conv2 = torch.cat((conv3_up, conv2), dim=1)
        conv2 = self.agg_0(conv2)
        conv2 = self.feature_att_up_16(conv2, features[2])

        conv2_up = self.conv2_up(conv2)
        conv1 = torch.cat((conv2_up, conv1), dim=1)
        conv1 = self.agg_1(conv1)
        conv1 = self.feature_att_up_8(conv1, features[1])

        conv = self.conv1_up(conv1)

        return conv

class Feat_transfer_cnet(nn.Module):
    def __init__(self, dim_list, output_dim):
        super(Feat_transfer_cnet, self).__init__()

        self.res_16x = nn.Conv2d(dim_list[0]+192, output_dim, kernel_size=3, padding=1, stride=1)
        self.res_8x = nn.Conv2d(dim_list[0]+96, output_dim, kernel_size=3, padding=1, stride=1)
        self.res_4x = nn.Conv2d(dim_list[0]+48, output_dim, kernel_size=3, padding=1, stride=1)

    def forward(self, features, stem_x_list):
        """
                前向传播：多尺度特征融合
                Args:
                    features (list): 主干网络提取的多尺度特征列表（假设按分辨率从低到高排序）    在 monster 中是经过deepanythingv2提取的深度特征列表（和深度相关性比较大）
                      - features[0]: 最低分辨率特征（如1/4x），形状 [B, C, H/4, W/4]
                      - features[1]: 中分辨率特征（如1/8x），形状 [B, C, H/8, W/8]
                      - features[2]: 最高分辨率特征（如1/16x），形状 [B, C, H/16, W/16]
                    stem_x_list (list): 辅助分支提供的多尺度特征列表（如来自另一个网络分支）   在 monster 中是由原始左视图图像经过下采样的得到的特征值列表（应该是几何特征，感觉是为了融合特征，减少信息损失）
                      - stem_x_list[0]: 高分辨率辅助特征，形状 [B, 128, H/16, W/16] 因为初始化output_dim是128
                      - stem_x_list[1]: 中分辨率辅助特征，形状 [B, 128, H/8, W/8]
                      - stem_x_list[2]: 低分辨率辅助特征，形状 [B, 128, H/4, W/4]
                Returns:
                    list: 融合后的多尺度特征列表，每个尺度特征重复两次（用于立体匹配的左右视图）
        """
        features_list = []
        feat_16x = self.res_16x(torch.cat((features[2], stem_x_list[0]), 1)) # 就是一个简单的沿着通道维度拼接后卷积，输出
        feat_8x = self.res_8x(torch.cat((features[1], stem_x_list[1]), 1))
        feat_4x = self.res_4x(torch.cat((features[0], stem_x_list[2]), 1))

        # ------------------- 构造输出列表 -------------------
        # 每个尺度的融合特征重复两次（用于立体匹配中的左右视图一致性计算）
        # [
        #   [[B,128,H/4,W/4], [B,128,H/4,W/4]],  # 低感受野分辨率特征对
        #   [[B,128,H/8,W/8], [B,128,H/8,W/8]],  # 中感受野分辨率特征对
        #   [[B,128,H/16,W/16], [B,128,H/16,W/16]]  # 大感受野视野分辨率特征对
        # ]
        features_list.append([feat_4x, feat_4x])
        features_list.append([feat_8x, feat_8x])
        features_list.append([feat_16x, feat_16x])
        return features_list



class Feat_transfer(nn.Module):
    def __init__(self, dim_list):
        super(Feat_transfer, self).__init__()
        self.conv4x = nn.Sequential(
            nn.Conv2d(in_channels=int(48+dim_list[0]), out_channels=48, kernel_size=5, stride=1, padding=2),
            nn.InstanceNorm2d(48), nn.ReLU()
            )
        self.conv8x = nn.Sequential(
            nn.Conv2d(in_channels=int(64+dim_list[0]), out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.InstanceNorm2d(64), nn.ReLU()
            )
        self.conv16x = nn.Sequential(
            nn.Conv2d(in_channels=int(192+dim_list[0]), out_channels=192, kernel_size=5, stride=1, padding=2),
            nn.InstanceNorm2d(192), nn.ReLU()
            )
        self.conv32x = nn.Sequential(
            nn.Conv2d(in_channels=dim_list[0], out_channels=160, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(160), nn.ReLU()
            )
        self.conv_up_32x = nn.ConvTranspose2d(160,
                                192,
                                kernel_size=3,
                                padding=1,
                                output_padding=1,
                                stride=2,
                                bias=False)
        self.conv_up_16x = nn.ConvTranspose2d(192,
                                64,
                                kernel_size=3,
                                padding=1,
                                output_padding=1,
                                stride=2,
                                bias=False)
        self.conv_up_8x = nn.ConvTranspose2d(64,
                                48,
                                kernel_size=3,
                                padding=1,
                                output_padding=1,
                                stride=2,
                                bias=False)
        
        self.res_16x = nn.Conv2d(dim_list[0], 192, kernel_size=1, padding=0, stride=1)
        self.res_8x = nn.Conv2d(dim_list[0], 64, kernel_size=1, padding=0, stride=1)
        self.res_4x = nn.Conv2d(dim_list[0], 48, kernel_size=1, padding=0, stride=1)




    def forward(self, features):
        features_mono_list = []
        feat_32x = self.conv32x(features[3])
        feat_32x_up = self.conv_up_32x(feat_32x)
        # 1x1卷积features[2],然后加上features[2], feat_32x_up的拼接再卷积一次，最后的卷积只是为了获取感受野范围内的特征向量
        feat_16x = self.conv16x(torch.cat((features[2], feat_32x_up), 1)) + self.res_16x(features[2])
        feat_16x_up = self.conv_up_16x(feat_16x)
        feat_8x = self.conv8x(torch.cat((features[1], feat_16x_up), 1)) + self.res_8x(features[1])
        feat_8x_up = self.conv_up_8x(feat_8x)
        feat_4x = self.conv4x(torch.cat((features[0], feat_8x_up), 1)) + self.res_4x(features[0])
        features_mono_list.append(feat_4x)
        features_mono_list.append(feat_8x)
        features_mono_list.append(feat_16x)
        features_mono_list.append(feat_32x)
        return features_mono_list





class Monster(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        context_dims = args.hidden_dims

        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11], 
            'vitl': [4, 11, 17, 23], 
            'vitg': [9, 19, 29, 39]
        }
        mono_model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        dim_list_ = mono_model_configs[self.args.encoder]['features']
        dim_list = []
        dim_list.append(dim_list_)
        self.update_block = BasicMultiUpdateBlock(self.args, hidden_dims=args.hidden_dims)

        self.context_zqr_convs = nn.ModuleList([nn.Conv2d(context_dims[i], args.hidden_dims[i]*3, 3, padding=3//2) for i in range(self.args.n_gru_layers)])

        self.feat_transfer = Feat_transfer(dim_list)
        self.feat_transfer_cnet = Feat_transfer_cnet(dim_list, output_dim=args.hidden_dims[0])


        self.stem_2 = nn.Sequential( # 一般来说一个卷积后加一个归一化比较好，目的是为了防止梯度爆炸
            BasicConv_IN(3, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(32), nn.ReLU()
            )
        self.stem_4 = nn.Sequential(
            BasicConv_IN(32, 48, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(48, 48, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(48), nn.ReLU()
            )

        self.stem_8 = nn.Sequential(
            BasicConv_IN(48, 96, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(96, 96, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(96), nn.ReLU()
            )

        self.stem_16 = nn.Sequential(
            BasicConv_IN(96, 192, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(192, 192, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(192), nn.ReLU()
            )

        self.spx = nn.Sequential(nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1),)
        self.spx_2 = Conv2x_IN(24, 32, True)
        self.spx_4 = nn.Sequential(
            BasicConv_IN(96, 24, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(24, 24, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(24), nn.ReLU()
            )

        self.spx_2_gru = Conv2x(32, 32, True)
        self.spx_gru = nn.Sequential(nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1),)

        self.conv = BasicConv_IN(96, 96, kernel_size=3, padding=1, stride=1)
        self.desc = nn.Conv2d(96, 96, kernel_size=1, padding=0, stride=1)

        self.corr_stem = BasicConv(8, 8, is_3d=True, kernel_size=3, stride=1, padding=1)
        self.corr_feature_att = FeatureAtt(8, 96)
        self.cost_agg = hourglass(8)
        self.classifier = nn.Conv3d(8, 1, 3, 1, 1, bias=False)

        depth_anything = DepthAnythingV2(**mono_model_configs[args.encoder])
        depth_anything_decoder = DepthAnythingV2_decoder(**mono_model_configs[args.encoder])
        state_dict_dpt = torch.load(f'./pretrained/depth_anything_v2_{args.encoder}.pth', map_location='cpu')
        # state_dict_dpt = torch.load(f'/home/cjd/cvpr2025/fusion/Depth_Anything_V2_list3/depth_anything_v2_{args.encoder}.pth', map_location='cpu')
        depth_anything.load_state_dict(state_dict_dpt, strict=True)
        depth_anything_decoder.load_state_dict(state_dict_dpt, strict=False)

        # 下面两层是DepthAnythingV2的主要结构，论文自己说为了momo模块
        self.mono_encoder = depth_anything.pretrained # 返回的是DinoVisionTransformer的模型实例，已经加载权重那种实例 论文默认vitl
        self.mono_decoder = depth_anything.depth_head # 返回的是depth_head的模型实例，，已经加载权重那种实例

        # 具体来讲呢就是depth_anything_decoder中的depth_head_decoder是输出同层的分辨率大小，然后经过手动上采，样DPTHead只是前馈特征
        self.feat_decoder = depth_anything_decoder.depth_head # 返回的是depth_head_decoder的模型实例，需要训练权重那种实例，和上面不同的是去掉和修改了输出附近的一些层

        self.mono_encoder.requires_grad_(False)
        self.mono_decoder.requires_grad_(False)

        del depth_anything, state_dict_dpt, depth_anything_decoder
        self.REMP = REMP()


        self.update_block_mix_stereo = BasicMultiUpdateBlock_mix2(self.args, hidden_dims=args.hidden_dims)
        self.update_block_mix_mono = BasicMultiUpdateBlock_mix2(self.args, hidden_dims=args.hidden_dims)


        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def infer_mono(self, image1, image2):
        """

        Args:
            image1:
            image2:

        Returns:

        """
        height_ori, width_ori = image1.shape[2:] # 获取原始高宽 [H, W]

        # 图像尺寸预处理 ---------------------------------------------------------------
        # 将输入图像缩放到14/16倍（适配DinoViT的14x14 patch处理）
        resize_image1 = F.interpolate(image1, scale_factor=14 / 16, mode='bilinear', align_corners=True) # 输出：[B, C, H*14/16, W*14/16]
        resize_image2 = F.interpolate(image2, scale_factor=14 / 16, mode='bilinear', align_corners=True)

        # 计算patch网格划分 ----------------------------------------------------------
        patch_h, patch_w = resize_image1.shape[-2] // 14, resize_image1.shape[-1] // 14 # 垂直方向patch数量 (H_scaled // patch_size) 水平方向patch数量 (W_scaled // patch_size)

        # 特征编码阶段 ---------------------------------------------------------------
        # 通过单目编码器提取中间层特征（使用预定义的关键层索引）
        # features_left_encoder: 包含各层特征及class token的列表，列表中有四个元素，每个元素中的第一个元素是特征输出[B, N, D]，第二个是class token [B, D]
        features_left_encoder = self.mono_encoder.get_intermediate_layers(resize_image1, self.intermediate_layer_idx[self.args.encoder], return_class_token=True)
        features_right_encoder = self.mono_encoder.get_intermediate_layers(resize_image2, self.intermediate_layer_idx[self.args.encoder], return_class_token=True)
        depth_mono = self.mono_decoder(features_left_encoder, patch_h, patch_w)
        depth_mono = F.relu(depth_mono)
        # 保证缩放回原始图像宽高，方便后续统计
        depth_mono = F.interpolate(depth_mono, size=(height_ori, width_ori), mode='bilinear', align_corners=False)

        # 利用depth_head的模型实例提取左右不同视野的深度特征图（带上采样）
        features_left_4x, features_left_8x, features_left_16x, features_left_32x = self.feat_decoder(features_left_encoder, patch_h, patch_w)
        features_right_4x, features_right_8x, features_right_16x, features_right_32x = self.feat_decoder(features_right_encoder, patch_h, patch_w)

        return depth_mono, [features_left_4x, features_left_8x, features_left_16x, features_left_32x], [features_right_4x, features_right_8x, features_right_16x, features_right_32x]

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
            if isinstance(m, nn.SyncBatchNorm):
                m.eval()

    def upsample_disp(self, disp, mask_feat_4, stem_2x):

        # with autocast(enabled=self.args.mixed_precision):
        xspx = self.spx_2_gru(mask_feat_4, stem_2x)
        spx_pred = self.spx_gru(xspx)
        spx_pred = F.softmax(spx_pred, 1)
        up_disp = context_upsample(disp*4., spx_pred).unsqueeze(1)

        return up_disp


    def forward(self, image1, image2, iters=12, flow_init=None, test_mode=False):
        """ Estimate disparity between pair of frames """

        # 图像归一化处理：将像素值从[0,255]线性变换到[-1,1]并确保内存连续
        image1 = (2 * (image1 / 255.0) - 1.0).contiguous()
        image2 = (2 * (image2 / 255.0) - 1.0).contiguous()

        # 使用Momo encoder进行编码
        with torch.autocast(device_type='cuda', dtype=torch.float32): 
            depth_mono, features_mono_left,  features_mono_right = self.infer_mono(image1, image2)

        scale_factor = 0.25
        size = (int(depth_mono.shape[-2] * scale_factor), int(depth_mono.shape[-1] * scale_factor))
        disp_mono_4x = F.interpolate(depth_mono, size=size, mode='bilinear', align_corners=False) # 4倍下采样

        # feat_transfer多尺度特征融合模块, 构建金字塔特征集合
        # todo: features_mono_left和features_mono_right都是原图大小的1/32, 1/16, 1/8, and 1/4信息还要进行特征融合？
        # 在原始的DepthHead中，最终的特征输出只是一个前馈来获得不同视野的特征图，就是小感受野的没有大感受野的信息广，所以做个FNN金字塔提取特征信息
        features_left = self.feat_transfer(features_mono_left)
        features_right = self.feat_transfer(features_mono_right)

        # 论文里这里说的不是很明确，猜测目的和残差思想差不多，保留一些图片的原始特征，不同的是，这里只是cat拼接特征维度，而非加法，应该是小小的涨点方法吧
        # 左图下采样，右图下采样
        # 1 从色彩特征构建出几何特征，用于与深度特征的融合，然后再去构建geo_encoding_volume体积块（只用了左右原图的4倍下采样）
        # 2 下采样生成输入图像分辨率的1/4、1/8和1/16的多尺度上下文特征，和同尺度深度特征列表融合后（方法：conv卷积），构建新的具有深度特征和几何特征的1/4、1/8和1/16的多尺度上下文特
        # todo: 为什么左图下采样到 1/16，左图只下采样到 1/2？
        stem_2x = self.stem_2(image1)
        stem_4x = self.stem_4(stem_2x)
        stem_8x = self.stem_8(stem_4x)
        stem_16x = self.stem_16(stem_8x)
        stem_2y = self.stem_2(image2)
        stem_4y = self.stem_4(stem_2y)

        stem_x_list = [stem_16x, stem_8x, stem_4x]
        features_left[0] = torch.cat((features_left[0], stem_4x), 1) # 拼接
        features_right[0] = torch.cat((features_right[0], stem_4y), 1)

        match_left = self.desc(self.conv(features_left[0])) # 两次卷积
        match_right = self.desc(self.conv(features_right[0]))
        gwc_volume = build_gwc_volume(match_left, match_right, self.args.max_disp//4, 8) # 论文来源 GWC cvpr2019
        gwc_volume = self.corr_stem(gwc_volume) # 通过3D卷积对代价体积进行初步特征提取。
        gwc_volume = self.corr_feature_att(gwc_volume, features_left[0]) # 引入注意力机制，增强与左图特征的关联
        geo_encoding_volume = self.cost_agg(gwc_volume, features_left) # 整合多尺度上下文，生成最终的几何编码（论文来源IGEV cvpr2023）

        # Init disp from geometry encoding volume
        prob = F.softmax(self.classifier(geo_encoding_volume).squeeze(1), dim=1)
        init_disp = disparity_regression(prob, self.args.max_disp//4) # 获得初始视差图
        
        del prob, gwc_volume

        if not test_mode:
            xspx = self.spx_4(features_left[0])
            xspx = self.spx_2(xspx, stem_2x)
            spx_pred = self.spx(xspx)
            spx_pred = F.softmax(spx_pred, 1)

        # IGEV中提出上下文网络由一系列残差块和下采样层组成，生成输入图像分辨率的1/4、1/8和1/16的多尺度上下文特征。这些多尺度上下文特征用于初始化ConvGRU的隐藏状态，并在每次迭代中插入到ConvGRU中以更新视差图。
        # cnet_list = self.cnet(image1, num_layers=self.args.n_gru_layers)
        cnet_list = self.feat_transfer_cnet(features_mono_left, stem_x_list) # 这里作者改为由自己实现的上下文来计算特征上下文列表，这里的特征应该是视差特征，在IGEV中提出cz，cr，cq这些用特征上下文网络生成固定值比较好
        net_list = [torch.tanh(x[0]) for x in cnet_list] # 列表中每个对象的左边元素（应该算是左图的特征图）利用tanh激活函数进行归一化到[-1, 1]  # 初始的隐藏层状态列表 隐藏状态通常需要一个对称的范围，所以用tanh，负值用于抑制无关信息
        inp_list = [torch.relu(x[1]) for x in cnet_list] # 中间特征用于生成门控参数，在循环中没有更新inp_list，所以中间特征是固定的
        inp_list = [torch.relu(x) for x in inp_list] # 应该是一个冗余操作
        inp_list = [ # 为每个 GRU 层生成三个门控参数（cz, cr, cq） 门控参数，可以理解为门的偏置，限制[0, 1]估计是为了防止梯度爆炸
            list( # 将结果转换为列表
                conv(i).split( # 对卷积结果沿通道维度分割
                    split_size=conv.out_channels//3, # 每块大小为总通道数的1/3
                    dim=1
                )
            )
            for i,conv in zip(inp_list, self.context_zqr_convs)] # zip(inp_list, self.context_zqr_convs) [(x1, conv1), (x2, conv2), (x3, conv3)] # 遍历输入和对应的卷积层
        net_list_mono = [x.clone() for x in net_list]

        geo_block = Combined_Geo_Encoding_Volume
        geo_fn = geo_block(match_left.float(), match_right.float(), geo_encoding_volume.float(), radius=self.args.corr_radius, num_levels=self.args.corr_levels)
        b, c, h, w = match_left.shape
        # coords = torch.arange(w).float() -> 生成一维坐标序列 [0, 1, ..., w-1]
        # coords = coords.to(match_left.device) -> 确保与输入张量在同一设备（CPU/GPU）
        # coords = coords.repeat(b, h, 1, 1) -> 复制到所有批次和行，形状 (B, H, W, 1) （coords存的是像素点的行序列号）
        coords = torch.arange(w).float().to(match_left.device).reshape(1,1,w,1).repeat(b, h, 1, 1).contiguous()
        disp = init_disp
        disp_preds = []
        for itr in range(iters): # 论文IGEV也是这个循环，但是循环的内部有些不同，但是迭代更新视差图的思想类似
            disp = disp.detach()
            if itr >= int(1):
                disp_mono_4x = disp_mono_4x.detach()
            geo_feat = geo_fn(disp, coords)
            if itr > int(iters-8):  # 为了生成后期update_block_mix_mono update_block_mix_stereo两个的操作所需的数据
                if itr == int(iters-7):
                    bs, _, _, _ = disp.shape
                    for i in range(bs): # 每个batch的每张图片使用一次最小二乘法
                        with torch.autocast(device_type='cuda', dtype=torch.float32): 
                            scale, shift = compute_scale_shift(disp_mono_4x[i].clone().squeeze(1).to(torch.float32), disp[i].clone().squeeze(1).to(torch.float32))
                        disp_mono_4x[i] = scale * disp_mono_4x[i] + shift
                
                warped_right_mono = disp_warp(features_right[0], disp_mono_4x.clone().to(features_right[0].dtype))[0]  
                flaw_mono = warped_right_mono - features_left[0] 

                warped_right_stereo = disp_warp(features_right[0], disp.clone().to(features_right[0].dtype))[0]  
                flaw_stereo = warped_right_stereo - features_left[0] 
                geo_feat_mono = geo_fn(disp_mono_4x, coords)

            if itr <= int(iters-8): # 前中期ConvGru迭代
                net_list, mask_feat_4, delta_disp = self.update_block(net_list, inp_list, geo_feat, disp, iter16=self.args.n_gru_layers==3, iter08=self.args.n_gru_layers>=2)
            else: # 后期ConvGru迭代
                net_list, mask_feat_4, delta_disp = self.update_block_mix_stereo(net_list, inp_list, flaw_stereo, disp, geo_feat, flaw_mono, disp_mono_4x, geo_feat_mono, iter16=self.args.n_gru_layers==3, iter08=self.args.n_gru_layers>=2)
                net_list_mono, mask_feat_4_mono, delta_disp_mono = self.update_block_mix_mono(net_list_mono, inp_list, flaw_mono, disp_mono_4x, geo_feat_mono, flaw_stereo, disp, geo_feat, iter16=self.args.n_gru_layers==3, iter08=self.args.n_gru_layers>=2)
                disp_mono_4x = disp_mono_4x + delta_disp_mono
                disp_mono_4x_up = self.upsample_disp(disp_mono_4x, mask_feat_4_mono, stem_2x)
                disp_preds.append(disp_mono_4x_up)

            disp = disp + delta_disp
            if test_mode and itr < iters-1:
                continue

            disp_up = self.upsample_disp(disp, mask_feat_4, stem_2x)

            if itr == iters - 1:
                refine_value = self.REMP(disp_mono_4x_up, disp_up, image1, image2)
                disp_up = disp_up + refine_value
            disp_preds.append(disp_up)

        if test_mode:
            return disp_up

        init_disp = context_upsample(init_disp*4., spx_pred.float()).unsqueeze(1)
        return init_disp, disp_preds, depth_mono