import torch
import torch.nn as nn
import torch.nn.functional as F
from opt_einsum import contract

class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=2):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, output_dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))

class DispHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=1):
        super(DispHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, output_dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))

class ConvGRU(nn.Module):
    def __init__(self, hidden_dim, input_dim, kernel_size=3):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size//2)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size//2)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size//2)

    def forward(self, h, cz, cr, cq, *x_list):
        # cz 是更新门的偏置/上下文参数
        # cr 是重置门的偏置/上下文参数
        # cq 是候选状态的偏置/上下文参数
        # 为何不用 [0, 1]，而用tanh：
        # 丢失负值信息，导致隐藏状态无法动态调整（例如抑制无效特征）。
        # 非对称范围可能使门控机制偏向某一方向（如始终保留或遗忘信息），降低模型灵活性。
        # 门控r，z为何是[0, 1]
        # 用于决定多少信息来自旧或新
        x = torch.cat(x_list, dim=1)
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx) + cz)
        r = torch.sigmoid(self.convr(hx) + cr)
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)) + cq)
        h = (1-z) * h + z * q
        return h

class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))


    def forward(self, h, *x):
        # horizontal
        x = torch.cat(x, dim=1)
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))        
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))       
        h = (1-z) * h + z * q

        return h

def interp(x, dest):
    original_dtype = x.dtype
    x_fp32 = x.float()
    interp_args = {'mode': 'bilinear', 'align_corners': True}
    with torch.cuda.amp.autocast(enabled=False):
        output_fp32 = F.interpolate(x_fp32, dest.shape[2:], **interp_args)
    if original_dtype != torch.float32:
        output = output_fp32.to(original_dtype)
    else:
        output = output_fp32
    return output

class BasicMotionEncoder(nn.Module):
    def __init__(self, args):
        super(BasicMotionEncoder, self).__init__()
        self.args = args
        cor_planes = args.corr_levels * (2*args.corr_radius + 1) * (8+1)
        self.convc1 = nn.Conv2d(cor_planes, 64, 1, padding=0)
        self.convc2 = nn.Conv2d(64, 64, 3, padding=1)
        self.convd1 = nn.Conv2d(1, 64, 7, padding=3)
        self.convd2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv = nn.Conv2d(64+64, 128-1, 3, padding=1)

    def forward(self, disp, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        disp_ = F.relu(self.convd1(disp))
        disp_ = F.relu(self.convd2(disp_))

        cor_disp = torch.cat([cor, disp_], dim=1)
        out = F.relu(self.conv(cor_disp))
        return torch.cat([out, disp], dim=1)

def pool2x(x):
    return F.avg_pool2d(x, 3, stride=2, padding=1)

def pool4x(x):
    return F.avg_pool2d(x, 5, stride=4, padding=1)

# def interp(x, dest):
#     interp_args = {'mode': 'bilinear', 'align_corners': True}
#     return F.interpolate(x, dest.shape[2:], **interp_args)

class BasicMultiUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dims=[]):
        super().__init__()
        self.args = args

        # ----------------- 核心组件初始化 -----------------
        # 运动编码器：将视差(disp)和相关体积(corr)编码为运动特征
        self.encoder = BasicMotionEncoder(args)
        encoder_output_dim = 128 # 编码器输出维度（运动特征维度）

        # ----------------- 多层级GRU定义 -----------------
        # GRU层级说明（04/08/16表示下采样倍数）：
        # - gru16: 处理最低分辨率（16x下采样）的特征，捕捉全局上下文
        # - gru08: 处理中间分辨率（8x下采样）的特征，衔接高低层信息
        # - gru04: 处理高分辨率（4x下采样）的特征，生成局部细节修正

        # GRU04: 输入维度=hidden_dims[2]，输出维度=运动特征+高层特征（若层数>1）
        self.gru04 = ConvGRU(hidden_dims[2], encoder_output_dim + hidden_dims[1] * (args.n_gru_layers > 1))
        # GRU08: 输入维度=hidden_dims[1]，输出维度=低层特征（若层数==3） + 高层特征
        self.gru08 = ConvGRU(hidden_dims[1], hidden_dims[0] * (args.n_gru_layers == 3) + hidden_dims[2])
        # GRU16: 输入维度=hidden_dims[0]，输出维度=hidden_dims[1]
        self.gru16 = ConvGRU(hidden_dims[0], hidden_dims[1])

        # ----------------- 输出头定义 -----------------
        # 视差预测头：从最高分辨率隐藏状态预测delta_disp
        self.disp_head = DispHead(hidden_dims[2], hidden_dim=256, output_dim=1)
        factor = 2**self.args.n_downsample

        # 上采样掩码生成模块：将gru04的输出转换为32通道掩码
        self.mask_feat_4 = nn.Sequential(
            nn.Conv2d(hidden_dims[2], 32, 3, padding=1),
            nn.ReLU(inplace=True))

    def forward(self, net, inp, corr=None, disp=None, iter04=True, iter08=True, iter16=True, update=True):
        """
                前向传播：多层级GRU迭代更新

                Args:
                    net (list): 多层级隐藏状态 [net16, net08, net04]，初始为feat_transfer_cnet的输出
                    inp (list): 多层级输入特征 [inp16, inp08, inp04]
                    corr (Tensor): 相关性体积，形状(B,H,W,1,W2)
                    disp (Tensor): 当前视差图，形状(B,1,H,W)
                    iter04/iter08/iter16 (bool): 控制是否更新对应层级的GRU
                    update (bool): 是否生成delta_disp

                Returns:
                    net (list): 更新后的隐藏状态列表
                    mask_feat_4 (Tensor): 上采样掩码特征，形状(B,32,H,W)
                    delta_disp (Tensor): 视差修正量，形状(B,1,H,W)
        """
        # ----------------- 层级式GRU更新 -----------------
        # 更新顺序：从低分辨率到高分辨率（16x -> 8x -> 4x） 不同的分辨率都有自己的隐藏层

        # 1. 更新16x层级的GRU（最低分辨率）
        if iter16:
            # 输入：net[2] + inp[2]（低层输入，有三个，分别是cz， cr，cq） + pool2x(net[1])（来自8x层的池化特征）
            net[2] = self.gru16(net[2], *(inp[2]), pool2x(net[1]))

        # 2. 更新8x层级的GRU（中间分辨率）
        if iter08:
            if self.args.n_gru_layers > 2:
                # 输入：inp[1] + pool2x(net[0])（来自4x层的池化） + interp(net[2], net[1])（16x层的上采样）
                net[1] = self.gru08(net[1], *(inp[1]), pool2x(net[0]), interp(net[2], net[1]))
            else:
                # 输入：inp[1] + pool2x(net[0])
                net[1] = self.gru08(net[1], *(inp[1]), pool2x(net[0]))

        # 3. 更新4x层级的GRU（最高分辨率）
        if iter04:
            # 运动特征编码：将视差和相关体积转换为128维特征
            motion_features = self.encoder(disp, corr)
            if self.args.n_gru_layers > 1:
                # 输入：inp[0] + motion_features + interp(net[1], net[0])（8x层的上采样）
                net[0] = self.gru04(net[0], *(inp[0]), motion_features, interp(net[1], net[0]))
            else:
                # 输入：inp[0] + motion_features
                net[0] = self.gru04(net[0], *(inp[0]), motion_features)

        # ----------------- 输出生成 -----------------
        if not update:
            return net # 仅返回隐藏状态（用于中间迭代不生成输出）

        # 从最高分辨率隐藏状态（net[0]）生成视差修正量
        delta_disp = self.disp_head(net[0]) # 输出形状(B,1,H,W)
        # 生成上采样掩码（用于RAFT-style凸上采样）
        mask_feat_4 = self.mask_feat_4(net[0])
        return net, mask_feat_4, delta_disp

class BasicMultiUpdateBlock_mix(nn.Module):
    def __init__(self, args, hidden_dims=[]):
        super().__init__()
        self.args = args
        self.encoder = BasicMotionEncoder_mix(args)
        encoder_output_dim = 128

        self.gru04 = ConvGRU(hidden_dims[2], encoder_output_dim + hidden_dims[1] * (args.n_gru_layers > 1))
        self.gru08 = ConvGRU(hidden_dims[1], hidden_dims[0] * (args.n_gru_layers == 3) + hidden_dims[2])
        self.gru16 = ConvGRU(hidden_dims[0], hidden_dims[1])
        self.disp_head = DispHead(hidden_dims[2], hidden_dim=256, output_dim=2)
        factor = 2**self.args.n_downsample

        self.mask_feat_4 = nn.Sequential(
            nn.Conv2d(hidden_dims[2], 32, 3, padding=1),
            nn.ReLU(inplace=True))

    def forward(self, net, inp, flaw_stereo=None, disp=None, corr=None, flaw_mono=None, disp_mono=None, corr_mono=None, iter04=True, iter08=True, iter16=True, update=True):

        if iter16:
            net[2] = self.gru16(net[2], *(inp[2]), pool2x(net[1]))
        if iter08:
            if self.args.n_gru_layers > 2:
                net[1] = self.gru08(net[1], *(inp[1]), pool2x(net[0]), interp(net[2], net[1]))
            else:
                net[1] = self.gru08(net[1], *(inp[1]), pool2x(net[0]))
        if iter04:
            motion_features = self.encoder(disp, corr, flaw_stereo, disp_mono, corr_mono, flaw_mono)
            if self.args.n_gru_layers > 1:
                net[0] = self.gru04(net[0], *(inp[0]), motion_features, interp(net[1], net[0]))
            else:
                net[0] = self.gru04(net[0], *(inp[0]), motion_features)

        if not update:
            return net

        delta_disp_all = self.disp_head(net[0])
        delta_disp = delta_disp_all[:, :1]
        delta_disp_mono = delta_disp_all[:, 1:2]
        mask_feat_4 = self.mask_feat_4(net[0])
        return net, mask_feat_4, delta_disp, delta_disp_mono

class BasicMotionEncoder_mix(nn.Module):
    def __init__(self, args):
        super(BasicMotionEncoder_mix, self).__init__()
        self.args = args
        cor_planes = 96 + args.corr_levels * (2*args.corr_radius + 1) * (8+1)
        self.convc1 = nn.Conv2d(cor_planes, 64, 1, padding=0)
        self.convc2 = nn.Conv2d(64, 64, 3, padding=1)

        self.convc1_mono = nn.Conv2d(cor_planes, 64, 1, padding=0)
        self.convc2_mono = nn.Conv2d(64, 64, 3, padding=1)

        self.convd1 = nn.Conv2d(1, 64, 7, padding=3)
        self.convd2 = nn.Conv2d(64, 64, 3, padding=1)

        self.convd1_mono = nn.Conv2d(1, 64, 7, padding=3)
        self.convd2_mono = nn.Conv2d(64, 64, 3, padding=1)

        self.conv = nn.Conv2d(128, 64-1, 3, padding=1)
        self.conv_mono = nn.Conv2d(128, 64-1, 3, padding=1)


    def forward(self, disp, corr, flaw_stereo, disp_mono, corr_mono, flaw_mono):
        cor = F.relu(self.convc1(torch.cat([corr, flaw_stereo], dim=1)))
        cor = F.relu(self.convc2(cor))
        cor_mono = F.relu(self.convc1_mono(torch.cat([corr_mono, flaw_mono], dim=1)))
        cor_mono = F.relu(self.convc2_mono(cor_mono))

        disp_ = F.relu(self.convd1(disp))
        disp_ = F.relu(self.convd2(disp_))

        disp_mono_ = F.relu(self.convd1_mono(disp_mono))
        disp_mono_ = F.relu(self.convd2_mono(disp_mono_))

        cor_disp = torch.cat([cor, disp_], dim=1)
        cor_disp_mono = torch.cat([cor_mono, disp_mono_], dim=1)

        out = F.relu(self.conv(cor_disp))
        out_mono = F.relu(self.conv_mono(cor_disp_mono))

        return torch.cat([out, disp, out_mono, disp_mono], dim=1)


class BasicMultiUpdateBlock_2(nn.Module):
    def __init__(self, args, hidden_dims=[]):
        super().__init__()
        self.args = args
        self.encoder = BasicMotionEncoder_2(args)
        encoder_output_dim = 128

        self.gru04 = ConvGRU(hidden_dims[2], encoder_output_dim + hidden_dims[1] * (args.n_gru_layers > 1))
        self.gru08 = ConvGRU(hidden_dims[1], hidden_dims[0] * (args.n_gru_layers == 3) + hidden_dims[2])
        self.gru16 = ConvGRU(hidden_dims[0], hidden_dims[1])
        self.disp_head = DispHead(hidden_dims[2], hidden_dim=256, output_dim=1)
        factor = 2**self.args.n_downsample

        self.mask_feat_4 = nn.Sequential(
            nn.Conv2d(hidden_dims[2], 32, 3, padding=1),
            nn.ReLU(inplace=True))

    def forward(self, net, inp, flaw_stereo=None, disp=None, corr=None, confidence=None, flaw_mono=None, disp_mono=None, corr_mono=None, iter04=True, iter08=True, iter16=True, update=True):

        if iter16:
            net[2] = self.gru16(net[2], *(inp[2]), pool2x(net[1]))
        if iter08:
            if self.args.n_gru_layers > 2:
                net[1] = self.gru08(net[1], *(inp[1]), pool2x(net[0]), interp(net[2], net[1]))
            else:
                net[1] = self.gru08(net[1], *(inp[1]), pool2x(net[0]))
        if iter04:
            motion_features = self.encoder(disp, corr, flaw_stereo, disp_mono, corr_mono, flaw_mono, confidence)
            if self.args.n_gru_layers > 1:
                net[0] = self.gru04(net[0], *(inp[0]), motion_features, interp(net[1], net[0]))
            else:
                net[0] = self.gru04(net[0], *(inp[0]), motion_features)

        if not update:
            return net

        delta_disp = self.disp_head(net[0])
        mask_feat_4 = self.mask_feat_4(net[0])
        return net, mask_feat_4, delta_disp


class BasicMotionEncoder_2(nn.Module):
    def __init__(self, args):
        super(BasicMotionEncoder_2, self).__init__()
        self.args = args
        cor_planes = 96 + args.corr_levels * (2*args.corr_radius + 1) * (8+1)
        self.convc1 = nn.Conv2d(int(cor_planes + 1), 64, 1, padding=0)
        self.convc2 = nn.Conv2d(64, 64, 3, padding=1)

        self.convc1_mono = nn.Conv2d(cor_planes, 64, 1, padding=0)
        self.convc2_mono = nn.Conv2d(64, 64, 3, padding=1)

        self.convd1 = nn.Conv2d(1, 64, 7, padding=3)
        self.convd2 = nn.Conv2d(64, 64, 3, padding=1)

        self.convd1_mono = nn.Conv2d(1, 64, 7, padding=3)
        self.convd2_mono = nn.Conv2d(64, 64, 3, padding=1)

        self.conv = nn.Conv2d(129, 64-1, 3, padding=1)
        self.conv_mono = nn.Conv2d(128, 64-1, 3, padding=1)


    def forward(self, disp, corr, flaw_stereo, disp_mono, corr_mono, flaw_mono, confidence):
        cor = F.relu(self.convc1(torch.cat([corr, flaw_stereo, confidence], dim=1)))
        cor = F.relu(self.convc2(cor))
        cor_mono = F.relu(self.convc1_mono(torch.cat([corr_mono, flaw_mono], dim=1)))
        cor_mono = F.relu(self.convc2_mono(cor_mono))

        disp_ = F.relu(self.convd1(disp))
        disp_ = F.relu(self.convd2(disp_))

        disp_mono_ = F.relu(self.convd1_mono(disp_mono))
        disp_mono_ = F.relu(self.convd2_mono(disp_mono_))

        cor_disp = torch.cat([cor, disp_, confidence], dim=1)
        cor_disp_mono = torch.cat([cor_mono, disp_mono_], dim=1)

        out = F.relu(self.conv(cor_disp))
        out_mono = F.relu(self.conv_mono(cor_disp_mono))

        return torch.cat([out, disp, out_mono, disp_mono], dim=1)
    

class BasicMultiUpdateBlock_mono(nn.Module):
    def __init__(self, args, hidden_dims=[]):
        super().__init__()
        self.args = args
        self.encoder = BasicMotionEncoder_mono(args)
        encoder_output_dim = 128

        self.gru04 = ConvGRU(hidden_dims[2], encoder_output_dim + hidden_dims[1] * (args.n_gru_layers > 1))
        self.gru08 = ConvGRU(hidden_dims[1], hidden_dims[0] * (args.n_gru_layers == 3) + hidden_dims[2])
        self.gru16 = ConvGRU(hidden_dims[0], hidden_dims[1])
        self.disp_head = DispHead(hidden_dims[2], hidden_dim=256, output_dim=1)
        factor = 2**self.args.n_downsample

        self.mask_feat_4 = nn.Sequential(
            nn.Conv2d(hidden_dims[2], 32, 3, padding=1),
            nn.ReLU(inplace=True))

    def forward(self, net, inp, corr=None, disp=None, iter04=True, iter08=True, iter16=True, update=True):

        if iter16:
            net[2] = self.gru16(net[2], *(inp[2]), pool2x(net[1]))
        if iter08:
            if self.args.n_gru_layers > 2:
                net[1] = self.gru08(net[1], *(inp[1]), pool2x(net[0]), interp(net[2], net[1]))
            else:
                net[1] = self.gru08(net[1], *(inp[1]), pool2x(net[0]))
        if iter04:
            motion_features = self.encoder(disp, corr)
            if self.args.n_gru_layers > 1:
                net[0] = self.gru04(net[0], *(inp[0]), motion_features, interp(net[1], net[0]))
            else:
                net[0] = self.gru04(net[0], *(inp[0]), motion_features)

        if not update:
            return net

        delta_disp = self.disp_head(net[0])
        mask_feat_4 = self.mask_feat_4(net[0])
        return net, mask_feat_4, delta_disp


class BasicMotionEncoder_mono(nn.Module):
    def __init__(self, args):
        super(BasicMotionEncoder_mono, self).__init__()
        self.args = args
        cor_planes = args.corr_levels * (2*args.corr_radius + 1) * (8+1)
        self.convc1 = nn.Conv2d(cor_planes, 64, 1, padding=0)
        self.convc2 = nn.Conv2d(64, 64, 3, padding=1)
        self.convd1 = nn.Conv2d(1, 64, 7, padding=3)
        self.convd2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv = nn.Conv2d(64+64, 128-1, 3, padding=1)

    def forward(self, disp, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        disp_ = F.relu(self.convd1(disp))
        disp_ = F.relu(self.convd2(disp_))

        cor_disp = torch.cat([cor, disp_], dim=1)
        out = F.relu(self.conv(cor_disp))
        return torch.cat([out, disp], dim=1)


class BasicMultiUpdateBlock_mix_conf(nn.Module):
    def __init__(self, args, hidden_dims=[]):
        super().__init__()
        self.args = args
        self.encoder = BasicMotionEncoder_mix_conf(args)
        encoder_output_dim = 128

        self.gru04 = ConvGRU(hidden_dims[2], encoder_output_dim + hidden_dims[1] * (args.n_gru_layers > 1))
        self.gru08 = ConvGRU(hidden_dims[1], hidden_dims[0] * (args.n_gru_layers == 3) + hidden_dims[2])
        self.gru16 = ConvGRU(hidden_dims[0], hidden_dims[1])
        self.disp_head = DispHead(hidden_dims[2], hidden_dim=256, output_dim=2)
        factor = 2**self.args.n_downsample

        self.mask_feat_4 = nn.Sequential(
            nn.Conv2d(hidden_dims[2], 32, 3, padding=1),
            nn.ReLU(inplace=True))

    def forward(self, net, inp, flaw_stereo=None, disp=None, corr=None, flaw_mono=None, disp_mono=None, corr_mono=None, conf_stereo=None, conf_mono=None, iter04=True, iter08=True, iter16=True, update=True):

        if iter16:
            net[2] = self.gru16(net[2], *(inp[2]), pool2x(net[1]))
        if iter08:
            if self.args.n_gru_layers > 2:
                net[1] = self.gru08(net[1], *(inp[1]), pool2x(net[0]), interp(net[2], net[1]))
            else:
                net[1] = self.gru08(net[1], *(inp[1]), pool2x(net[0]))
        if iter04:
            motion_features = self.encoder(disp, corr, flaw_stereo, disp_mono, corr_mono, flaw_mono, conf_stereo, conf_mono)
            if self.args.n_gru_layers > 1:
                net[0] = self.gru04(net[0], *(inp[0]), motion_features, interp(net[1], net[0]))
            else:
                net[0] = self.gru04(net[0], *(inp[0]), motion_features)

        if not update:
            return net

        delta_disp_all = self.disp_head(net[0])
        delta_disp = delta_disp_all[:, :1]
        delta_disp_mono = delta_disp_all[:, 1:2]
        mask_feat_4 = self.mask_feat_4(net[0])
        return net, mask_feat_4, delta_disp, delta_disp_mono
    

class BasicMotionEncoder_mix_conf(nn.Module):
    def __init__(self, args):
        super(BasicMotionEncoder_mix_conf, self).__init__()
        self.args = args
        cor_planes = 96 + args.corr_levels * (2*args.corr_radius + 1) * (8+1)

        self.conv_conf1 = nn.Conv2d(1, 64, 7, padding=3)
        self.conv_conf2 = nn.Conv2d(64, 64, 3, padding=1)

        self.conv_conf1_mono = nn.Conv2d(1, 64, 7, padding=3)
        self.conv_conf2_mono = nn.Conv2d(64, 64, 3, padding=1)

        self.convc1 = nn.Conv2d(int(cor_planes + 64), 64, 1, padding=0)
        self.convc2 = nn.Conv2d(64, 64, 3, padding=1)

        self.convc1_mono = nn.Conv2d(int(cor_planes + 64), 64, 1, padding=0)
        self.convc2_mono = nn.Conv2d(64, 64, 3, padding=1)

        self.convd1 = nn.Conv2d(1, 64, 7, padding=3)
        self.convd2 = nn.Conv2d(64, 64, 3, padding=1)

        self.convd1_mono = nn.Conv2d(1, 64, 7, padding=3)
        self.convd2_mono = nn.Conv2d(64, 64, 3, padding=1)

        self.conv = nn.Conv2d(128, 64-2, 3, padding=1)
        self.conv_mono = nn.Conv2d(128, 64-2, 3, padding=1)


    def forward(self, disp, corr, flaw_stereo, disp_mono, corr_mono, flaw_mono, conf_stereo, conf_mono):

        conf_stereo_ = F.relu(self.conv_conf1(conf_stereo))
        conf_stereo_ = F.relu(self.conv_conf2(conf_stereo_))

        conf_mono_ = F.relu(self.conv_conf1_mono(conf_mono))
        conf_mono_ = F.relu(self.conv_conf2_mono(conf_mono_))

        cor = F.relu(self.convc1(torch.cat([corr, flaw_stereo, conf_stereo_], dim=1)))
        cor = F.relu(self.convc2(cor))
        cor_mono = F.relu(self.convc1_mono(torch.cat([corr_mono, flaw_mono, conf_mono_], dim=1)))
        cor_mono = F.relu(self.convc2_mono(cor_mono))

        disp_ = F.relu(self.convd1(disp))
        disp_ = F.relu(self.convd2(disp_))

        disp_mono_ = F.relu(self.convd1_mono(disp_mono))
        disp_mono_ = F.relu(self.convd2_mono(disp_mono_))

        cor_disp = torch.cat([cor, disp_], dim=1)
        cor_disp_mono = torch.cat([cor_mono, disp_mono_], dim=1)

        out = F.relu(self.conv(cor_disp))
        out_mono = F.relu(self.conv_mono(cor_disp_mono))

        return torch.cat([out, disp, conf_stereo, out_mono, disp_mono, conf_mono], dim=1)


class BasicMultiUpdateBlock_mix2(nn.Module):
    def __init__(self, args, hidden_dims=[]):
        super().__init__()
        self.args = args
        self.encoder = BasicMotionEncoder_mix2(args)
        encoder_output_dim = 128

        self.gru04 = ConvGRU(hidden_dims[2], encoder_output_dim + hidden_dims[1] * (args.n_gru_layers > 1))
        self.gru08 = ConvGRU(hidden_dims[1], hidden_dims[0] * (args.n_gru_layers == 3) + hidden_dims[2])
        self.gru16 = ConvGRU(hidden_dims[0], hidden_dims[1])
        self.disp_head = DispHead(hidden_dims[2], hidden_dim=256, output_dim=1)
        factor = 2**self.args.n_downsample

        self.mask_feat_4 = nn.Sequential(
            nn.Conv2d(hidden_dims[2], 32, 3, padding=1),
            nn.ReLU(inplace=True))

    def forward(self, net, inp, flaw_stereo=None, disp=None, corr=None, flaw_mono=None, disp_mono=None, corr_mono=None, iter04=True, iter08=True, iter16=True, update=True):

        if iter16:
            net[2] = self.gru16(net[2], *(inp[2]), pool2x(net[1]))
        if iter08:
            if self.args.n_gru_layers > 2:
                net[1] = self.gru08(net[1], *(inp[1]), pool2x(net[0]), interp(net[2], net[1]))
            else:
                net[1] = self.gru08(net[1], *(inp[1]), pool2x(net[0]))
        if iter04:
            motion_features = self.encoder(disp, corr, flaw_stereo, disp_mono, corr_mono, flaw_mono)
            if self.args.n_gru_layers > 1:
                net[0] = self.gru04(net[0], *(inp[0]), motion_features, interp(net[1], net[0]))
            else:
                net[0] = self.gru04(net[0], *(inp[0]), motion_features)

        if not update:
            return net
        delta_disp = self.disp_head(net[0])
        mask_feat_4 = self.mask_feat_4(net[0])
        return net, mask_feat_4, delta_disp

class BasicMotionEncoder_mix2(nn.Module):
    def __init__(self, args):
        super(BasicMotionEncoder_mix2, self).__init__()
        self.args = args
        cor_planes = 96 + args.corr_levels * (2*args.corr_radius + 1) * (8+1)
        self.convc1 = nn.Conv2d(cor_planes, 64, 1, padding=0)
        self.convc2 = nn.Conv2d(64, 64, 3, padding=1)

        self.convc1_mono = nn.Conv2d(cor_planes, 64, 1, padding=0)
        self.convc2_mono = nn.Conv2d(64, 64, 3, padding=1)

        self.convd1 = nn.Conv2d(1, 64, 7, padding=3)
        self.convd2 = nn.Conv2d(64, 64, 3, padding=1)

        self.convd1_mono = nn.Conv2d(1, 64, 7, padding=3)
        self.convd2_mono = nn.Conv2d(64, 64, 3, padding=1)

        self.conv = nn.Conv2d(128, 64-1, 3, padding=1)
        self.conv_mono = nn.Conv2d(128, 64-1, 3, padding=1)


    def forward(self, disp, corr, flaw_stereo, disp_mono, corr_mono, flaw_mono):
        cor = F.relu(self.convc1(torch.cat([corr, flaw_stereo], dim=1)))
        cor = F.relu(self.convc2(cor))
        cor_mono = F.relu(self.convc1_mono(torch.cat([corr_mono, flaw_mono], dim=1)))
        cor_mono = F.relu(self.convc2_mono(cor_mono))

        disp_ = F.relu(self.convd1(disp))
        disp_ = F.relu(self.convd2(disp_))

        disp_mono_ = F.relu(self.convd1_mono(disp_mono))
        disp_mono_ = F.relu(self.convd2_mono(disp_mono_))

        cor_disp = torch.cat([cor, disp_], dim=1)
        cor_disp_mono = torch.cat([cor_mono, disp_mono_], dim=1)

        out = F.relu(self.conv(cor_disp))
        out_mono = F.relu(self.conv_mono(cor_disp_mono))

        return torch.cat([out, disp, out_mono, disp_mono], dim=1)