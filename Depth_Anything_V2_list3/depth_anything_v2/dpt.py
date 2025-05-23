import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose

from .dinov2 import DINOv2
from .util.blocks import FeatureFusionBlock, _make_scratch
from .util.transform import Resize, NormalizeImage, PrepareForNet


def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class ConvBlock(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_feature, out_feature, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_feature),
            nn.ReLU(True)
        )
    
    def forward(self, x):
        return self.conv_block(x)


class DPTHead(nn.Module):
    def __init__(
        self, 
        in_channels, 
        features=256, 
        use_bn=False, 
        out_channels=[256, 512, 1024, 1024], 
        use_clstoken=False
    ):
        super(DPTHead, self).__init__()
        
        self.use_clstoken = use_clstoken

        # 生成四个独立的project，每个独立的project的输出通道都不同，根据实际情况使用具体第几个project
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        
        self.resize_layers = nn.ModuleList([          # 也是个列表，按照情况选择其中具体哪一个使用
            nn.ConvTranspose2d(                       # 转置卷积层，上采样使用，不同的kernel_size相当于放大不同的倍数
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(), # 恒等层，对输入不做任何处理，具体为什么呢，不知道
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])
        
        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))
        
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )
        
        self.scratch.stem_transpose = None
        
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)
        
        head_features_1 = features
        head_features_2 = 32
        
        self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Identity(),
        )
    
    def forward(self, out_features, patch_h, patch_w, only_feat=False):
        out = []
        # 在deepanythingv2中的decoder中只用了vit的[B, N, D]特征输出
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0] # 实际用了这段代码

            # [B, N, D] -> [B, D, N] -> [B, D, patch_h, patch_w]
            # 炫耀先permute的原因：N一般代表图像块的总数，D这一列代表每个图像块的特征维度（特征向量）
            #                   x一般是按照行优先存储，越右就越临近
            #                   reshape分割是行优先分割，我们希望分割的是N，序列（或块）的存放形状，而非将特征分割
            #                   所以需要先permute再reshape
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))

            # [256, 512, 1024, 1024]
            # projects[i] 的输出张量的通道数量对应上面4个
            x = self.projects[i](x) # [B, D, patch_h, patch_w] -> [B, C, patch_h, patch_w] 应该相当于特征通道调整吧
            x = self.resize_layers[i](x)
            
            out.append(x)
        
        layer_1, layer_2, layer_3, layer_4 = out

        # print('layer_1', layer_1.shape)
        # print('layer_2', layer_2.shape)
        # print('layer_3', layer_3.shape)
        # print('layer_4', layer_4.shape)
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])        
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        # feat_4 = self.scratch.refinenet4(layer_4_rn, size=layer_4_rn.shape[2:])
        # up_feat_4 = F.interpolate(feat_4, size=layer_3_rn.shape[2:], mode='bilinear', align_corners=True)  
        # feat_3 = self.scratch.refinenet3(up_feat_4, layer_3_rn, size=layer_3_rn.shape[2:])
        # up_feat_3 = F.interpolate(feat_3, size=layer_2_rn.shape[2:], mode='bilinear', align_corners=True)
        # feat_2 = self.scratch.refinenet2(up_feat_3, layer_2_rn, size=layer_2_rn.shape[2:])
        # up_feat_2 = F.interpolate(feat_2, size=layer_1_rn.shape[2:], mode='bilinear', align_corners=True)
        # feat_1 = self.scratch.refinenet1(up_feat_2, layer_1_rn, size=layer_1_rn.shape[2:])

        
        out = self.scratch.output_conv1(path_1)
        out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        # features_decoder = out.clone()
        out = self.scratch.output_conv2(out)
        
        return out


class DPTHead_decoder(nn.Module):
    def __init__(
        self, 
        in_channels, 
        features=256, 
        use_bn=False, 
        out_channels=[256, 512, 1024, 1024], 
        use_clstoken=False
    ):
        super(DPTHead_decoder, self).__init__()
        
        self.use_clstoken = use_clstoken

        ###################################################################
        # 特征通道数对齐模块
        # 作用：将不同阶段特征图的通道数映射到目标维度（out_channels）
        # 结构：四个独立的1x1卷积层，用于通道数调整
        ###################################################################
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])
        
        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))
        
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )
        
        self.scratch.stem_transpose = None
        
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)
        
        # head_features_1 = features
        # head_features_2 = 32
        
        # self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
        # self.scratch.output_conv2 = nn.Sequential(
        #     nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(True),
        #     nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
        #     nn.ReLU(True),
        #     nn.Identity(),
        # )
    
    def forward(self, out_features, patch_h, patch_w):
        out = []
        # 遍历四个不同的特征图
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]
            
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            
            out.append(x)
        
        layer_1, layer_2, layer_3, layer_4 = out

        # print('layer_1', layer_1.shape)
        # print('layer_2', layer_2.shape)
        # print('layer_3', layer_3.shape)
        # print('layer_4', layer_4.shape)
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        # path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])        
        # path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        # path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        # path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        # todo:why
        # 具体来讲呢就是depth_anything_decoder中的depth_head_decoder是输出同层的分辨率大小，然后经过手动上采样
        # 就是refinenet是内部特征的融合，上采样是为了保留感受野较小的特征信息，而是否上采样也是和DPTHead的区别
        # 然后就是DPTHead只是前馈特征
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_4_rn.shape[2:])
        up_path_4 = F.interpolate(path_4, size=layer_3_rn.shape[2:], mode='bilinear', align_corners=True)  
        path_3 = self.scratch.refinenet3(up_path_4, layer_3_rn, size=layer_3_rn.shape[2:])
        up_path_3 = F.interpolate(path_3, size=layer_2_rn.shape[2:], mode='bilinear', align_corners=True)
        path_2 = self.scratch.refinenet2(up_path_3, layer_2_rn, size=layer_2_rn.shape[2:])
        up_path_2 = F.interpolate(path_2, size=layer_1_rn.shape[2:], mode='bilinear', align_corners=True)
        path_1 = self.scratch.refinenet1(up_path_2, layer_1_rn, size=layer_1_rn.shape[2:])

        return path_1, path_2, path_3, path_4

        # feat_4 = self.scratch.refinenet4(layer_4_rn, size=layer_4_rn.shape[2:])
        # up_feat_4 = F.interpolate(feat_4, size=layer_3_rn.shape[2:], mode='bilinear', align_corners=True)  
        # feat_3 = self.scratch.refinenet3(up_feat_4, layer_3_rn, size=layer_3_rn.shape[2:])
        # up_feat_3 = F.interpolate(feat_3, size=layer_2_rn.shape[2:], mode='bilinear', align_corners=True)
        # feat_2 = self.scratch.refinenet2(up_feat_3, layer_2_rn, size=layer_2_rn.shape[2:])
        # up_feat_2 = F.interpolate(feat_2, size=layer_1_rn.shape[2:], mode='bilinear', align_corners=True)
        # feat_1 = self.scratch.refinenet1(up_feat_2, layer_1_rn, size=layer_1_rn.shape[2:])

        # if only_feat:
        #     return [layer_4, layer_3, layer_2, layer_1]
        # else:
        
        #     out = self.scratch.output_conv1(path_1)
        #     out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        #     # features_decoder = out.clone()
        #     out = self.scratch.output_conv2(out)
            
        #     return out, [layer_4, layer_3, layer_2, layer_1]

# 原版DepthAnythingV2实现
class DepthAnythingV2(nn.Module):
    def __init__(
        self, 
        encoder='vitl', 
        features=256, 
        out_channels=[256, 512, 1024, 1024], 
        use_bn=False, 
        use_clstoken=False
    ):
        super(DepthAnythingV2, self).__init__()
        
        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11], 
            'vitl': [4, 11, 17, 23], 
            'vitg': [9, 19, 29, 39]
        }
        
        self.encoder = encoder
        self.pretrained = DINOv2(model_name=encoder)

        # print('self.pretrained.embed_dim', self.pretrained.embed_dim)
        print('DepthAnythingV2 out_channels', out_channels)
        
        self.depth_head = DPTHead(self.pretrained.embed_dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken)
    
    def forward(self, x):
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
        
        features = self.pretrained.get_intermediate_layers(x, self.intermediate_layer_idx[self.encoder], return_class_token=True)
        
        depth, features_decoder = self.depth_head(features, patch_h, patch_w)
        depth = F.relu(depth)
        
        return depth.squeeze(1)

    def forward_features(self, x, only_feat=False):
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
        
        features = self.pretrained.get_intermediate_layers(x, self.intermediate_layer_idx[self.encoder], return_class_token=True)
        if only_feat:
            features_decoder = self.depth_head(features, patch_h, patch_w, only_feat)
            return features_decoder
        else:
            depth, features_decoder = self.depth_head(features, patch_h, patch_w, only_feat)
            depth = F.relu(depth)
            
            return depth.squeeze(1), features_decoder
    
    @torch.no_grad()
    def infer_image(self, raw_image, input_size=518):
        image, (h, w) = self.image2tensor(raw_image, input_size)
        
        depth = self.forward(image)
        
        depth = F.interpolate(depth[:, None], (h, w), mode="bilinear", align_corners=True)[0, 0]
        
        return depth.cpu().numpy()
    
    def image2tensor(self, raw_image, input_size=518):        
        transform = Compose([
            Resize(
                width=input_size,
                height=input_size,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])
        
        h, w = raw_image.shape[:2]
        
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
        
        image = transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0)
        
        DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        image = image.to(DEVICE)
        
        return image, (h, w)

# 只留下了DepthAnythingV2的decoder模块，在具体使用的时候只用了DPTHead_decoder这个层的结构，连forward的两个方法都没用
class DepthAnythingV2_decoder(nn.Module):
    def __init__(
        self, 
        encoder='vitl', 
        features=256, 
        out_channels=[256, 512, 1024, 1024], 
        use_bn=False, 
        use_clstoken=False
    ):
        super(DepthAnythingV2_decoder, self).__init__()
        
        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11], 
            'vitl': [4, 11, 17, 23], 
            'vitg': [9, 19, 29, 39]
        }
        print('DepthAnythingV2_decoder out_channels', out_channels)
        
        # self.encoder = encoder
        # self.pretrained = DINOv2(model_name=encoder)
        
        self.depth_head = DPTHead_decoder(out_channels[-1], features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken)
    
    def forward(self, x):
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
        
        features = self.pretrained.get_intermediate_layers(x, self.intermediate_layer_idx[self.encoder], return_class_token=True)
        
        depth, features_decoder = self.depth_head(features, patch_h, patch_w)
        depth = F.relu(depth)
        
        return depth.squeeze(1)

    def forward_features(self, x, only_feat=False):
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
        
        features = self.pretrained.get_intermediate_layers(x, self.intermediate_layer_idx[self.encoder], return_class_token=True)
        if only_feat:
            features_decoder = self.depth_head(features, patch_h, patch_w, only_feat)
            return features_decoder
        else:
            depth, features_decoder = self.depth_head(features, patch_h, patch_w, only_feat)
            depth = F.relu(depth)
            
            return depth.squeeze(1), features_decoder
    
    @torch.no_grad()
    def infer_image(self, raw_image, input_size=518):
        image, (h, w) = self.image2tensor(raw_image, input_size)
        
        depth = self.forward(image)
        
        depth = F.interpolate(depth[:, None], (h, w), mode="bilinear", align_corners=True)[0, 0]
        
        return depth.cpu().numpy()
    
    def image2tensor(self, raw_image, input_size=518):        
        transform = Compose([
            Resize(
                width=input_size,
                height=input_size,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])
        
        h, w = raw_image.shape[:2]
        
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
        
        image = transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0)
        
        DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        image = image.to(DEVICE)
        
        return image, (h, w)

