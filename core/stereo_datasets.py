import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import logging
import os
import re
import copy
import math
import random
from pathlib import Path
from glob import glob
import os.path as osp

import sys
sys.path.append(os.getcwd())

from core.utils import frame_utils
from core.utils.augmentor import FlowAugmentor, SparseFlowAugmentor


class StereoDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False, reader=None):
        """立体匹配数据集基类

                参数说明：
                - aug_params (dict, optional): 数据增强参数配置字典。包含如：
                    - crop_size: 裁剪尺寸 [H, W]
                    - color_aug: 颜色增强参数
                    - img_pad: 图像填充尺寸（特殊键，会被提取到 self.img_pad）
                - sparse (bool): 是否处理稀疏视差数据（真值非全图稠密）
                - reader (callable, optional): 自定义视差读取函数，默认使用 PFM/PNG 读取器
        """

        # ------------------------------
        # 初始化数据增强模块
        # ------------------------------
        self.augmentor = None # 先初始化数据增强器实例为none
        self.sparse = sparse # 根据sparse设置是否为稀疏视差模式
        self.img_pad = aug_params.pop("img_pad", None) if aug_params is not None else None # 从 aug_params 中提取 img_pad 并移除（避免冲突）

        # 当aug_params非空且指定了裁剪尺寸时，初始化对应的增强器
        if aug_params is not None and "crop_size" in aug_params:
            # 稀疏数据使用 SparseFlowAugmentor（处理有效掩码） 就是真正的视差是部分有效还是全部有效
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params) # 稠密数据使用标准 FlowAugmentor

        # ------------------------------
        # 视差读取函数配置
        # ------------------------------
        if reader is None:
            # 默认视差读取器（支持 PFM/PNG 格式）
            self.disparity_reader = frame_utils.read_gen
        else:
            # 允许用户自定义读取逻辑（如处理特殊格式）
            self.disparity_reader = reader

        # ------------------------------
        # 数据集状态标记
        # ------------------------------
        self.is_test = False # 是否为测试模式（禁用增强/真值加载）
        self.init_seed = False # 随机种子初始化标记（用于多进程数据加载）

        # ------------------------------
        # 数据路径存储列表 实际在子类中实现
        # ------------------------------
        self.flow_list = [] # 光流文件路径（立体匹配中通常为空）
        self.disparity_list = [] # 视差真值文件路径列表
        self.image_list = [] # 图像对路径列表，格式 [[左图路径, 右图路径], ...]
        self.extra_info = [] # 附加信息（如场景名称、相机参数等）

    def __getitem__(self, index):

        # 测试模式处理
        if self.is_test:
            # 读取左右视图图像（测试模式无需增强）
            img1 = frame_utils.read_gen(self.image_list[index][0]) # 左视图路径 → PIL Image
            img2 = frame_utils.read_gen(self.image_list[index][1]) # 右视图路径 → PIL Image
            # 转换为NumPy数组并截取前3通道（兼容RGBA等格式）
            img1 = np.array(img1).astype(np.uint8)[..., :3] # H x W x 3
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            # 转换为PyTorch张量并调整维度顺序 [C, H, W]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float() # 3 x H x W, 值域[0,255]
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index] # 返回图像+元信息（如场景名）

        # 训练模式初始化
        # 多进程数据加载的随机种子设置（确保每个worker独立随机性）
        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info() # 获取当前数据加载worker信息
            if worker_info is not None:
                # 设置随机种子（基于worker ID）
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True  # 仅初始化一次

        # 循环索引（防止超出数据范围，配合数据增强倍数）
        index = index % len(self.image_list) # 循环索引（防止超出数据范围，配合数据增强倍数）

        # 视差数据加载
        # 循环索引（防止超出数据范围，配合数据增强倍数）
        disp = self.disparity_reader(self.disparity_list[index]) # 返回真实的视差图或（视差, 有效掩码）

        # 分离视差与有效掩码（ETH3D等数据集的真值为稀疏标注）
        if isinstance(disp, tuple):
            disp, valid = disp # valid为布尔掩码（标记有效视差区域）
        else:
            valid = disp < 512 # 生成有效掩码（假设视差>512为无效值）

        # 图像读取与格式处理
        img1 = frame_utils.read_gen(self.image_list[index][0]) # 左视图 → PIL Image
        img2 = frame_utils.read_gen(self.image_list[index][1]) # 右视图 → PIL Image

        # 转换为NumPy数组（H x W x 3）
        img1 = np.array(img1).astype(np.uint8) # 值域[0,255]
        img2 = np.array(img2).astype(np.uint8)

        # 转换为浮点型视差数据（单位：像素）
        disp = np.array(disp).astype(np.float32)

        # 构造光流张量（立体匹配任务中垂直流为0）
        flow = np.stack([disp, np.zeros_like(disp)], axis=-1) # H x W x 2

        # grayscale images
        # # 处理单通道输入（如Middlebury数据集）
        if len(img1.shape) == 2: # 灰度图（ H x W ）
            img1 = np.tile(img1[...,None], (1, 1, 3)) # 复制为H x W x3
            img2 = np.tile(img2[...,None], (1, 1, 3))
        else: # 多通道图像截取前3通道（兼容RGBA等格式）
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        # 数据增强
        if self.augmentor is not None: # 启用数据增强（如旋转、裁剪、颜色抖动）
            if self.sparse: # 稀疏视差数据需同步增强有效掩码
                img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            else: # 密集视差数据无需处理掩码
                img1, img2, flow = self.augmentor(img1, img2, flow)

        # 图像转Tensor [C, H, W] 并归一化到[0,1]（除以255）
        img1 = torch.from_numpy(img1).permute(2, 0, 1).float() # 输出 3 x H x W
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        # 光流转Tensor [C, H, W]
        flow = torch.from_numpy(flow).permute(2, 0, 1).float() # 输出 2xHxW

        # 有效掩码处理（稀疏数据直接转Tensor，密集数据根据阈值生成）
        if self.sparse:
            valid = torch.from_numpy(valid) # 直接使用标注的掩码
        else:
            valid = (flow[0].abs() < 512) & (flow[1].abs() < 512) # 直接使用标注的掩码

        if self.img_pad is not None: # 填充尺寸（如[384, 512]）

            padH, padW = self.img_pad
            # 对称填充（上下padH，左右padW）
            img1 = F.pad(img1, [padW]*2 + [padH]*2) # 左右填充 → 上下填充
            img2 = F.pad(img2, [padW]*2 + [padH]*2)

        # 光流仅保留水平分量（立体匹配任务垂直流恒为0）
        flow = flow[:1] # 输出 1 x H x W

        # 返回数据项：路径、图像、光流、有效掩码
        # self.image_list[index] + [self.disparity_list[index]],  # 路径信息（左图、右图、视差路径）
        # img1,  # 左视图张量 3xHxW
        # img2,  # 右视图张量 3xHxW
        # flow,  # 视差张量 1xHxW
        # valid.float()  # 有效区域掩码 1xHxW（1有效/0无效）
        return self.image_list[index] + [self.disparity_list[index]], img1, img2, flow, valid.float()


    def __mul__(self, v):
        # 创建当前对象的深拷贝（避免修改原始数据）
        copy_of_self = copy.deepcopy(self)
        # 对关键数据列表进行复制扩展
        copy_of_self.flow_list = v * copy_of_self.flow_list
        copy_of_self.image_list = v * copy_of_self.image_list
        copy_of_self.disparity_list = v * copy_of_self.disparity_list
        copy_of_self.extra_info = v * copy_of_self.extra_info
        return copy_of_self
        
    def __len__(self):
        return len(self.image_list)


class SceneFlowDatasets(StereoDataset):
    def __init__(self, aug_params=None, root='/data2/cjd/StereoDatasets/sceneflow', dstype='frames_finalpass', things_test=False):
        super(SceneFlowDatasets, self).__init__(aug_params)
        assert os.path.exists(root)
        self.root = root
        self.dstype = dstype

        if things_test:
            self._add_things("TEST")
        else:
            self._add_things("TRAIN")
            self._add_monkaa("TRAIN")
            self._add_driving("TRAIN")

    def _add_things(self, split='TRAIN'):
        """ Add FlyingThings3D data """

        original_length = len(self.disparity_list)
        root = self.root
        # left_images = sorted( glob(osp.join(root, self.dstype, split, '*/*/left/*.png')) )
        # right_images = [ im.replace('left', 'right') for im in left_images ]
        right_images = sorted( glob(osp.join(root, self.dstype, split, '*/*/right/*.png')) )
        left_images = [ im.replace('right', 'left') for im in right_images ]
        disparity_images = [ im.replace(self.dstype, 'disparity').replace('.png', '.pfm') for im in left_images ]

        for idx, (img1, img2, disp) in enumerate(zip(left_images, right_images, disparity_images)):
                self.image_list += [ [img1, img2] ]
                self.disparity_list += [ disp ]
        logging.info(f"Added {len(self.disparity_list) - original_length} from FlyingThings {self.dstype}")

    def _add_monkaa(self, split="TRAIN"):
        """ Add FlyingThings3D data """

        original_length = len(self.disparity_list)
        root = self.root
        # left_images = sorted( glob(osp.join(root, self.dstype, split, '*/left/*.png')) )
        # right_images = [ image_file.replace('left', 'right') for image_file in left_images ]
        right_images = sorted( glob(osp.join(root, self.dstype, split, '*/*/right/*.png')) )
        left_images = [ im.replace('right', 'left') for im in right_images ]
        disparity_images = [ im.replace(self.dstype, 'disparity').replace('.png', '.pfm') for im in left_images ]

        for img1, img2, disp in zip(left_images, right_images, disparity_images):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]
        logging.info(f"Added {len(self.disparity_list) - original_length} from Monkaa {self.dstype}")


    def _add_driving(self, split="TRAIN"):
        """ Add FlyingThings3D data """

        original_length = len(self.disparity_list)
        root = self.root
        # left_images = sorted( glob(osp.join(root, self.dstype, split, '*/*/*/left/*.png')) )
        # right_images = [ image_file.replace('left', 'right') for image_file in left_images ]
        right_images = sorted( glob(osp.join(root, self.dstype, split, '*/*/right/*.png')) )
        left_images = [ im.replace('right', 'left') for im in right_images ]
        disparity_images = [ im.replace(self.dstype, 'disparity').replace('.png', '.pfm') for im in left_images ]

        for img1, img2, disp in zip(left_images, right_images, disparity_images):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]
        logging.info(f"Added {len(self.disparity_list) - original_length} from Driving {self.dstype}")


class ETH3D(StereoDataset):
    def __init__(self, aug_params=None, root='/home/lxy/dataset/ETH3D', split='training'):
        """
                ETH3D 数据集加载类（专用于双视图立体匹配任务）

                参数说明：
                - aug_params: 数据增强参数（继承自父类 StereoDataset）
                - root: 数据集根目录（对应图片中下载的所有数据存放路径） 绝对路径
                - split: 数据集划分，可选 'training' 或 'test'
        """
        # 初始化父类 StereoDataset，设置稀疏视差（sparse=True 表示真值视差可能不连续）
        super(ETH3D, self).__init__(aug_params, sparse=True)
        # 调试代码：打印实际检查的路径和是否存在
        print("[Debug] 检查的绝对路径:", root)
        print("[Debug] 路径是否存在:", os.path.exists(root))
        assert os.path.exists(root), f"数据集路径 {root} 不存在，请检查下载路径"

        # ------------------------------
        # 加载双视图图像对
        # ------------------------------
        # 对应图片中的 "Download all undistorted.jpg images(13.6 MB)"
        # 路径结构：two_view_{split}/场景名/im0.png (左视图), im1.png (右视图)
        image1_list = sorted( glob(osp.join(root, f'two_view_{split}/*/im0.png')) )   # 左视图列表
        image2_list = sorted( glob(osp.join(root, f'two_view_{split}/*/im1.png')) )   # 右视图列表

        # ------------------------------
        # 加载视差真值（仅训练集有公开真值）
        # ------------------------------
        # 对应图片中的 "Download all ground truth (1.8 GB)"
        # 测试集真值未公开，使用占位符（playground_1l 的视差文件）
        # 实际评估需通过 ETH3D 官方提交系统
        # 测试集真值未公开，使用占位符（playground_1l 的视差文件）
        # 实际评估需通过 ETH3D 官方提交系统
        disp_list = sorted( glob(osp.join(root, 'two_view_training_gt/*/disp0GT.pfm')) ) if split == 'training' else [osp.join(root, 'two_view_training_gt/playground_1l/disp0GT.pfm')] * len(image1_list)

        # ------------------------------
        # 存储数据路径到列表
        # ------------------------------
        # 初始化存储容器（继承自 StereoDataset）
        # self.image_list = []  # 存储图像对路径，格式 [[左图1, 右图1], [左图2, 右图2], ...]
        # self.disparity_list = []  # 存储视差真值路径，格式 [视差1, 视差2, ...]

        # 遍历每个场景的数据路径
        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            self.image_list += [ [img1, img2] ]  # 添加当前场景的图像对
            self.disparity_list += [ disp ]      # 添加当前场景的视差真值

        # ------------------------------
        # 注意事项
        # ------------------------------
        # 1. 代码未使用图片中提到的遮挡文件（1.2 GB），若需使用需手动加载并处理遮挡掩膜
        # 2. "distorted" 数据（4.7 GB/12.1 GB）未使用，仅处理未失真图像

class SintelStereo(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/SintelStereo'):
        super().__init__(aug_params, sparse=True, reader=frame_utils.readDispSintelStereo)

        image1_list = sorted( glob(osp.join(root, 'training/*_left/*/frame_*.png')) )
        image2_list = sorted( glob(osp.join(root, 'training/*_right/*/frame_*.png')) )
        disp_list = sorted( glob(osp.join(root, 'training/disparities/*/frame_*.png')) ) * 2

        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            assert img1.split('/')[-2:] == disp.split('/')[-2:]
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]

class FallingThings(StereoDataset):
    def __init__(self, aug_params=None, root='/data2/cjd/data_wxq/fallingthings'):
        super().__init__(aug_params, reader=frame_utils.readDispFallingThings)
        assert os.path.exists(root)

        image1_list = sorted(glob(root + '/*/*/*left.jpg'))
        image2_list = sorted(glob(root + '/*/*/*right.jpg'))
        disp_list = sorted(glob(root + '/*/*/*left.depth.png'))

        image1_list += sorted(glob(root + '/*/*/*/*left.jpg'))
        image2_list += sorted(glob(root + '/*/*/*/*right.jpg'))
        disp_list += sorted(glob(root + '/*/*/*/*left.depth.png'))

        assert len(image1_list) == len(image2_list) == len(disp_list)

        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]

class TartanAir(StereoDataset):
    def __init__(self, aug_params=None, root='datasets', keywords=[]):
        super().__init__(aug_params, reader=frame_utils.readDispTartanAir)
        assert os.path.exists(root)

        with open(os.path.join(root, 'tartanair_filenames.txt'), 'r') as f:
            filenames = sorted(list(filter(lambda s: 'seasonsforest_winter/Easy' not in s, f.read().splitlines())))
            for kw in keywords:
                filenames = sorted(list(filter(lambda s: kw in s.lower(), filenames)))

        image1_list = [osp.join(root, e) for e in filenames]
        image2_list = [osp.join(root, e.replace('_left', '_right')) for e in filenames]
        disp_list = [osp.join(root, e.replace('image_left', 'depth_left').replace('left.png', 'left_depth.npy')) for e in filenames]

        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]

class KITTI(StereoDataset):
    def __init__(self, aug_params=None, root='/data2/cjd/StereoDatasets/kitti/2015', image_set='training'):
        super(KITTI, self).__init__(aug_params, sparse=True, reader=frame_utils.readDispKITTI)
        assert os.path.exists(root)

        root_12 = '/data2/cjd/StereoDatasets/kitti/2012/'
        image1_list = sorted(glob(os.path.join(root_12, image_set, 'colored_0/*_10.png'))) # 添加2012右视图
        image2_list = sorted(glob(os.path.join(root_12, image_set, 'colored_1/*_10.png'))) # 添加2012右视图
        disp_list = sorted(glob(os.path.join(root_12, 'training', 'disp_occ/*_10.png'))) if image_set == 'training' else [osp.join(root, 'training/disp_occ/000085_10.png')]*len(image1_list) #训练模式全部训练样本，其它模式固定样本

        root_15 = '/data2/cjd/StereoDatasets/kitti/2015/'
        image1_list += sorted(glob(os.path.join(root_15, image_set, 'image_2/*_10.png'))) # 添加2015左视图
        image2_list += sorted(glob(os.path.join(root_15, image_set, 'image_3/*_10.png'))) # 添加2015右视图
        disp_list += sorted(glob(os.path.join(root_15, 'training', 'disp_occ_0/*_10.png'))) if image_set == 'training' else [osp.join(root, 'training/disp_occ_0/000085_10.png')]*len(image1_list)

        for idx, (img1, img2, disp) in enumerate(zip(image1_list, image2_list, disp_list)):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]

class VKITTI2(StereoDataset):
    def __init__(self, aug_params=None, root='/data/cjd/stereo_dataset/vkitti2/'):
        super(VKITTI2, self).__init__(aug_params, sparse=True, reader=frame_utils.readDispVKITTI2)
        assert os.path.exists(root)

        image1_list = sorted(glob(os.path.join(root, 'Scene*/*/frames/rgb/Camera_0/rgb*.jpg')))
        image2_list = sorted(glob(os.path.join(root, 'Scene*/*/frames/rgb/Camera_1/rgb*.jpg')))
        disp_list = sorted(glob(os.path.join(root, 'Scene*/*/frames/depth/Camera_0/depth*.png')))

        assert len(image1_list) == len(image2_list) == len(disp_list)

        for idx, (img1, img2, disp) in enumerate(zip(image1_list, image2_list, disp_list)):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]

class Middlebury(StereoDataset):
    def __init__(self, aug_params=None, root='/data2/cjd/StereoDatasets/middlebury', split='2014', resolution='F'):
        super(Middlebury, self).__init__(aug_params, sparse=True, reader=frame_utils.readDispMiddlebury)
        assert os.path.exists(root)
        assert split in ["2005", "2006", "2014", "2021", "MiddEval3"]
        if split == "2005":
            scenes = list((Path(root) / "2005").glob("*"))
            for scene in scenes:
                self.image_list += [[str(scene / "view1.png"), str(scene / "view5.png")]]
                self.disparity_list += [str(scene / "disp1.png")]    
                for illum in ["1", "2", "3"]:
                    for exp in ["0", "1", "2"]:       
                        self.image_list += [[str(scene / f"Illum{illum}/Exp{exp}/view1.png"), str(scene / f"Illum{illum}/Exp{exp}/view5.png")]]
                        self.disparity_list += [str(scene / "disp1.png")]        
        elif split == "2006":
            scenes = list((Path(root) / "2006").glob("*"))
            for scene in scenes:
                self.image_list += [[str(scene / "view1.png"), str(scene / "view5.png")]]
                self.disparity_list += [str(scene / "disp1.png")]    
                for illum in ["1", "2", "3"]:
                    for exp in ["0", "1", "2"]:       
                        self.image_list += [[str(scene / f"Illum{illum}/Exp{exp}/view1.png"), str(scene / f"Illum{illum}/Exp{exp}/view5.png")]]
                        self.disparity_list += [str(scene / "disp1.png")]
        elif split == "2014":
            scenes = list((Path(root) / "2014").glob("*"))
            for scene in scenes:
                for s in ["E", "L", ""]:
                    self.image_list += [ [str(scene / "im0.png"), str(scene / f"im1{s}.png")] ]
                    self.disparity_list += [ str(scene / "disp0.pfm") ]
        elif split == "2021":
            scenes = list((Path(root) / "2021/data").glob("*"))
            for scene in scenes:
                self.image_list += [[str(scene / "im0.png"), str(scene / "im1.png")]]
                self.disparity_list += [str(scene / "disp0.pfm")]
                for s in ["0", "1", "2", "3"]:
                    if os.path.exists(str(scene / f"ambient/L0/im0e{s}.png")):
                        self.image_list += [[str(scene / f"ambient/L0/im0e{s}.png"), str(scene / f"ambient/L0/im1e{s}.png")]]
                        self.disparity_list += [str(scene / "disp0.pfm")]
        else:
            image1_list = sorted(glob(os.path.join(root, "MiddEval3", f'training{resolution}', '*/im0.png')))
            image2_list = sorted(glob(os.path.join(root, "MiddEval3", f'training{resolution}', '*/im1.png')))
            disp_list = sorted(glob(os.path.join(root, "MiddEval3", f'training{resolution}', '*/disp0GT.pfm')))
            assert len(image1_list) == len(image2_list) == len(disp_list) > 0, [image1_list, split]
            for img1, img2, disp in zip(image1_list, image2_list, disp_list):
                self.image_list += [ [img1, img2] ]
                self.disparity_list += [ disp ]

class CREStereoDataset(StereoDataset):
    def __init__(self, aug_params=None, root='/data2/cjd/StereoDatasets/crestereo'):
        super(CREStereoDataset, self).__init__(aug_params, reader=frame_utils.readDispCREStereo)
        assert os.path.exists(root)

        image1_list = sorted(glob(os.path.join(root, '*/*_left.jpg')))
        image2_list = sorted(glob(os.path.join(root, '*/*_right.jpg')))
        disp_list = sorted(glob(os.path.join(root, '*/*_left.disp.png')))

        assert len(image1_list) == len(image2_list) == len(disp_list)

        for idx, (img1, img2, disp) in enumerate(zip(image1_list, image2_list, disp_list)):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]

class InStereo2K(StereoDataset):
    def __init__(self, aug_params=None, root='/data2/cjd/data_wxq/instereo2k'):
        super(InStereo2K, self).__init__(aug_params, sparse=True, reader=frame_utils.readDispInStereo2K)
        assert os.path.exists(root)

        image1_list = sorted(glob(root + '/train/*/*/left.png') + glob(root + '/test/*/left.png'))
        image2_list = sorted(glob(root + '/train/*/*/right.png') + glob(root + '/test/*/right.png'))
        disp_list = sorted(glob(root + '/train/*/*/left_disp.png') + glob(root + '/test/*/left_disp.png'))

        assert len(image1_list) == len(image2_list) == len(disp_list)

        for idx, (img1, img2, disp) in enumerate(zip(image1_list, image2_list, disp_list)):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]

class CARLA(StereoDataset):
    def __init__(self, aug_params=None, root='/data2/cjd/StereoDatasets/carla-highres'):
        super(CARLA, self).__init__(aug_params)
        assert os.path.exists(root)

        image1_list = sorted(glob(root + '/trainingF/*/im0.png'))
        image2_list = sorted(glob(root + '/trainingF/*/im1.png'))
        disp_list = sorted(glob(root + '/trainingF/*/disp0GT.pfm'))

        assert len(image1_list) == len(image2_list) == len(disp_list)

        for idx, (img1, img2, disp) in enumerate(zip(image1_list, image2_list, disp_list)):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]

class DrivingStereo(StereoDataset):
    def __init__(self, aug_params=None, root='/data2/cjd/StereoDatasets/drivingstereo/', image_set='rainy'):
        reader = frame_utils.readDispDrivingStereo_half
        super().__init__(aug_params, sparse=True, reader=reader)
        assert os.path.exists(root)
        image1_list = sorted(glob(os.path.join(root, image_set, 'left-image-half-size/*.jpg')))
        image2_list = sorted(glob(os.path.join(root, image_set, 'right-image-half-size/*.jpg')))
        disp_list = sorted(glob(os.path.join(root, image_set, 'disparity-map-half-size/*.png')))

        for idx, (img1, img2, disp) in enumerate(zip(image1_list, image2_list, disp_list)):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]

  
def fetch_dataloader(args):
    """ Create the data loader for the corresponding trainign set """
    """创建训练集数据加载器（DataLoader）

        参数:
            args (argparse.Namespace): 包含所有训练参数的配置对象，主要使用：
                - image_size: 输入图像尺寸 [H, W]
                - spatial_scale: 空间缩放范围 [min, max]
                - noyjitter: 是否禁用亮度对比度抖动
                - saturation_range: 饱和度增强范围
                - img_gamma: 伽马校正系数
                - do_flip: 是否启用水平翻转
                - train_datasets: 要使用的数据集名称列表

        返回:
            torch.utils.data.Dataset: 组合后的训练数据集
    """
    # print('args.img_gamma', args.img_gamma)
    # ================== 数据增强参数配置 ==================
    # 'crop_size': list(args.image_size),  # 裁剪尺寸 [H, W]
    # 'min_scale': args.spatial_scale[0],   # 最小缩放比例（如0.8） 实际yaml中都是-0.2
    # 'max_scale': args.spatial_scale[1],   # 最大缩放比例（如1.2）
    # 'do_flip': False,                     # 默认不水平翻转
    # 'yjitter': not args.noyjitter         # 是否启用亮度对比度抖动
    aug_params = {'crop_size': list(args.image_size), 'min_scale': args.spatial_scale[0], 'max_scale': args.spatial_scale[1], 'do_flip': False, 'yjitter': not args.noyjitter}

    # 可选增强参数（根据args动态添加）
    if hasattr(args, "saturation_range") and args.saturation_range is not None:
        aug_params["saturation_range"] = list(args.saturation_range) # 饱和度增强范围
    if hasattr(args, "img_gamma") and args.img_gamma is not None:
        aug_params["gamma"] = args.img_gamma # 伽马校正系数
    if hasattr(args, "do_flip") and args.do_flip is not None:
        aug_params["do_flip"] = args.do_flip # 覆盖翻转设置

    # ================== 数据集组合逻辑 ==================
    train_dataset = None # 初始化组合数据集
    print('train_datasets', args.train_datasets)
    # 遍历所有需要组合的数据集名称
    for dataset_name in args.train_datasets:
        # 场景流数据集（合成数据）
        if dataset_name == 'sceneflow':
            new_dataset = SceneFlowDatasets(aug_params, dstype='frames_finalpass')
            logging.info(f"Adding {len(new_dataset)} samples from SceneFlow")
        # KITTI 真实场景数据集
        elif dataset_name == 'kitti':
            new_dataset = KITTI(aug_params)
            logging.info(f"Adding {len(new_dataset)} samples from KITTI")
        # Sintel 光流数据集（立体匹配版本）
        elif dataset_name == 'sintel_stereo':
            new_dataset = SintelStereo(aug_params)*140 # 应用数据增强，并生成140倍数据量
            logging.info(f"Adding {len(new_dataset)} samples from Sintel Stereo")
        # FallingThings 合成物体数据集
        elif dataset_name == 'falling_things':
            new_dataset = FallingThings(aug_params)*5 # 增强5倍
            logging.info(f"Adding {len(new_dataset)} samples from FallingThings")
        # TartanAir 无人机模拟数据集（支持关键词筛选）
        elif dataset_name.startswith('tartan_air'):
            new_dataset = TartanAir(aug_params, keywords=dataset_name.split('_')[2:])
            logging.info(f"Adding {len(new_dataset)} samples from Tartain Air")
        # ETH3D 微调组合数据集
        elif dataset_name == 'eth3d_finetune':
            # 包含三个真实场景数据源的组合：
            crestereo = CREStereoDataset(aug_params) # CREStereo数据集
            logging.info(f"Adding {len(crestereo)} samples from CREStereo Dataset")            
            eth3d = ETH3D(aug_params) # ETH3D高精度数据
            logging.info(f"Adding {len(eth3d)} samples from ETH3D")
            instereo2k = InStereo2K(aug_params) # InStereo2K工业级数据
            logging.info(f"Adding {len(instereo2k)} samples from InStereo2K")
            new_dataset = eth3d * 1000 + instereo2k * 10 + crestereo # 组合比例：ETH3D*1000 + InStereo2K*10 + CREStereo
            logging.info(f"Adding {len(new_dataset)} samples from ETH3D Mixture Dataset")
        # Middlebury 完整训练组合数据集
        elif dataset_name == 'middlebury_train':
            # 包含6种数据源的超大规模组合：
            tartanair = TartanAir(aug_params) # 无人机模拟数据
            logging.info(f"Adding {len(tartanair)} samples from Tartain Air")
            sceneflow = SceneFlowDatasets(aug_params, dstype='frames_finalpass')
            logging.info(f"Adding {len(sceneflow)} samples from SceneFlow")
            fallingthings = FallingThings(aug_params) # 合成物体
            logging.info(f"Adding {len(fallingthings)} samples from FallingThings")
            carla = CARLA(aug_params) # 自动驾驶模拟数据
            logging.info(f"Adding {len(carla)} samples from CARLA")
            crestereo = CREStereoDataset(aug_params) # 研究数据集
            logging.info(f"Adding {len(crestereo)} samples from CREStereo Dataset")             
            instereo2k = InStereo2K(aug_params)
            logging.info(f"Adding {len(instereo2k)} samples from InStereo2K")
            # Middlebury 各个年份的子数据集（2005-2021）
            mb2005 = Middlebury(aug_params, split='2005')
            logging.info(f"Adding {len(mb2005)} samples from Middlebury 2005")
            mb2006 = Middlebury(aug_params, split='2006')
            logging.info(f"Adding {len(mb2006)} samples from Middlebury 2006")
            mb2014 = Middlebury(aug_params, split='2014')
            logging.info(f"Adding {len(mb2014)} samples from Middlebury 2014")
            mb2021 = Middlebury(aug_params, split='2021')
            logging.info(f"Adding {len(mb2021)} samples from Middlebury 2021")
            mbeval3 = Middlebury(aug_params, split='MiddEval3', resolution='H')
            logging.info(f"Adding {len(mbeval3)} samples from Middlebury Eval3")
            # 组合所有数据集
            new_dataset = tartanair + sceneflow + fallingthings + instereo2k * 50 + carla * 50 + crestereo + mb2005 * 200 + mb2006 * 200 + mb2014 * 200 + mb2021 * 200 + mbeval3 * 200
            logging.info(f"Adding {len(new_dataset)} samples from Middlebury Mixture Dataset")
        # Middlebury 微调专用组合
        elif dataset_name == 'middlebury_finetune':
            # 面向高精度微调的混合数据集：
            crestereo = CREStereoDataset(aug_params)
            logging.info(f"Adding {len(crestereo)} samples from CREStereo Dataset")                 
            instereo2k = InStereo2K(aug_params)
            logging.info(f"Adding {len(instereo2k)} samples from InStereo2K")
            carla = CARLA(aug_params)
            logging.info(f"Adding {len(carla)} samples from CARLA")
            mb2005 = Middlebury(aug_params, split='2005')
            logging.info(f"Adding {len(mb2005)} samples from Middlebury 2005")
            mb2006 = Middlebury(aug_params, split='2006')
            logging.info(f"Adding {len(mb2006)} samples from Middlebury 2006")
            mb2014 = Middlebury(aug_params, split='2014')
            logging.info(f"Adding {len(mb2014)} samples from Middlebury 2014")
            mb2021 = Middlebury(aug_params, split='2021')
            logging.info(f"Adding {len(mb2021)} samples from Middlebury 2021")
            mbeval3 = Middlebury(aug_params, split='MiddEval3', resolution='H')
            logging.info(f"Adding {len(mbeval3)} samples from Middlebury Eval3")
            mbeval3_f = Middlebury(aug_params, split='MiddEval3', resolution='F')
            logging.info(f"Adding {len(mbeval3)} samples from Middlebury Eval3")
            fallingthings = FallingThings(aug_params)
            logging.info(f"Adding {len(fallingthings)} samples from FallingThings")
            new_dataset = crestereo + instereo2k * 50 + carla * 50 + mb2005 * 200 + mb2006 * 200 + mb2014 * 200 + mb2021 * 200 + mbeval3 * 200 + mbeval3_f * 400 + fallingthings * 5
            logging.info(f"Adding {len(new_dataset)} samples from Middlebury Mixture Dataset")

        train_dataset = new_dataset if train_dataset is None else train_dataset + new_dataset # 合并到总数据集（通过重载的+运算符）

    return train_dataset


if __name__ == '__main__':
    import matplotlib
    import matplotlib.pyplot as plt
    import cv2


    def gray_2_colormap_np(img, cmap='rainbow', max=None):
        img = img.cpu().detach().numpy().squeeze()
        assert img.ndim == 2
        img[img < 0] = 0
        mask_invalid = img < 1e-10
        if max == None:
            img = img / (img.max() + 1e-8)
        else:
            img = img / (max + 1e-8)

        norm = matplotlib.colors.Normalize(vmin=0, vmax=1.1)
        cmap_m = matplotlib.cm.get_cmap(cmap)
        map = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap_m)
        colormap = (map.to_rgba(img)[:, :, :3] * 255).astype(np.uint8)
        colormap[mask_invalid] = 0

        return colormap


    def viz_disp(disp, scale=1, COLORMAP=cv2.COLORMAP_JET):
        disp_np = (torch.abs(disp[0].squeeze())).data.cpu().numpy()
        disp_np = (disp_np * scale).astype(np.uint8)
        disp_color = cv2.applyColorMap(disp_np, COLORMAP)
        return disp_color
    plot_dir = './temp/plots'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    dataset = VKITTI2()
    for i in range(5):
        _, *data_blob = dataset[i]
        image1, image2, disp_gt, valid = [x[None] for x in data_blob]
        image1_np = image1[0].squeeze().cpu().numpy()
        image1_np = (image1_np - image1_np.min()) / (image1_np.max() - image1_np.min()) * 255.0
        image1_np = image1_np.astype(np.uint8)

        disp_color = viz_disp(disp_gt, scale=5)
        cv2.imwrite(os.path.join(plot_dir, f'{i}_disp_gt.png'), disp_color)

        disp_gt_np = gray_2_colormap_np(disp_gt[0].squeeze())
        cv2.imwrite(os.path.join(plot_dir, f'{i}_disp_gt1.png'), disp_gt_np[:, :, ::-1])

        image1 = image1[0].permute(1, 2, 0).cpu().numpy()[:, :, ::-1]
        cv2.imwrite(os.path.join(plot_dir, f'{i}_img1.png'), image1)





