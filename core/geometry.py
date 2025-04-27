import torch
import torch.nn.functional as F
from core.utils.utils import bilinear_sampler


class Combined_Geo_Encoding_Volume:
    def __init__(self, init_fmap1, init_fmap2, geo_volume, num_levels=2, radius=4):
        self.num_levels = num_levels # 金字塔层级数 monster中使用了默认的2
        self.radius = radius # 邻域半径
        self.geo_volume_pyramid = [] # 几何编码金字塔列表
        self.init_corr_pyramid = [] # 初始相关性金字塔列表

        # all pairs correlation
        # Step 1: 计算初始左右图特征的相关性体积
        # corr()方法返回形状(B,H,W,1,W2), 其中W2为右图宽度方向的匹配范围
        # 每个位置(i,j)存储左图(i,j)与右图(i,:)所有位置的点积相似度
        init_corr = Combined_Geo_Encoding_Volume.corr(init_fmap1, init_fmap2)

        # Step 2: 预处理几何编码体积 (geo_volume)
        # 原始输入: geo_volume shape=(b, c, d, h, w)
        # 调整维度: (b, c, d, h, w) -> (b, h, w, c, d) -> (b*h*w, c, 1, d)
        # 目的: 将空间维度(h,w)合并到batch维度，方便后续采样
        b, h, w, _, w2 = init_corr.shape
        b, c, d, h, w = geo_volume.shape
        geo_volume = geo_volume.permute(0, 3, 4, 1, 2).reshape(b*h*w, c, 1, d)

        # Step 3: 初始化金字塔底层
        # 处理初始相关性: (b, h, w, 1, w2) -> (b*h*w, 1, 1, w2)
        init_corr = init_corr.reshape(b*h*w, 1, 1, w2) # 输出：(B*H*W,1,1,W2)
        self.geo_volume_pyramid.append(geo_volume) # 添加几何编码底层
        self.init_corr_pyramid.append(init_corr) # 添加相关性底层

        # Step 4: 构建下采样金字塔（多尺度特征）
        # 几何编码金字塔：沿深度维度D下采样（缩小视差搜索范围）
        for i in range(self.num_levels-1):
            # 几何编码下采样: 深度方向(d)按因子2缩小 (kernel=[1,2])
            geo_volume = F.avg_pool2d(geo_volume, [1,2], stride=[1,2]) # 平均池化 D维度减半
            self.geo_volume_pyramid.append(geo_volume)

        for i in range(self.num_levels-1):
            # 相关性下采样: 宽度方向(w2)按因子2缩小 (kernel=[1,2])
            init_corr = F.avg_pool2d(init_corr, [1,2], stride=[1,2]) # 平均池化 W2维度减半
            self.init_corr_pyramid.append(init_corr)




    def __call__(self, disp, coords):
        """ 生成多层级融合特征
                Args:
                    disp (Tensor): 当前迭代的视差图, shape=(B,1,H,W)
                    coords (Tensor): 归一化像素坐标, shape=(B,H,W,1)    coords 是一个 表示像素水平位置（列坐标）的张量，用于在立体匹配任务中结合几何编码和视差信息，调整特征的对齐和采样位置
                Returns:
                    out (Tensor): 融合后的特征图, shape=(B,C,H,W)

        coords形状：(B, H, W, 1) （coords存的是像素点的行序列号，显式存储）
        其中：
            B：批次大小（batch size）
            H：图像高度（height）
            W：图像宽度（width）
            1：每个像素的水平坐标（列索引），范围 [0, W-1]  在深度学习中W只是隐含信息和尺寸参数，无法直接用于位置调整
        """
        r = self.radius
        b, _, h, w = disp.shape
        out_pyramid = []

        # 遍历每个金字塔层级（如2层）
        for i in range(self.num_levels):
            # ----------------- 几何编码体积采样 ------------------
            geo_volume = self.geo_volume_pyramid[i]

            # 生成水平偏移量：[-r, -r+1, ..., r] 等分数，范围是-r到r，共2r+1个点
            dx = torch.linspace(-r, r, 2*r+1) # 输出形状是[2*r+1]
            dx = dx.view(1, 1, 2*r+1, 1).to(disp.device) # 输出形状(1,1,2r+1,1),
            # 计算采样中心：基于当前视差disp，并按层级缩放（2^i）
            # disp形状(B,1,H,W) -> 展平为(B*H*W,1,1,1)后除以2^i （高层级特征图分辨率低（如层级i的分辨率是原图的1/(2^i)），视差值需按比例缩小。）
            # 视差调整：将原图视差disp转换为当前层级的坐标空间（除以2^i）。
            # 邻域扩展：添加dx偏移量，生成以调整后视差为中心的候选位置。
            # 几何意义：在层级i中，候选视差范围为 [disp_scaled - r, disp_scaled + r] 共 2*r+1 个采样点
            x0 = dx + disp.reshape(b*h*w, 1, 1, 1) / 2**i # 会触发广播机制，因为形状不同 输出形状(B*H*W,1,2r+1,1)
            y0 = torch.zeros_like(x0) # # y方向无偏移（极线约束） 和x0形状一样的新张量，张量里面全是0

            # 构造采样坐标：组合x和y偏移量（y始终为0）
            disp_lvl = torch.cat([x0,y0], dim=-1) # 输出形状(B*H*W,1,2r+1,2)

            # 双线性采样：从几何编码体积中提取局部邻域特征
            # 输入geo_volume形状(B*H*W,C,1,D) -> 输出(B*H*W,C,2r+1,D)
            geo_volume = bilinear_sampler(geo_volume, disp_lvl)
            geo_volume = geo_volume.view(b, h, w, -1) # 展平为(B,H,W,C*(2r+1)*D)

            # ----------------- 初始相关性采样 ------------------
            init_corr = self.init_corr_pyramid[i] # 当前层级的初始相关性

            # 计算采样位置：基于坐标和视差调整（消除视差偏移后加邻域偏移）
            init_x0 = coords.reshape(b*h*w, 1, 1, 1)/2**i - disp.reshape(b*h*w, 1, 1, 1) / 2**i + dx
            init_coords_lvl = torch.cat([init_x0,y0], dim=-1)

            # 双线性采样：从相关性体积中提取局部邻域特征
            # 输入init_corr形状(B*H*W,1,1,W2) -> 输出(B*H*W,1,2r+1,W2)
            init_corr = bilinear_sampler(init_corr, init_coords_lvl)
            init_corr = init_corr.view(b, h, w, -1) # 展平为(B,H,W,1*(2r+1)*W2)

            # 收集当前层级特征
            out_pyramid.append(geo_volume) # 几何编码特征
            out_pyramid.append(init_corr) # 初始相关性特征

        # 沿通道维度拼接所有层级特征
        # 迭代优化的需求**
        # GRU/ConvGRU 的输入：后续的迭代优化模块（如 update_block）需要同时感知几何和表观信息，动态修正视差。
        # 特征融合公式：
        # GRU输入=几何特征⊕表观特征(⊕表示拼接)
        out = torch.cat(out_pyramid, dim=-1) # 输出形状(B,H,W,总通道数)
        return out.permute(0, 3, 1, 2).contiguous().float() # 调整维度：通道维度前置 -> (B,C,H,W)

    
    @staticmethod
    def corr(fmap1, fmap2):
        """
            计算左右图特征间的稠密相关性体积（逐位置点积）

            Args:
                fmap1 (Tensor): 左图特征张量，shape=(B, D, H, W1)
                                B: batch大小, D: 特征维度, H: 高度, W1: 左图宽度
                fmap2 (Tensor): 右图特征张量，shape=(B, D, H, W2)
                                W2: 右图宽度（通常为原图宽度或经过位移后的尺寸）

            Returns:
                corr (Tensor): 3D相关性体积，shape=(B, H, W1, 1, W2)
                               每个空间位置(W1)与右图W2宽度的所有位置计算相似度
        """
        # 获取输入张量的维度信息
        B, D, H, W1 = fmap1.shape # 解构左图维度
        _, _, _, W2 = fmap2.shape # 解构右图维度（保持B,D,H对齐）

        # 显式声明视图（冗余操作，实际可省略，可能用于维度明确化） 大概是为了保证左右图的某些维度一致性
        fmap1 = fmap1.view(B, D, H, W1) # 左图特征: (B,D,H,W1)
        fmap2 = fmap2.view(B, D, H, W2) # 右图特征: (B,D,H,W2)

        # 爱因斯坦求和计算稠密相关性
        # 输入1: 'aijk' -> 左图 (B=batch, D=i, H=j, W1=k)
        # 输入2: 'aijh' -> 右图 (B=batch, D=i, H=j, W2=h)
        # 输出: 'ajkh' -> 相关性体积 (B=a, H=j, W1=k, W2=h)
        # 物理意义: 对每个batch(a)、每行像素(j)，计算左图位置k与右图位置h的D维特征点积
        corr = torch.einsum('aijk,aijh->ajkh', fmap1, fmap2)

        # 维度重塑添加通道维度（兼容后续处理）
        # 从 (B, H, W1, W2) -> (B, H, W1, 1, W2)
        # 目的：可能用于后续3D卷积操作，1作为伪通道维度
        corr = corr.reshape(B, H, W1, 1, W2).contiguous()
        return corr

    # 对于某个具体位置(b=0, h=100, w1=50, 1, w2=80)： 那个 1 是伪通道维度（仅为兼容3D卷积操作保留）
    #
    # 物理意义：第0个批次，第100行，左图第50列像素与右图第80列像素的匹配得分。
    # 计算方式：左图特征向量
    # fmap1[0, :, 100, 50]
    # 与右图特征向量
    # fmap2[0, :, 100, 80]
    # 的点积（余弦相似度）。
