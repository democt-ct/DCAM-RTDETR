'''by lyuwenyu
'''
import copy
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from onnx.reference.ops.op_sigmoid import Sigmoid, sigmoid
from sympy.codegen import Print
from torch.ao.nn.quantized import BatchNorm2d
from torch.cuda import device
from torch.nn import MaxPool2d
from torch.nn.functional import max_pool2d, conv2d
from torchgen.api.ufunc import kernel_name
from torch.cuda.amp import autocast  # 混合精度训练
from torch.utils.checkpoint import checkpoint  # 梯度检查点
from timm.layers import DropPath  # DropPath 正则化
from .utils import get_activation
from src.core import register
from timm.layers import trunc_normal_
import numpy as np  # 导入 numpy 库
__all__ = ['HybridEncoder']

import random
import torch
from torch import nn
import torch.nn.functional as F
 #模块结合了卷积层、批归一化和激活函数，实现标准的卷积操作并对输出进行批归一化（BatchNorm）处理，最后通过激活函数（默认为 ReLU 或指定的激活函数）进行非线性变换。
class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size,
            stride,
            padding=(kernel_size-1)//2 if padding is None else padding,
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        # self.norm = nn.GroupNorm(num_groups=32,num_channels=ch_in,eps=1e-5)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))
#类似VGG结构，包含两个分支的卷积：一个是 3x3 卷积，另一个是 1x1 卷积，并且通过加法融合两者的输出。作用：该块的主要作用是提升网络的表现能力，通过分支结构来增强特征表达。
class RepVggBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu'):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)

        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        # self.__delattr__('conv1')
        # self.__delattr__('conv2')

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)

        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: ConvNormLayer):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std
#结合了 CSPNet（Cross-Stage Partial Network）和 RepVggBlock 的模块，目的是对输入特征进行跨通道的特征交互。作用：：通过两个 1x1 卷积分支，提取输入特征的不同部分，接着使用多个 RepVggBlock 进行融合处理。
class CSPRepLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=None,
                 act="silu"):
        super(CSPRepLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)  #conv1 提取输入特征的一部分，经过多个 RepVggBlock。
        self.conv2 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)  #conv2 提取原始输入特征的另一部分。
        self.bottlenecks = nn.Sequential(*[
            RepVggBlock(hidden_channels, hidden_channels, act=act) for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)
# transformer：这些模块实现了 Transformer 编码器层（自注意力机制）和 Transformer 编码器。功能：在图像特征中实现尺度内的交互，利用自注意力机制计算各个位置之间的相关性，从而捕获全局依赖。
# 自注意力机制（Transformer）中的尺度间融合
class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 normalize_before=False):
        super().__init__()
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation)

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)  #尺度内交互的体现，通过多头注意力机制学习到特征图各个位置间的相关性，从而实现全局特征交互。

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src
class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)

        if self.norm is not None:
            output = self.norm(output)

        return output


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                   groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.activation(x)
class LocalAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, local_kernel_sizes=[3,5,7]):
        super(LocalAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Multi-scale local convolutions
        self.local_convs = nn.ModuleList([
            nn.Sequential(
                DepthwiseSeparableConv(embed_dim, embed_dim, kernel_size=k, padding=k // 2),
                nn.Conv2d(embed_dim, embed_dim * 3, kernel_size=1, groups=num_heads)
            ) for k in local_kernel_sizes
        ])

        # Dynamic weight calculation
        self.dynamic_weight_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(embed_dim, len(local_kernel_sizes), kernel_size=1),
            nn.Softmax(dim=1)
        )

        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(1, embed_dim, 1, 1))

    def forward(self, x):
        batch_size, embed_dim, height, width = x.size()
        x_with_position = x + self.positional_encoding

        # Split heads and process multi-scale local attention
        local_outs = []
        for local_conv in self.local_convs:
            qkv = local_conv(x_with_position)
            q, k, v = qkv.chunk(3, dim=1)

            # Reshape for multi-head attention
            q = q.view(batch_size, self.num_heads, self.head_dim, -1)
            k = k.view(batch_size, self.num_heads, self.head_dim, -1)
            v = v.view(batch_size, self.num_heads, self.head_dim, -1)

            # Scaled dot-product attention
            attn = torch.einsum('bhqd, bhkd -> bhqk', q, k) * self.scale
            attn = F.softmax(attn, dim=-1)
            out = torch.einsum('bhqk, bhvd -> bhqd', attn, v)
            out = out.view(batch_size, embed_dim, height, width)
            local_outs.append(out)

        # Dynamic weight fusion
        dynamic_weights = self.dynamic_weight_fc(x)
        weighted_out = sum(w * out for w, out in zip(dynamic_weights.chunk(len(local_outs), dim=1), local_outs))

        # Residual connection
        return x + weighted_out
class LightweightPolicyNetwork(nn.Module):
    def __init__(self, in_channels):
        super(LightweightPolicyNetwork, self).__init__()
        self.conv = nn.Sequential(
            DepthwiseSeparableConv(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.Conv2d(in_channels // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.conv(x)
class LocalGlobalAttention(nn.Module):
    def __init__(self, in_channels, out_channels, embed_dim=256, num_heads=8, global_kernel_sizes=[1,9,11]):
        super(LocalGlobalAttention, self).__init__()

        # Local attention
        self.local_attention = LocalAttention(embed_dim, num_heads)

        # Global attention: Multi-scale depthwise separable convolutions
        self.global_convs = nn.ModuleList([
            DepthwiseSeparableConv(in_channels, out_channels, kernel_size=k, padding=k // 2)
            for k in global_kernel_sizes
        ])

        # Dynamic fusion weight
        self.policy_network = LightweightPolicyNetwork(in_channels)

        # Activation
        self.activation = nn.GELU()

    def forward(self, x):
        residual = x

        # Local attention
        local_out = self.local_attention(x)

        # Global attention
        global_outs = [conv(x) for conv in self.global_convs]
        global_out = sum(global_outs) / len(global_outs)  # Average multi-scale outputs

        # Dynamic fusion
        fusion_weight = self.policy_network(x)
        fused_out = (1 - fusion_weight) * local_out + fusion_weight * global_out

        # Residual connection
        return residual + self.activation(fused_out)

class AMSF(nn.Module):
    def __init__(self, in_channels, out_channels, scale_num=3):
        super(AMSF, self).__init__()
        self.scale_num = scale_num
        self.out_channels = out_channels

        # 定义卷积层
        self.conv1 = ConvNormLayer(in_channels,out_channels,3,1,1,act='GELU')
        self.conv2 = ConvNormLayer(in_channels, out_channels, 3, 1, 1, act='GELU')
        self.conv3 = ConvNormLayer(in_channels, out_channels, 3, 1, 1, act='GELU')

        # 定义自适应权重计算层
        self.weight_conv1 = nn.Conv2d(out_channels*3, 1, kernel_size=3, stride=1, padding=1)
        self.weight_conv2 = nn.Conv2d( out_channels*3, 1, kernel_size=3, stride=1, padding=1)
        self.weight_conv3 = nn.Conv2d(out_channels*3, 1, kernel_size=3, stride=1, padding=1)

        self.conv_last = nn.Conv2d(3 * out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.Sigmoid = nn.Sigmoid()

        # 残差连接
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        residual = self.residual_conv(x)
        x3 = self.conv1(x)
        x5 = self.conv2(x3)
        x7 = self.conv3(x5)

        x3_mp, _ =  torch.max(x3, dim=1, keepdim=True)
        x5_mp, _ = torch.max(x5, dim=1, keepdim=True)
        x7_mp, _ = torch.max(x7, dim=1, keepdim=True)
        x_cat = torch.cat([x3,x5,x7],dim=1)

        # w3 = torch.sigmoid(self.weight_conv1(x_cat * torch.sigmoid(x3_mp)))
        # w5 = torch.sigmoid(self.weight_conv2(x_cat * torch.sigmoid(x5_mp)))
        # w7 = torch.sigmoid(self.weight_conv3(x_cat * torch.sigmoid(x7_mp)))

        w3 = torch.softmax(self.weight_conv1(x_cat * torch.sigmoid(x3_mp)), dim=1)
        w5 = torch.softmax(self.weight_conv2(x_cat * torch.sigmoid(x5_mp)), dim=1)
        w7 = torch.softmax(self.weight_conv3(x_cat * torch.sigmoid(x7_mp)), dim=1)

        # w3 = torch.sigmoid(self.weight_conv1(x_cat @ torch.sigmoid(x3_mp)))
        # w5 = torch.sigmoid(self.weight_conv2(x_cat @ torch.sigmoid(x5_mp)))
        # w7 = torch.sigmoid(self.weight_conv3(x_cat @ torch.sigmoid(x7_mp)))

        y = self.conv_last(torch.cat([w3*x3,w5*x5,w7*x7],dim=1))

        # Step 7: 残差连接
        y += residual

        return y

# HybridEncoder 是这个结构的核心类，负责对多尺度特征进行处理和融合。
@register
class HybridEncoder(nn.Module):
    def __init__(self,
                 in_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 hidden_dim=256,
                 nhead=8,
                 dim_feedforward = 1024,
                 dropout=0.0,
                 enc_act='gelu',
                 use_encoder_idx=[2],
                 num_encoder_layers=1,
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1.0,
                 act='silu',
                 eval_spatial_size=None,
                 use_amsf=True,
                 use_gpa=True,
                ):
        super(HybridEncoder, self).__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size
        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        self.out_strides = feat_strides
        self.use_amsf = use_amsf
        self.use_gpa = use_gpa
        # 初始化 AMS-F 模块
        if self.use_amsf:
            self.amsf = AMSF(
                in_channels=hidden_dim,  # 输入通道数
                out_channels=hidden_dim,  # 输出通道数（可调整）
                scale_num=len(in_channels)  # 多尺度特征数量
            )
        if self.use_gpa:
            self.gpa =LocalGlobalAttention(hidden_dim, hidden_dim)
        #输入通道和特征图预处理，该部分通过 1x1 卷积将输入特征图的通道数从原始的通道数（如 512, 1024, 2048）投影到一个固定的隐藏维度256（hidden_dim）。
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden_dim)
                )
            )
  # encoder transformer编码器。功能：根据给定的 use_encoder_idx，在不同尺度的特征图上使用 Transformer 编码器进行处理，实现尺度内的特征交互。
        encoder_layer = TransformerEncoderLayer(
            hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=enc_act)
        self.encoder = nn.ModuleList([
    #range，根据长度值生成对应的整数序列。加了for，循坏控制重复执行trans
            TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers) for _ in range(len(use_encoder_idx))
        ])
  # top-down fpn
 #特征金字塔网络（FPN）中的尺度间融合，FPN：从高层到低层进行特征融合，通过上采样和跨尺度拼接增强特征。
        #侧向连接卷积层，用作较高层次特征映射到相同的通道数。
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        #2，1，在1的时候进行了循环
        for _ in range(len(in_channels) - 1, 0, -1):
            self.lateral_convs.append(ConvNormLayer(hidden_dim, hidden_dim, 1, 1, act=act))
            self.fpn_blocks.append(CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion))

 # bottom-up pan，PAN（Path Aggregation Networks）：从低层到高层传递信息，帮助模型更好地理解大物体的特征。
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        # 2，1
        for _ in range(len(in_channels) - 1):
            self.downsample_convs.append(
                ConvNormLayer(hidden_dim, hidden_dim, 3, 2, act=act)
            )
            self.pan_blocks.append(
                CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
            )
        self._reset_parameters()
    def _reset_parameters(self):
        if self.eval_spatial_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                #位置编码嵌入，功能：为了增强 Transformer 中的位置信息，模型构建了一个基于正弦余弦的 2D 位置嵌入（sinusoidal position embedding），将其添加到特征图中，传递给 Transformer 编码器。
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_spatial_size[1] // stride, self.eval_spatial_size[0] // stride,
                    self.hidden_dim, self.pe_temperature)
                setattr(self, f'pos_embed{idx}', pos_embed)
                # self.register_buffer(f'pos_embed{idx}', pos_embed)
    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):
        '''
        '''
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)
        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]
        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    def forward(self, feats):
        # print([feat.shape for feat in feats])   #打印输入输出形状
        assert len(feats) == len(self.in_channels)
#通过self.input_proj中的投影层对每个输入特征进行处理，并将结果存储在proj_feats列表中。通道数变为256，其余不变。
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]

        if self.use_amsf:
        #     # proj_feats[2]  = self.amsf(proj_feats[2])             #仅仅处理最后一层元素
            proj_feats = [self.amsf(feat) for feat in proj_feats]   #遍历所有元素

        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                # flatten [B, C, H, W] to [B, HxW, C]。将特征图从[B, C, H, W]维度扁平化为[B, HxW, C]。
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
    #如果模型处于训练模式或者没有预设的空间尺寸，则构建2D正弦余弦位置嵌入。
                if self.training or self.eval_spatial_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)
                else:
    # 从self中获取pos_embed加上索引enc_ind的属性移动到---
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None).to(src_flatten.device)
            #编码器处理特征
    # 将扁平化的特征和位置传递给编码器层。（b,hxw,c）
                memory = self.encoder[i](src_flatten, pos_embed=pos_embed)
    # 将编码器的输出重新调整为原始特征图的形状。
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()
    #             # print("hybridencoder-1",[x.is_contiguous() for x in proj_feats])

        #fpn融合
  #高层特征，等价于proj_feats[2]
        inner_outs = [proj_feats[-1]]
        # print([feat.shape for feat in proj_feats[-1]])
    # 从高层到低层进行特征融合。range（2，0，-1），2，1的索引
        for idx in range(len(self.in_channels) - 1, 0, -1):
    # 获取s5。inner_outs始终指向最高层特征图
            feat_high = inner_outs[0]
    # 获取s4。
            feat_low = proj_feats[idx - 1]
        # 通过横向卷积层处理s5。
            feat_high = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_high)
            inner_outs[0] = feat_high
    # 对高层特征进行上采样。
            upsample_feat = F.interpolate(feat_high, scale_factor=2., mode='nearest')
    # 在特征金字塔网络（FPN）块中融合上采样的高层特征和低层特征(不同来源的特征图)。跨尺度融合
            inner_out = self.fpn_blocks[len(self.in_channels)-1-idx](torch.concat([upsample_feat, feat_low], dim=1))
            inner_outs.insert(0, inner_out)  #每次处理完就要插入到最起那面，保持特征图顺序与原始网络结构一致。

        if self.use_gpa:
            inner_outs = [self.gpa(inner_out) for inner_out in inner_outs]
        outs = [inner_outs[0]]
        # print("1", inner_outs[-1].shape)高中低

        #跨尺度融合操作，在PAN中
    #0,1
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            # print("4",outs[-1].shape)
            feat_high = inner_outs[idx + 1]
            #对feat_low进行下采样
            downsample_feat = self.downsample_convs[idx](feat_low)
            #：将下采样后的特征图（downsample_feat）和较高分辨率的特征图（feat_high）拼接，然后传入 pan_blocks 进行处理。dim=1表示在通道上进行拼接,代码通过下采样特征和拼接不同分辨率特征，实现了尺度内的交互。
            out = self.pan_blocks[idx](torch.concat([downsample_feat, feat_high], dim=1))  #跨尺度融合
            #将当前处理后的输出特征图（out）添加到 outs 列表中。
            outs.append(out)   #pan输出
        #
        # if self.use_gpa:
        #     outs = [self.amsf(out) for out in outs]

        return outs
# print([feat.shape for feat in feats])打印输入输出形状