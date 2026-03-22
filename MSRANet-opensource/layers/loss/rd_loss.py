import random
import torch
import torch.nn as nn
import torch.nn.functional as F

def identity_kl(logits_s, logits_t, K):
    KL = nn.KLDivLoss(reduction='batchmean')
    k = K // 2
    logits_s = logits_s.split(k, 0)
    logits_t = logits_t.split(k, 0)
    logits_s = torch.cat(logits_s, 1)  # .t()  # 5,206*12->206*12,5
    logits_t = torch.cat(logits_t, 1)  # .t()
    loss_st = KL(F.log_softmax(logits_s, 1), F.softmax(logits_t, 1))
    loss_ts = KL(F.log_softmax(logits_t, 1), F.softmax(logits_s, 1))
    id_kl_loss = loss_st + loss_ts
    return id_kl_loss

class ChannelAttention(nn.Module):
    """通道通道注意力模块：生成每个通道的重要性权重"""
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # 转换为(b, c)
        y = self.fc(y).view(b, c, 1, 1)  # 转换为(b, c, 1, 1)
        return x * y.expand_as(x), y  # 返回加权特征和注意力权重

class SpatialAttention(nn.Module):
    """空间注意力模块：生成每个空间位置的重要性权重"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 通道平均
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 通道最大
        y = torch.cat([avg_out, max_out], dim=1)  # 拼接为(b, 2, h, w)
        y = self.conv(y)  # 卷积得到(b, 1, h, w)
        return x * self.sigmoid(y), self.sigmoid(y)  # 返回加权特征和注意力权重

def get_attention_weights(x):
    
    # 计算通道注意力
    ca = ChannelAttention(x.size(1))
    x_ca, c_att = ca(x)
    
    # 计算空间注意力
    sa = SpatialAttention()
    x_sa, s_att = sa(x_ca)
    
    return x_ca, c_att, x_sa, s_att

class RelaxedDistillationLoss(nn.Module):
    def __init__(self, mode, distill_channel=True, distill_spatial=True):
        super(RelaxedDistillationLoss, self).__init__()
        self.attention_consistency_loss = F.MSELoss(reduction='mean')
        assert mode in ['sysu','regdb', 'llcm']
        self.mode = mode
        self.distill_channel = distill_channel  # 是否蒸馏通道注意力
        self.distill_spatial = distill_spatial  # 是否蒸馏空间注意力
        
        # 损失函数：使用KL散度或MSE，这里选择MSE更稳定
        self.loss_fn = nn.MSELoss()

    def forward(self, student_features, teacher_features, logits_s, logits_t, sub, temperature=2.0):

        inter_ic_v = identity_kl(logits_s[sub == 0], logits_t[sub == 0], self.k_size)
        inter_ic_i = identity_kl(logits_s[sub == 1], logits_t[sub == 1], self.k_size)
        inter_ic = inter_ic_v + inter_ic_i
        intra_ic = identity_kl(logits_s[sub == 0], logits_s[sub == 1], self.k_size)

        if self.mode == 'sysu':
            ic_loss = intra_ic + inter_ic * 0.8

        elif self.mode == 'regdb':
            ic_loss = intra_ic + inter_ic * 0.3
        
        else:
            ic_loss = intra_ic + inter_ic * 0.3

        student_attention = get_attention_weights(student_features)
        teacher_attention = get_attention_weights(teacher_features)
        
        # 计算通道注意力损失
        channel_loss = self.attention_consistency_loss(student_attention[1], teacher_attention[1].detach())  # 教师权重不参与梯度计算

        # 计算空间注意力损失
        spatial_loss = self.attention_consistency_loss(student_attention[3], teacher_attention[3].detach())  # 教师权重不参与梯度计算

        ac_loss = channel_loss + spatial_loss

        return ic_loss, ac_loss   