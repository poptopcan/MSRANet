import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

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

class AttentionDistillation(nn.Module):
    """注意力蒸馏模块：对齐学生和教师的注意力分布"""
    def __init__(self, distill_channel=True, distill_spatial=True):
        super().__init__()
        self.distill_channel = distill_channel  # 是否蒸馏通道注意力
        self.distill_spatial = distill_spatial  # 是否蒸馏空间注意力
        
        # 损失函数：使用KL散度或MSE，这里选择MSE更稳定
        self.loss_fn = nn.MSELoss()

    def forward(self, student_attention, teacher_attention):
        """
        计算注意力蒸馏损失
        参数:
            student_attention: 学生模型的注意力权重字典
            teacher_attention: 教师模型的注意力权重字典
        返回:
            总注意力蒸馏损失
        """
        total_loss = 0.0
        loss_components = {}
        
        # 蒸馏通道注意力
        if self.distill_channel:
            channel_loss = 0.0
            # 遍历每一层的通道注意力
            for (s_name, s_att), (t_name, t_att) in zip(
                student_attention['channel'].items(), 
                teacher_attention['channel'].items()
            ):
                # 确保注意力权重形状匹配
                if s_att.shape != t_att.shape:
                    s_att = F.interpolate(s_att, size=t_att.shape[2:], mode='nearest')
                
                # 计算当前层的通道注意力损失
                layer_loss = self.loss_fn(s_att, t_att.detach())  # 教师权重不参与梯度计算
                channel_loss += layer_loss
                loss_components[f'channel_{s_name}'] = layer_loss.item()
            
            # 平均所有层的通道注意力损失
            avg_channel_loss = channel_loss / len(student_attention['channel'])
            total_loss += avg_channel_loss
            loss_components['avg_channel_loss'] = avg_channel_loss.item()
        
        # 蒸馏空间注意力
        if self.distill_spatial:
            spatial_loss = 0.0
            # 遍历每一层的空间注意力
            for (s_name, s_att), (t_name, t_att) in zip(
                student_attention['spatial'].items(), 
                teacher_attention['spatial'].items()
            ):
                # 确保注意力权重形状匹配
                if s_att.shape != t_att.shape:
                    s_att = F.interpolate(s_att, size=t_att.shape[2:], mode='bilinear', align_corners=True)
                
                # 计算当前层的空间注意力损失
                layer_loss = self.loss_fn(s_att, t_att.detach())  # 教师权重不参与梯度计算
                spatial_loss += layer_loss
                loss_components[f'spatial_{s_name}'] = layer_loss.item()
            
            # 平均所有层的空间注意力损失
            avg_spatial_loss = spatial_loss / len(student_attention['spatial'])
            total_loss += avg_spatial_loss
            loss_components['avg_spatial_loss'] = avg_spatial_loss.item()
        
        return total_loss, loss_components

def get_attention_weights(model, x, layer_names, has_attention=False):
    """
    获取模型各层的注意力权重
    参数:
        model: 待提取注意力的模型
        x: 输入数据
        layer_names: 需要提取注意力的层名称列表
        has_attention: 模型是否自带注意力模块
    返回:
        包含通道和空间注意力权重的字典
    """
    attention = {'channel': {}, 'spatial': {}}
    hooks = []
    
    # 如果模型没有自带注意力模块，动态添加
    if not has_attention:
        # 为指定层添加注意力模块
        for name in layer_names:
            layer = dict([*model.named_modules()])[name]
            
            # 保存原始前向传播函数
            original_forward = layer.forward
            
            # 创建新的前向传播函数，添加注意力计算
            def new_forward(self, x, name=name):
                # 计算通道注意力
                ca = ChannelAttention(x.size(1))
                x_ca, c_att = ca(x)
                attention['channel'][name] = c_att
                
                # 计算空间注意力
                sa = SpatialAttention()
                x_sa, s_att = sa(x_ca)
                attention['spatial'][name] = s_att
                
                return original_forward(x_sa)  # 继续原始前向传播
            
            # 替换前向传播函数
            layer.forward = new_forward.__get__(layer, layer.__class__)
    
    # 注册钩子获取注意力权重（适用于自带注意力模块的模型）
    else:
        def hook_fn(module, input, output, name):
            if hasattr(module, 'channel_attention'):
                attention['channel'][name] = module.channel_attention
            if hasattr(module, 'spatial_attention'):
                attention['spatial'][name] = module.spatial_attention
        
        # 为指定层注册钩子
        for name in layer_names:
            layer = dict([*model.named_modules()])[name]
            hook = layer.register_forward_hook(
                lambda m, i, o, name=name: hook_fn(m, i, o, name)
            )
            hooks.append(hook)
    
    # 前向传播计算注意力
    model(x)
    
    # 移除钩子
    for hook in hooks:
        hook.remove()
    
    return attention

def train_with_attention_distillation(student_model, teacher_model, train_dataset, 
                                     epochs=30, batch_size=32, lr=1e-4):
    """
    使用注意力蒸馏训练学生模型
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student_model.to(device)
    teacher_model.to(device)
    teacher_model.eval()  # 教师模型固定
    
    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 需要提取注意力的层（学生和教师应对应）
    layer_names = ['layer1', 'layer2', 'layer3']
    
    # 初始化蒸馏器和优化器
    attention_distiller = AttentionDistillation(distill_channel=True, distill_spatial=True)
    criterion_cls = nn.CrossEntropyLoss()  # 分类损失
    optimizer = torch.optim.Adam(student_model.parameters(), lr=lr)
    
    # 训练循环
    for epoch in range(epochs):
        student_model.train()
        total_loss = 0.0
        total_cls_loss = 0.0
        total_att_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 教师模型前向传播，获取注意力权重
            with torch.no_grad():
                teacher_attention = get_attention_weights(
                    teacher_model, inputs, layer_names, has_attention=True
                )
            
            # 学生模型前向传播，获取注意力权重和分类输出
            student_attention = get_attention_weights(
                student_model, inputs, layer_names, has_attention=True
            )
            student_logits = student_model(inputs)
            
            # 计算损失
            cls_loss = criterion_cls(student_logits, labels)
            att_loss, _ = attention_distiller(student_attention, teacher_attention)
            
            # 总损失：分类损失 + 注意力蒸馏损失（权重可调整）
            loss = cls_loss + 0.5 * att_loss
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 累计损失
            total_loss += loss.item() * inputs.size(0)
            total_cls_loss += cls_loss.item() * inputs.size(0)
            total_att_loss += att_loss.item() * inputs.size(0)
        
        # 计算平均损失
        avg_loss = total_loss / len(train_dataset)
        avg_cls_loss = total_cls_loss / len(train_dataset)
        avg_att_loss = total_att_loss / len(train_dataset)
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Total Loss: {avg_loss:.4f}, Cls Loss: {avg_cls_loss:.4f}, Att Loss: {avg_att_loss:.4f}")
        print("-" * 50)
    
    return student_model

class ChannelAttentionDistillation(nn.Module):
    def __init__(self, reduction=16):
        """
        通道注意力蒸馏：对齐学生和教师的通道重要性
        :param reduction: 注意力压缩系数（降低计算量）
        """
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 通道注意力的核心：全局平均池化
        self.attention = nn.Sequential(
            nn.Linear(None, None),  # 动态匹配通道数，forward中初始化
            nn.ReLU(),
            nn.Linear(None, None),
            nn.Sigmoid()
        )

    def init_attention(self, in_channels):
        """动态初始化注意力层的通道数（避免提前固定输入维度）"""
        self.attention[0] = nn.Linear(in_channels, in_channels // self.reduction)
        self.attention[2] = nn.Linear(in_channels // self.reduction, in_channels)

    def forward(self, student_feat, teacher_feat):
        """
        :param student_feat: 学生特征 (batch, C1, H, W)
        :param teacher_feat: 教师特征 (batch, C2, H, W)
        :return: 通道注意力蒸馏损失
        """
        # 确保学生和教师特征通道数一致（若不一致，先通过1x1卷积匹配）
        if student_feat.shape[1] != teacher_feat.shape[1]:
            student_feat = nn.Conv2d(student_feat.shape[1], teacher_feat.shape[1], 1).to(student_feat.device)(student_feat)
        C = student_feat.shape[1]
        
        # 初始化注意力层（首次调用时）
        if self.attention[0].in_features is None:
            self.init_attention(C)

        # 计算教师通道注意力（固定，不回传梯度）
        teacher_avg = self.avg_pool(teacher_feat).view(-1, C)  # (batch, C)
        teacher_att = self.attention(teacher_avg).detach()  # (batch, C)
        
        # 计算学生通道注意力
        student_avg = self.avg_pool(student_feat).view(-1, C)  # (batch, C)
        student_att = self.attention(student_avg)  # (batch, C)
        
        # 用KL散度或MSE对齐注意力分布（此处用MSE更稳定）
        return F.mse_loss(student_att, teacher_att)