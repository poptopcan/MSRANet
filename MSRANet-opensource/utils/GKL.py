import torch
import torch.nn as nn

class GKLLoss(nn.Module):
    """
    广义KL散度损失（基于2025年最新研究）
    论文: Generalized Kullback-Leibler Divergence Loss
    """
    def __init__(self, alpha=0.1, beta=0.9):
        super().__init__()
        self.alpha = alpha  # 均方误差权重
        self.beta = beta    # 交叉熵权重

    def forward(self, student_logits, teacher_logits, labels):
        """
        参数:
            student_logits: 学生模型输出 (batch×num_classes)
            teacher_logits: 教师模型输出 (batch×num_classes)
            labels: 真实标签 (batch,)
        返回:
            融合损失值
        """
        # 转换为概率分布
        student_prob = torch.softmax(student_logits, dim=1)
        teacher_prob = torch.softmax(teacher_logits, dim=1)
        
        # 1. 加权均方误差部分 (wMSE)
        wmse = torch.mean((student_prob - teacher_prob)**2 * teacher_prob)
        
        # 2. 软标签交叉熵部分
        ce = torch.nn.functional.cross_entropy(student_logits, teacher_prob)
        
        # 3. 类级别全局信息融合
        class_weights = self._get_class_weights(teacher_prob)
        global_term = torch.mean(class_weights * (student_prob - teacher_prob)**2)
        
        # 总损失
        return self.alpha * wmse + self.beta * ce + 0.01 * global_term

    def _get_class_weights(self, teacher_prob):
        """计算类级别权重"""
        class_confidence = teacher_prob.mean(dim=0)
        return 1.0 / (class_confidence + 1e-6)