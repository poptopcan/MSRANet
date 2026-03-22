import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

def pairwise_distance(query_features, gallery_features):    
    x = query_features
    y = gallery_features
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
            torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist.addmm_(x, y.t(), beta=1, alpha=-2)
    return dist

class EDBLoss(nn.Module):
    def __init__(self, edb_w=1, ae_w=10, k=10, margin1=1.3, margin2=0.5):
        super(EDBLoss, self).__init__()
        self.edb_w = edb_w
        self.ae_w = ae_w
        self.k = k
        self.margin1 = margin1
        self.margin2 = margin2

    def forward(self, inputs, labels): 
        n = inputs.size(0)
        k = self.k

        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()

        dist = dist.detach().cpu()
        indices = np.argsort(dist)
        mask = labels.expand(n, n).eq(labels.expand(n, n).t()).cpu()
        for i in range(n):
            mask[i] = mask[i][indices[i]]

        loss_ae, loss_an, loss_ap = 0, 0, 0

        for i in range(n):
            border_value = dist[i][indices[i][k-1]]
            center_ap = dist[i][indices[i]][k:][mask[i][k:]]
            center_an = dist[i][indices[i]][:k][~mask[i][:k]]
            center_ae = dist[i][indices[i]][:k][mask[i][:k]]
            if center_ap.numel() > 0:
                loss_ap += (center_ap - border_value).mean()
            if center_an.numel() > 0:
                loss_an += ( -center_an + border_value + self.margin1).clamp(min=0).mean()
            if center_ae.numel() > 0:
                loss_ae += (-center_ae + self.margin2).clamp(min=0).mean()

        return loss_ap / n, loss_an / n, loss_ae / n
    
