import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import random

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

def pairwise_distance_np(query_features, gallery_features):    
    x = query_features.detach()
    y = gallery_features.detach()
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1).cpu().numpy()
    y = y.view(n, -1).cpu().numpy()
    
    sum_x = np.sum(x**2, axis=1, keepdims=True)
    sum_y = np.sum(y**2, axis=1, keepdims=True)
    
    dist = sum_x + sum_y.T
    dist -= 2 * np.dot(x, y.T)
    np.fill_diagonal(dist, 0)
    
    return dist 

class Calibration(nn.Module):
    def __init__(self):
        super(Calibration, self).__init__()
        self.softmax = nn.Softmax(dim=2)
        self.relu = nn.ReLU()

    def forward(self, x, x_identity):
        b, c, h, w = x.shape
        c_affinity = self.softmax(x.view(b,c,-1) @ x_identity.view(b,c,-1).transpose(-1,-2))  #[channel,channel]
        s_affinity = self.softmax(x.view(b,c,-1).transpose(-1,-2) @ x_identity.view(b,c,-1)) #[S2,S2]
        c_affinity_avg = c_affinity.mean(dim=-1)
        s_affinity_avg = s_affinity.mean(dim=-1)
        affinity_matrix = torch.einsum('bi,bj->bij', c_affinity_avg, s_affinity_avg)
        affinity_matrix_positive = self.relu(affinity_matrix - 0.5) + 0.5
        affinity_matrix_compesation = self.relu(0.5 - affinity_matrix) + 0.5
        return x * affinity_matrix_positive + x_identity * affinity_matrix_compesation

class IA_MVF(nn.Module):  

    def feature_calibrate(self, x, x_identity):
        b, c, h, w = x.shape
        c_affinity = self.softmax(x.view(b,c,-1) @ x_identity.view(b,c,-1).transpose(-1,-2))  #[channel,channel]
        s_affinity = self.softmax(x.view(b,c,-1).transpose(-1,-2) @ x_identity.view(b,c,-1)) #[S2,S2]
        c_affinity_avg = c_affinity.mean(dim=-1)
        s_affinity_avg = s_affinity.mean(dim=-1)
        affinity_matrix = torch.einsum('bi,bj->bij', c_affinity_avg, s_affinity_avg)
        affinity_matrix_positive = self.relu(affinity_matrix - 0.5) + 0.5
        affinity_matrix_compesation = self.relu(0.5 - affinity_matrix) + 0.5
        return x * affinity_matrix_positive + x_identity * affinity_matrix_compesation 

    def __init__(self, P=6, K=10, dim=2048, refine_ratio=8, h=16, w=8, pn=10):
        super(IA_MVF, self).__init__()

        self.k = K // 2
        self.p = P
        self.pn = max(self.k,pn)
        self.dim = dim
        self.rf_dim = dim // refine_ratio
        self.softmax = nn.Softmax(dim=2)

        self.rf_i = nn.Conv2d(self.dim, self.rf_dim, kernel_size=1, stride=1, bias=False)
        self.rf_v = nn.Conv2d(self.dim, self.rf_dim, kernel_size=1, stride=1, bias=False)

        self.rf_mv_i = nn.Conv2d(self.k*self.rf_dim, self.dim, kernel_size=1, stride=1, bias=False)
        self.rf_mv_v = nn.Conv2d(self.k*self.rf_dim, self.dim, kernel_size=1, stride=1, bias=False)

        self.rm_s_i = nn.Conv2d(self.dim, self.dim, kernel_size=1, stride=1, bias=False)
        self.rm_s_v = nn.Conv2d(self.dim, self.dim, kernel_size=1, stride=1, bias=False)

        self.mv_fuse = nn.Conv2d(2*self.dim, self.dim, kernel_size=1, stride=1, bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(self.dim//2)
        self.IN = nn.InstanceNorm2d(2*self.dim)
        self.sigmoid = nn.Sigmoid()

        for mo in self.modules():
            if isinstance(mo, nn.Conv2d):
                nn.init.kaiming_normal_(mo.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, inputs, labels, sub):  #[b,c,h,w]
        n,c,h,w = inputs.shape
        mask_i = sub == 1
        mask_v = sub == 0

        x_i = inputs[mask_i].detach()
        labels_i = labels[mask_i]
        x_v = inputs[mask_v].detach()
        labels_v = labels[mask_v]

        x_i = self.rf_i(x_i)
        x_v = self.rf_v(x_v) 

        mv_list = torch.zeros((n,self.k*self.rf_dim,h,w), device='cuda').detach()
        for i in range(n):
            idx = labels[i] == labels_i[:]
            mv_list[i] = x_i[idx].view(1,-1,h,w) #b,k*c,h,w
        mvl_i = self.rf_mv_i(mv_list)
        mvl_i = self.feature_calibrate(x_i, mvl_i)
        mvl_i = self.relu(mvl_i)
        for i in range(n):
            idx = labels[i] == labels_v[:]
            mv_list[i] = x_v[idx].view(1,-1,h,w)
        mvl_v = self.rf_mv_v(mv_list)
        mvl_v = self.feature_calibrate(x_v, mvl_v)
        mvl_v = self.relu(mvl_v)
        del mv_list
        mv_i = self.IN(self.rm_s_i(mvl_i))
        mv_v = self.IN(self.rm_s_v(mvl_v))
        # mv_i = self.rm_s_i(mvl_i)
        # mv_v = self.rm_s_v(mvl_v)
    
        x_fuse = torch.cat((mv_i,mv_v),dim=1)
        # x_fuse = self.IN(x_fuse)
        # x_fuse = torch.cat((s_cpct_i,s_cpct_v),dim=1)

        x_fuse = self.sigmoid(self.mv_fuse(x_fuse))
        # with torch.no_grad():
        #   self.mv_prototype = 0.3 * x_fuse.mean(dim=0) + 0.7 * self.mv_prototype  #7,3
        
        feats = inputs + inputs * x_fuse

        return feats, mv_v, mv_i

class II_MVF(nn.Module):  
    def __init__(self, P=6,K=10, dim=2048, refine_ratio=8, h=16, w=8, pn=10):
        super(II_MVF, self).__init__()
        self.p = P
        self.k = K // 2
        self.pn = max(pn,self.k)
        self.dim = dim
        self.rf_dim = dim // refine_ratio

        self.register_buffer("mv_prototype", nn.init.kaiming_normal_(torch.empty(self.pn,self.dim,h,w)))

        self.rf_i = nn.Conv2d(self.dim, self.rf_dim, kernel_size=1, stride=1, bias=False)
        self.rf_v = nn.Conv2d(self.dim, self.rf_dim, kernel_size=1, stride=1, bias=False)
        self.rf = nn.Conv2d(self.dim, self.rf_dim, kernel_size=1, stride=1, bias=False)

        self.rf_mv_i = nn.Conv2d(self.k*self.rf_dim, self.dim, kernel_size=1, stride=1, bias=False)
        self.rf_mv_v = nn.Conv2d(self.k*self.rf_dim, self.dim, kernel_size=1, stride=1, bias=False)
        self.rf_mv = nn.Conv2d(2*self.k*self.rf_dim, self.dim, kernel_size=1, stride=1, bias=False)

        self.rm_s_i = nn.Conv2d(self.dim, self.dim, kernel_size=1, stride=1, bias=False)
        self.rm_s_v = nn.Conv2d(self.dim, self.dim, kernel_size=1, stride=1, bias=False)
        self.rm_s = nn.Conv2d(self.dim, self.dim, kernel_size=1, stride=1, bias=False)

        self.mv_fuse = nn.Conv2d(2*self.dim, self.dim, kernel_size=1, stride=1, bias=False)
        self.fuse = nn.Conv2d(self.dim, self.dim, kernel_size=1, stride=1, bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(self.dim//2)
        self.IN = nn.InstanceNorm2d(2*self.dim)
        self.sigmoid = nn.Sigmoid()
        self.calibration = Calibration()

        for mo in self.modules():
            if isinstance(mo, nn.Conv2d):
                nn.init.kaiming_normal_(mo.weight, mode='fan_out', nonlinearity='relu')
 
    def forward(self, inputs, training=False):  #[b,c,h,w]
        n,c,h,w = inputs.shape
        k = self.k
        ratio = 0.2 #0.3
        if training:
            with torch.no_grad():
                rd_idxs = random.choices(range(n//2), k=self.pn)
            self.mv_prototype = 0.1 * inputs[rd_idxs] + 0.9 * self.mv_prototype
            dist = pairwise_distance(inputs,self.mv_i_prototype).detach().cpu().numpy()
            indices = np.argsort(dist)
            for i in range(n//2):
                idx = indices[i,:k]
                self.mv_i_prototype[idx] = ratio * inputs[i] + (1-ratio) * self.mv_prototype[idx]

        mv_list = torch.zeros((n,2*self.k*self.rf_dim,h,w), device='cuda').detach()

        dist = pairwise_distance(inputs,self.mv_prototype).detach().cpu().numpy()
        indices = np.argsort(dist)

        x = self.rf(self.mv_prototype)

        for i in range(n):
            idx = indices[i,:2*k]
            mv_list = x[idx].view(1,-1,h,w)
        mvl = self.rf_mv(mv_list)
        mvl = self.calibration(x, mvl)
        mvl = self.relu(mvl)
        del mv_list

        mv_a = self.IN(self.rm_s(mvl))
        x_fuse = self.sigmoid(self.fuse(mv_a))
        
        feats = inputs + inputs * x_fuse

        return feats
 