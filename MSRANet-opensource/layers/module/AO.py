import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
#去除中间融合，以训练时的共性原型维护测试时的共性
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

class AO_v1(nn.Module):  
    def __init__(self, K=10, dim=2048, cp_ratio=8):
        super(AO_v1, self).__init__()
        self.k = K // 2
        self.dim = dim
        self.cp_dim = dim // cp_ratio
        self.cpct_r_i = nn.Conv2d(self.dim, self.cp_dim, kernel_size=1, stride=1, bias=False)
        self.cpct_r_v = nn.Conv2d(self.dim, self.cp_dim, kernel_size=1, stride=1, bias=False)
        self.cpct_1 = nn.Conv2d(self.dim, self.dim//2, kernel_size=1, stride=1, bias=False)
        self.cpct_i = nn.Conv2d(self.k*self.cp_dim, self.dim, kernel_size=1, stride=1, bias=False)  #out=self.dim//2
        self.cpct_v = nn.Conv2d(self.k*self.cp_dim, self.dim, kernel_size=1, stride=1, bias=False)
        self.fuse_i = nn.Conv2d(self.dim, self.dim, kernel_size=1, stride=1, bias=False)
        self.fuse_v = nn.Conv2d(self.dim, self.dim, kernel_size=1, stride=1, bias=False)
        self.fuse = nn.Conv2d(2*self.dim, self.dim, kernel_size=1, stride=1, bias=False)
        self.fuse_ms = nn.Conv2d(2*self.dim, self.dim, kernel_size=1, stride=1, bias=False)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(self.dim//2)
        self.IN = nn.InstanceNorm2d(self.dim//2)
        self.sigmoid = nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, inputs, labels, sub):  #[b,c,h,w]
        n,c,h,w = inputs.shape
        mask_i = sub == 1
        mask_v = sub == 0

        x_i = inputs[mask_i].detach()
        labels_i = labels[mask_i]
        x_v = inputs[mask_v].detach()
        labels_v = labels[mask_v]
        x_i = self.cpct_r_i(x_i)
        x_v = self.cpct_r_v(x_v)
        # x_cpct = self.relu(self.cpct_1(inputs))
        aff_list = torch.zeros((n,self.k*self.cp_dim,h,w), device='cuda').detach()
        for i in range(n):
            idx = labels[i] == labels_i[:]
            aff_list[i] = x_i[idx].view(1,-1,h,w) #b+1,k*c,h,w
        s_cpct_i = self.relu(self.cpct_i(aff_list))
        for i in range(n):
            idx = labels[i] == labels_v[:]
            aff_list[i] = x_v[idx].view(1,-1,h,w) #b+1,k*c,h,w
        s_cpct_v = self.relu(self.cpct_v(aff_list))
        del aff_list
        # x_si_f = torch.cat((x_cpct,s_cpct_i),dim=1)
        # x_sv_f = torch.cat((x_cpct,s_cpct_v),dim=1)
        # x_si_f = self.IN(self.fuse_i(s_cpct_i))
        # x_sv_f = self.IN(self.fuse_v(s_cpct_v))

        x_fuse = torch.cat((s_cpct_i,s_cpct_v),dim=1)
        # x_fuse = torch.cat((s_cpct_i,s_cpct_v),dim=1)
        x_fuse = self.sigmoid(self.fuse(x_fuse))
        # x_fuse = self.fuse(x_fuse)
        
        feats = inputs + inputs * x_fuse
        # feats = self.fuse_ms(torch.cat((inputs,x_fuse),dim=1))
        return feats

class AOL_v(nn.Module):  
    def __init__(self, K=10, dim=2048, cp_ratio=8, h=16, w=8):
        super(AOL_v1, self).__init__()
        self.k = K // 2
        self.dim = dim
        self.cp_dim = dim // cp_ratio
        self.cpct_r = nn.Conv2d(self.dim, self.cp_dim, kernel_size=1, stride=1, bias=False)
        self.cpct_1 = nn.Conv2d(self.dim, self.dim//2, kernel_size=1, stride=1, bias=False)
        self.cpct_list = nn.Conv2d(self.k*self.cp_dim, self.dim, kernel_size=1, stride=1, bias=False)   #out=self.dim//2
        self.fuse = nn.Conv2d(self.dim, self.dim, kernel_size=1, stride=1, bias=False)
        self.conv = nn.Conv2d(self.dim, self.dim, kernel_size=1, stride=1, bias=False)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(self.dim//2)
        self.IN = nn.InstanceNorm2d(self.dim//2)
        self.sigmoid = nn.Sigmoid()
        self.similar_prototype = nn.Parameter(nn.init.kaiming_normal_(torch.empty(self.dim,h,w)))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, inputs, labels):  #[b,c,h,w]
        n,c,h,w = inputs.shape
        k = self.k

        feat_cp = self.cpct_r(inputs)
        dist = pairwise_distance(inputs,inputs).detach().cpu().numpy()
        indices = np.argsort(dist)
        neg_neigh_ids = []
        if self.training:
            aff_list = torch.zeros((n,k*self.cp_dim,h,w), device='cuda').detach()
            for i in range(n):
                fkn_idx = indices[i,:k]
                bkn_idx = indices[fkn_idx,:k]
                fi = np.where(bkn_idx==i)[0]
                mask = (labels[i] == labels[fkn_idx[fi]]).cpu().numpy()
                neg_neigh_ids.append(fkn_idx[fi[~mask]])
                if len(fi) < k:
                    fi = np.random.choice(fi,k,True)
                aff_list[i] = feat_cp[fkn_idx[fi]].view(1,-1,h,w) #b,k*cp_dim,h,w
            # for i in range(n):
            #     fkn_idx = indices[i,:k]
            #     aff_list[i] = feat_cp[fkn_idx].view(1,-1,h,w) #b,k*cp_dim,h,w
            s_cpct_list = self.relu(self.cpct_list(aff_list))
            del aff_list
            similar_cf = s_cpct_list.mean(dim=0)
            self.similar_prototype.data = 0.7 * similar_cf + 0.3 * self.similar_prototype.data

        else:
            s_cpct_list = self.similar_prototype.unsqueeze(0)

        # x_cpct = self.relu(self.cpct_1(inputs))
        # x_f = torch.cat((x_cpct,s_cpct_list),dim=1)
        # x_f = self.IN(self.fuse(s_cpct_list))
        x_f = self.sigmoid(self.conv(s_cpct_list))
        
        feats = inputs + inputs * x_f

        return feats, neg_neigh_ids
    
class AOL_v2(nn.Module):  
    def __init__(self, K=10, dim=2048, cp_ratio=8, h=16, w=8, m=1):
        super(AOL_v1, self).__init__()
        self.k = K // 2
        self.dim = dim
        self.cp_dim = dim // cp_ratio
        self.cpct_r = nn.Conv2d(self.dim, self.cp_dim, kernel_size=1, stride=1, bias=False)
        self.cpct_1 = nn.Conv2d(self.dim, self.dim//2, kernel_size=1, stride=1, bias=False)
        self.cpct_list = nn.Conv2d(self.k*self.cp_dim, self.dim, kernel_size=1, stride=1, bias=False)   #out=self.dim//2
        self.fuse = nn.Conv2d(self.dim, self.dim, kernel_size=1, stride=1, bias=False)
        self.conv = nn.Conv2d(self.dim, self.dim, kernel_size=1, stride=1, bias=False)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(self.dim//2)
        self.IN = nn.InstanceNorm2d(self.dim//2)
        self.classfier_pro = nn.Linear(dim,m)
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        # self.similar_prototype = nn.Parameter(nn.init.kaiming_normal_(torch.empty(m,self.dim,h,w)))
        self.register_buffer("similar_prototype", nn.init.kaiming_normal_(torch.empty(m,self.dim,h,w)))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, inputs, labels):  #[b,c,h,w]
        n,c,h,w = inputs.shape
        k = self.k

        feat_cp = self.cpct_r(inputs)
        dist = pairwise_distance(inputs,inputs).detach().cpu().numpy()
        indices = np.argsort(dist)
        neg_neigh_ids = []
        if self.training:
            aff_list = torch.zeros((n,k*self.cp_dim,h,w), device='cuda').detach()
            for i in range(n):
                fkn_idx = indices[i,:k]
                bkn_idx = indices[fkn_idx,:k]
                fi = np.where(bkn_idx==i)[0]
                mask = (labels[i] == labels[fkn_idx[fi]]).cpu().numpy()
                neg_neigh_ids.append(fkn_idx[fi[~mask]])
                if len(fi) < k:
                    fi = np.random.choice(fi,k,True)
                aff_list[i] = feat_cp[fkn_idx[fi]].view(1,-1,h,w) #b,k*cp_dim,h,w
            # for i in range(n):
            #     fkn_idx = indices[i,:k]
            #     aff_list[i] = feat_cp[fkn_idx].view(1,-1,h,w) #b,k*cp_dim,h,w
            s_cpct_list = self.relu(self.cpct_list(aff_list))
            del aff_list
            similar_cf = s_cpct_list.mean(dim=0)    #考虑最突出的共性，以阈值生成掩码或注意力
            # scf_pool = similar_cf.mean(dim=(1,2))
            # scf_idx = torch.argmax(self.softmax(self.classfier_pro(scf_pool)))
            # self.similar_prototype.data[scf_idx] = 0.4 * similar_cf + 0.6 * self.similar_prototype.data[scf_idx]
            # self.similar_prototype[scf_idx] = 0.4 * similar_cf + 0.6 * self.similar_prototype[scf_idx]
            self.similar_prototype = 0.7 * similar_cf + 0.3 * self.similar_prototype

        else:
            # inputs_pool = inputs.mean(dim=(2,3))
            # in_idx = torch.argmax(self.softmax(self.classfier_pro(inputs_pool)),dim=-1)
            # s_cpct_list = self.similar_prototype.data[in_idx]
            s_cpct_list = self.similar_prototype

        # x_cpct = self.relu(self.cpct_1(inputs))
        # x_f = torch.cat((x_cpct,s_cpct_list),dim=1)
        # x_f = self.IN(self.fuse(s_cpct_list))
        x_f = self.sigmoid(self.conv(s_cpct_list))
        
        feats = inputs + inputs * x_f

        return feats, neg_neigh_ids
    
class AOL_v1(nn.Module):  
    def __init__(self, K=10, dim=2048, cp_ratio=8, h=16, w=8, m=1):
        super(AOL_v1, self).__init__()
        self.k = K // 2
        self.dim = dim
        self.cp_dim = dim // cp_ratio
        self.sigmoid = nn.Sigmoid()
        self.forecast = nn.Sequential(
            nn.Conv2d(self.dim, self.cp_dim, kernel_size=1, stride=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.cp_dim, self.dim, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid()
        )
        self.fuse_ms = nn.Conv2d(2*self.dim, self.dim, kernel_size=1, stride=1, bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, inputs, labels):  #[b,c,h,w]
        
        neg_neigh_ids = []
        x_f = self.forecast(inputs)
        
        feats = inputs + inputs * x_f
        # feats = self.fuse_ms(torch.cat((inputs,x_f),dim=1))

        return feats, neg_neigh_ids