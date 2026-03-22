import torch
import torch.nn as nn
from torch.nn import functional as F
import gc
import math
import random

from models.resnet import resnet50, embed_net
from utils.calc_acc import calc_acc

from layers import TripletLoss
from layers.loss.rd_loss import RelaxedDistillationLoss
from layers.module.RA import IA_MVF, II_MVF
from utils.tnse import visual

def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)

def gem_p(x):
    x_gp = gem(x).squeeze()  
    x_gp = x_gp.view(x_gp.size(0), -1)  
    return x_gp

def pairwise_dist(x, y):
    
    xx = (x**2).sum(dim=1, keepdim=True)
    yy = (y**2).sum(dim=1, keepdim=True).t()
    dist = xx + yy - 2.0 * torch.mm(x, y.t())
    dist = dist.clamp(min=1e-6).sqrt()
    return dist

def kl_soft_dist(feat1,feat2):
    n_st = feat1.size(0)
    dist_st = pairwise_dist(feat1, feat2)
    mask_st_1 = torch.ones(n_st, n_st, dtype=bool)
    for i in range(n_st):  
        mask_st_1[i, i] = 0
    dist_st_2 = []
    for i in range(n_st):
        dist_st_2.append(dist_st[i][mask_st_1[i]])
    dist_st_2 = torch.stack(dist_st_2)
    return dist_st_2

def Bg_kl(logits1, logits2):    #输入:(60,206),(60,206)
    KL = nn.KLDivLoss(reduction='batchmean')
    kl_loss_12 = KL(F.log_softmax(logits1, 1), F.softmax(logits2, 1))
    kl_loss_21 = KL(F.log_softmax(logits2, 1), F.softmax(logits1, 1))
    bg_loss_kl = kl_loss_12 + kl_loss_21
    return kl_loss_12, bg_loss_kl

def Sm_kl(logits1, logits2, labels):
    KL = nn.KLDivLoss(reduction='batchmean')
    m_kl = torch.div((labels == labels[0]).sum(), 2, rounding_mode='floor')
    v_logits_s = logits1.split(m_kl, 0)
    i_logits_s = logits2.split(m_kl, 0)
    sm_v_logits = torch.cat(v_logits_s, 1)  # .t()  # 5,206*12->206*12,5
    sm_i_logits = torch.cat(i_logits_s, 1)  # .t()
    sm_kl_loss_vi = KL(F.log_softmax(sm_v_logits, 1), F.softmax(sm_i_logits, 1))
    sm_kl_loss_iv = KL(F.log_softmax(sm_i_logits, 1), F.softmax(sm_v_logits, 1))
    sm_kl_loss = sm_kl_loss_vi + sm_kl_loss_iv
    return sm_kl_loss_vi, sm_kl_loss

def Random_kl(logits1, logits2, k, kn=20):
    KL = nn.KLDivLoss(reduction='batchmean')
    idx = random.sample(range(logits1.size(0)),kn)
    logits1 = logits1[idx]
    logits2 = logits2[idx]
    kl_loss_12 = KL(F.log_softmax(logits1, 1), F.softmax(logits2, 1))
    kl_loss_21 = KL(F.log_softmax(logits2, 1), F.softmax(logits1, 1))
    rd_loss_kl = kl_loss_12 + kl_loss_21

    m_kl = k // 2
    logits_1 = logits1.split(m_kl, 0)
    logits_2 = logits2.split(m_kl, 0)
    ikl_logits_1 = torch.cat(logits_1, 1)  
    ikl_logits_2 = torch.cat(logits_2, 1)  
    ikl_loss_12 = KL(F.log_softmax(ikl_logits_1, 1), F.softmax(ikl_logits_2, 1))
    ikl_loss_21 = KL(F.log_softmax(ikl_logits_2, 1), F.softmax(ikl_logits_1, 1))
    ikl_loss = ikl_loss_12 + ikl_loss_21

    loss_kl = rd_loss_kl + ikl_loss
    return loss_kl

def Cm_kl(logits1, logits2, k):
    KL = nn.KLDivLoss(reduction='batchmean')
    m_kl = k // 2
    v_logits_s = logits1.split(m_kl, 0)
    i_logits_s = logits2.split(m_kl, 0)
    sm_v_logits = torch.cat(v_logits_s, 1)  # .t()  # 5,206*12->206*12,5
    sm_i_logits = torch.cat(i_logits_s, 1)  # .t()
    sm_kl_loss_vi = KL(F.log_softmax(sm_v_logits, 1), F.softmax(sm_i_logits, 1))
    sm_kl_loss_iv = KL(F.log_softmax(sm_i_logits, 1), F.softmax(sm_v_logits, 1))
    sm_kl_loss = sm_kl_loss_vi + sm_kl_loss_iv
    return sm_kl_loss_vi, sm_kl_loss

class Baseline(nn.Module):
    def __init__(self, num_classes=None, drop_last_stride=False, decompose=False, **kwargs):
        super(Baseline, self).__init__()

        self.drop_last_stride = drop_last_stride
        self.decompose = decompose
        self.cli = kwargs.get('cli', False)
        self.backbone = embed_net(drop_last_stride=drop_last_stride, decompose=decompose, mm = self.cli)

        self.base_dim = 2048
        self.dim = 2048
        self.part_num = kwargs.get('num_parts', 0)
        self.mutual_learning = kwargs.get('mutual_learning', False)

        print("output feat length:{}".format(self.base_dim + self.dim * self.part_num))
        self.bn_neck = nn.BatchNorm1d(self.base_dim + self.dim * self.part_num)
        nn.init.constant_(self.bn_neck.bias, 0) 
        self.bn_neck.bias.requires_grad_(False)
        self.bn_neck_sp = nn.BatchNorm1d(self.base_dim + self.dim * self.part_num)
        nn.init.constant_(self.bn_neck_sp.bias, 0)
        self.bn_neck_sp.bias.requires_grad_(False)

        if kwargs.get('eval', False):
            return

        self.classification = kwargs.get('classification', False)
        self.triplet = kwargs.get('triplet', False)
        self.center_cluster = kwargs.get('center_cluster', False)
        self.center_loss = kwargs.get('center', False)
        self.margin = kwargs.get('margin', 0.3)
        self.CSA1 = kwargs.get('bg_kl', False)
        self.CSA2 = kwargs.get('sm_kl', False)
        self.TGSA = kwargs.get('distalign', False)
        self.IP = kwargs.get('IP', False)
        self.fb_dt = kwargs.get('fb_dt', False)
        self.p_size = kwargs.get('p_size', 6)
        self.k_size = kwargs.get('k_size', 10)
        self.ao = kwargs.get('ao', False)
        self.edb = kwargs.get('edb', False)
        self.dataset = kwargs.get('dataset', 'default')
        self.cnt = -1
        self.pn = 30
        # self.save_num = 10

        self.rd_loss_fn = RelaxedDistillationLoss(mode = self.dataset)
        if self.ao:
            if num_classes==206:
                self.II_mvf = II_MVF(P=self.p_size, K=self.k_size, dim=self.base_dim, refine_ratio=4,h=16,w=8, pn=self.pn)#reg16*8 sysu24*9
                self.IA_mvf = IA_MVF(P=self.p_size, K=self.k_size, dim=self.base_dim, refine_ratio=4,h=16,w=8, pn=self.pn)
                # self.II_maf = II_MAF(P=self.p_size, K=self.k_size, dim=self.base_dim, refine_ratio=4,h=16,w=8)#reg16*8 sysu24*9
                # self.IA_maf = IA_MAF(P=self.p_size, K=self.k_size, dim=self.base_dim, refine_ratio=4,h=16,w=8)
            else:
                self.II_mvf = II_MVF(P=self.p_size, K=self.k_size, dim=self.base_dim, refine_ratio=4,h=24,w=9, pn=self.pn)#reg16*8 sysu24*9
                self.IA_mvf = IA_MVF(P=self.p_size, K=self.k_size, dim=self.base_dim, refine_ratio=4,h=24,w=9, pn=self.pn)
                # self.II_maf = II_MAF(P=self.p_size, K=self.k_size, dim=self.base_dim, refine_ratio=4,h=24,w=9)#reg16*8 sysu24*9
                # self.IA_maf = IA_MAF(P=self.p_size, K=self.k_size, dim=self.base_dim, refine_ratio=4,h=24,w=9)
            
            # self.ao_sp = AO_v1(K=self.k_size, dim=self.base_dim)
            # if self.p_size==6:
            #     self.aol_sh = AOL_v1(K=self.k_size, dim=self.base_dim, cp_ratio=4,h=16,w=8)#reg16*8 sysu24*9
            # else:
            #     self.aol_sh = AOL_v1(K=self.k_size, dim=self.base_dim, cp_ratio=4,h=24,w=9)#reg16*8 sysu24*9
            # self.ed_loss_fn = EDLoss()

        if self.decompose:
            self.classifier = nn.Linear(self.base_dim + self.dim * self.part_num, num_classes, bias=False)
            self.classifier_sp = nn.Linear(self.base_dim + self.dim * self.part_num, num_classes, bias=False)
            self.classifier_vis = nn.Linear(self.base_dim + self.dim * self.part_num, num_classes, bias=False)
            self.classifier_inf = nn.Linear(self.base_dim + self.dim * self.part_num, num_classes, bias=False)

            self.C_sp_f1 = nn.Linear(self.base_dim , num_classes, bias=False)
            self.C_sp_f2 = nn.Linear(self.base_dim , num_classes, bias=False)

        else:
            self.classifier_sp = nn.Linear(self.base_dim + self.dim * self.part_num, num_classes, bias=False)
            self.classifier = nn.Linear(self.base_dim + self.dim * self.part_num, num_classes, bias=False)
        if self.classification:
            self.id_loss = nn.CrossEntropyLoss(ignore_index=-1)
        if self.triplet:
            self.triplet_loss = TripletLoss(margin=self.margin)  

    def forward(self, inputs, labels=None, **kwargs):

        cam_ids = kwargs.get('cam_ids')
        sub = (cam_ids == 3) + (cam_ids == 6)
        
        
        x_IA, x_II= self.backbone(inputs)
        mv_i_p = None
        mv_v_p = None
        neg_neigh_ids = []
        IA_v_a = None
        IA_i_a = None
        # if self.ao:
        #     if self.training:
        #         x_IA, mv_i_p, mv_v_p = self.IA_mvf(x_IA, labels, sub)
            # x_II = self.II_mvf(x_II, mv_i_p, mv_v_p)
        if self.ao:
            if self.training:
                x_IA, IA_v_a, IA_i_a = self.IA_mvf(x_IA, labels, sub)
                
            x_II = self.II_mvf(x_II, self.training)
            

        sh_pl = gem(x_II).squeeze()  # Gem池化
        sh_pl = sh_pl.view(sh_pl.size(0), -1)  # Gem池化
        sp_pl = gem(x_IA).squeeze()  # Gem池化
        sp_pl = sp_pl.view(sp_pl.size(0), -1)  # Gem池化
        
        feats = sh_pl

        if not self.training:
            if feats.size(0) == 2048:
                feats = self.bn_neck(feats.permute(1, 0))
                logits = self.classifier(feats)
                return logits  # feats #

            else:
                feats = self.bn_neck(feats)
                return feats

        else:
            return self.train_forward(feats, sp_pl, labels,
                                       sub, neg_neigh_ids, IA_v_a, IA_i_a, **kwargs)

    def train_forward(self, feat, sp_pl, labels,
                       sub, neg_neigh_ids, IA_v_a, IA_i_a, **kwargs):
        epoch = kwargs.get('epoch')
        metric = {}
        loss = torch.tensor(0.0,device='cuda')

        # feat = 0.5 * feat + 0.5 * sp_pl

        if self.triplet:

            triplet_loss, dist, sh_ap, sh_an = self.triplet_loss(feat.float(), labels)
            triplet_loss_im, dist, sp_ap, sp_an = self.triplet_loss(sp_pl.float(), labels)
            trip_loss = triplet_loss + triplet_loss_im
            loss += trip_loss
            metric.update({'tri': trip_loss.data})


        bb = 120  #90
        nn = 120
        
        feat = self.bn_neck(feat)
        sp_pl = self.bn_neck_sp(sp_pl)

        if self.decompose:
            logits_sp = self.classifier_sp(sp_pl)  # self.bn_neck_un(sp_pl)
            loss_id_sp = self.id_loss(logits_sp.float(), labels)
            loss += loss_id_sp

        if self.classification:
            logits = self.classifier(feat)
            rd_loss = self.rd_loss_fn(feat, sp_pl, logits, logits_sp, sub)
            loss += rd_loss

            metric.update({'rd': rd_loss.data})
            
            if self.CSA1:

                _, inter_bg_v = Bg_kl(logits[sub == 0], logits_sp[sub == 0])
                _, inter_bg_i = Bg_kl(logits[sub == 1], logits_sp[sub == 1])

                _, intra_bg = Bg_kl(logits[sub == 0], logits[sub == 1])


                if feat.size(0) == bb:
                    bg_loss = intra_bg + (inter_bg_v + inter_bg_i) * 0.8  # intra_bg + (inter_bg_v + inter_bg_i) * 0.7

                else:
                    bg_loss = intra_bg + (inter_bg_v + inter_bg_i) * 0.3
                loss += bg_loss

                metric.update({'csa': bg_loss.data})

            if self.CSA2:
                # _, inter_Sm_v = Sm_kl(logits[sub == 0], logits_sp[sub == 0], labels)
                # _, inter_Sm_i = Sm_kl(logits[sub == 1], logits_sp[sub == 1], labels)
                # inter_Sm = inter_Sm_v + inter_Sm_i
                # _, intra_Sm = Sm_kl(logits[sub == 0], logits[sub == 1], labels)
                _, inter_Sm_v = Cm_kl(logits[sub == 0], logits_sp[sub == 0], self.k_size)
                _, inter_Sm_i = Cm_kl(logits[sub == 1], logits_sp[sub == 1], self.k_size)
                inter_Sm = inter_Sm_v + inter_Sm_i
                _, intra_Sm = Cm_kl(logits[sub == 0], logits[sub == 1], self.k_size)


                if feat.size(0) == bb:
                    sm_kl_loss = intra_Sm + inter_Sm * 0.8

                else:
                    sm_kl_loss = intra_Sm + inter_Sm * 0.3
                loss += sm_kl_loss
                metric.update({'msa': sm_kl_loss.data})

            random_kl = False
            if random_kl:
                inter_rkl_1 = Random_kl(logits[sub == 0], logits_sp[sub == 0], k=self.k_size, kn=20)
                inter_rkl_2 = Random_kl(logits[sub == 1], logits_sp[sub == 1], k=self.k_size, kn=20)
                inter_rkl = inter_rkl_1 + inter_rkl_2
                intra_rkl = Random_kl(logits[sub == 0], logits[sub == 1], k=self.k_size, kn=20)

                if feat.size(0) == nn:
                    rd_kl_loss = intra_rkl + inter_rkl * 0.8

                else:
                    rd_kl_loss = intra_rkl + inter_rkl * 0.3
                loss += rd_kl_loss
                metric.update({'rkl': rd_kl_loss.data})

            cls_loss = self.id_loss(logits.float(), labels)
            loss += cls_loss
            metric.update({'acc': calc_acc(logits.data, labels), 'ce': cls_loss.data})

        return loss, metric
