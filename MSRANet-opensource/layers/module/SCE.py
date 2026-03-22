import torch
import torch.nn as nn

class CLI(nn.Module):   #跨层交互模块
    def __init__(self, in_channel, expansion=2, downsample=True):
        super(CLI,self).__init__()
        if downsample:
            self.conv = nn.Conv2d(in_channel, expansion*in_channel, kernel_size=3, stride=2, padding=1, bias=False)
            self.smooth = nn.Conv2d(expansion*in_channel, expansion*in_channel, kernel_size=3, stride=1, padding=1, bias=False)

        else:
            self.conv = nn.Conv2d(in_channel, expansion*in_channel, kernel_size=3, stride=1, padding=1, bias=False)
            self.smooth = nn.Conv2d(expansion*in_channel, expansion*in_channel, kernel_size=3, stride=1, padding=1, bias=False)

        self.softmax = nn.Softmax(dim=2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self,Fl, Fh): #Fh:[2c,h/2,w/2];Fl:[c,h,w]

        b1,c1,h1,w1 = Fh.shape

        s1 = h1*w1

        Fl_e = self.conv(Fl)
        Fl_e = self.smooth(Fl_e)
        
        el_ram = self.softmax(Fh.view(b1,c1,-1) @ Fl_e.view(b1,c1,-1).transpose(-1,-2))  #[in_channel,in_channel]
        el_cam = self.softmax(Fh.view(b1,c1,-1).transpose(-1,-2) @ Fl_e.view(b1,c1,-1)) #[S2,S2]

        Fh_sw = el_ram @ Fh.view(b1,c1,-1) @ el_cam / c1 / s1    

        Fh_op = 0.5 * Fh + 0.5 * Fh_sw.view(b1, c1, h1, w1)
        # Fh_op = 0.5 * Fh + 0.5 * Fl_e.view(b1, c1, h1, w1)

        return Fh_op 
    
class SCE(nn.Module):   #多尺度语义融合模块
    def __init__(self, in_channel, out_channel, downsample=True):
        super(SCE,self).__init__()
        if downsample:
            self.trans = nn.Sequential(
                nn.ConvTranspose2d(out_channel, in_channel, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1)
            )
        else:
            self.trans = nn.Sequential(
                nn.ConvTranspose2d(out_channel, in_channel, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1)
            )

        self.learner = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1),
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1)
        )
        self.softmax = nn.Softmax(dim=2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self,Fl, Fh): #Fh:[2c,h/2,w/2];Fl:[c,h,w]

        b1,c1,h1,w1 = Fl.shape
        b2,c2,h2,w2 = Fh.shape

        s1 = h1*w1

        Fh_trans = self.trans(Fh)
        
        c_matrix = self.softmax(Fl.view(b1,c1,-1) @ Fh_trans.view(b1,c1,-1).transpose(-1,-2))  #[in_channel,in_channel]
        s_matrix = self.softmax(Fl.view(b1,c1,-1).transpose(-1,-2) @ Fh_trans.view(b1,c1,-1)) #[S2,S2]

        Fl_fuse = c_matrix @ Fl.view(b1,c1,-1) @ s_matrix / c1 / s1  

        Fl_fuse_proj = self.learner(Fl_fuse)

        Fh_sce = 0.5 * Fh + 0.5 * Fl_fuse_proj.view(b2, c2, h2, w2)

        return Fh_sce 