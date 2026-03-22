import torch
import random
import torchvision.transforms as T

class RandomGrayscale:  #随机灰度变换
    def __init__(self) -> None:
        pass

    def __call__(self, x):
        op = random.randint(0, 10)

        if op == 0:
            x[1] = x[2] = x[0]
        elif op == 1:
            x[0] = x[2] = x[1]
        elif op == 2:
            x[0] = x[1] = x[2]

        return x

class I2V:
    def __init__(self) -> None:
        pass

    def __call__(self, x):
        op = random.randint(0, 5)
        if op < 1:
            x[0] = x[2] / 0.299
            x[1] = x[2] / 0.587
            x[2] = x[2] / 0.114
        return x

class RandomResize:
    def __init__(self):
        self.smaller_resizer = T.Resize((336, 128))
        self.bigger_resizer = T.Resize((384, 128))
        self.mask = [i<336 for i in range(384)] #384个掩码值,<336为true

    def __call__(self, x):
        if random.random() < 0.5:   #随机浮点数
            x = self.smaller_resizer(x)
            return x
        else:
            x = self.bigger_resizer(x)
            # random.shuffle(self.mask)
            x = x[:, self.mask] #保留[:,0:336,128]
            return x

class RandomImageErasing:
    def __init__(self, scale) -> None:
        self.lr = scale[0]  #最小比例
        self.hr = scale[1]

    def __call__(self, x):
        H, W = x.shape[1], x.shape[2]        
        ratio_mask = random.random() * (self.hr - self.lr) + self.lr    #擦除比例
        area_mask = H * W * ratio_mask

        ratio_hw = random.random() * (2 - 0.5) + 0.5 
        h = int((area_mask * ratio_hw) ** 0.5)
        w = int(area_mask / h)

        src_h, src_w = self.get_pos(H, h, W, w)     #返回起始位置
        dst_h, dst_w = self.get_pos(H, h, W, w)
        
        x[:, src_h:src_h+h, src_w:src_w+w] = x[:, dst_h:dst_h+h, dst_w:dst_w+w]
        return x

    def get_pos(self, H, h, W, w):
        pos_h = int(random.random() * (H - h - 1))
        pos_w = int(random.random() * (W - w - 1))
        return pos_h, pos_w