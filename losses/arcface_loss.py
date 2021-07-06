import math
import torch
import torch.nn.functional as F
from torch import nn


class ArcFaceLoss(nn.Module):
    def __init__(self, margin=0.1, scale=16, easy_margin=False):
        super(ArcFaceLoss, self).__init__()
        self.m = margin
        self.s = scale
        self.easy_margin = easy_margin

    def forward(self, input, target):

        # make a one-hot index
        index = input.data * 0.0 #size=(B,Classnum)
        index.scatter_(1,target.data.view(-1,1),1)
        index = index.bool()

        cos_m = math.cos(self.m)
        sin_m = math.sin(self.m)
        cos_t = input[index]
        sin_t = torch.sqrt(1.0 - cos_t * cos_t)
        cos_t_add_m = cos_t * cos_m  - sin_t * sin_m

        if self.easy_margin:
            cond = F.relu(cos_t)
            keep = cos_t
        else:
            cond_v = cos_t - math.cos(math.pi - self.m)
            cond = F.relu(cond_v)
            keep = cos_t - math.sin(math.pi - self.m) * self.m

        cos_t_add_m = torch.where(cond.bool(), cos_t_add_m, keep)

        output = input * 1.0 #size=(B,Classnum)
        output[index] = cos_t_add_m
        output = self.s * output

        return F.cross_entropy(output, target)
