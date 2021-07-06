import math
import torch
import torch.nn.functional as F
from torch import nn


class CosFaceLoss(nn.Module):
    def __init__(self, scale=16, margin=0.1, **kwargs):
        super(CosFaceLoss, self).__init__()
        self.s = scale
        self.m = margin

    def forward(self, inputs, targets):

        one_hot = torch.zeros_like(inputs)
        one_hot.scatter_(1, targets.view(-1, 1), 1.0)

        output = self.s * (inputs - one_hot * self.m)

        return F.cross_entropy(output, targets)


class PairwiseCosFaceLoss(nn.Module):
    def __init__(self, scale=16, margin=0):
        super(PairwiseCosFaceLoss, self).__init__()
        self.s = scale
        self.m = margin

    def forward(self, inputs, targets):
        inputs = F.normalize(inputs, p=2, dim=1)
        similarities = torch.matmul(inputs, inputs.t())

        targets = targets.view(-1,1)
        mask = torch.eq(targets, targets.T).float().cuda()
        mask_self = torch.eye(targets.size(0)).float().cuda()
        mask_pos = mask - mask_self
        mask_neg = 1 - mask

        scores = (similarities + self.m) * mask_neg - similarities * mask_pos
        scores = scores * self.s
        
        neg_scores_LSE = torch.logsumexp(scores*mask_neg - 99999999*(1-mask_neg), dim=1)
        pos_scores_LSE = torch.logsumexp(scores*mask_pos - 99999999*(1-mask_pos), dim=1)

        loss = F.softplus(neg_scores_LSE + pos_scores_LSE).mean()

        return loss