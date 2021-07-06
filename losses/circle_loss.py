import math
import torch
import torch.nn.functional as F
from torch import nn


class CircleLoss(nn.Module):
    def __init__(self, scale=96, margin=0.3, **kwargs):
        super(CircleLoss, self).__init__()
        self.s = scale
        self.m = margin

    def forward(self, inputs, targets):

        mask = torch.zeros_like(inputs).cuda()
        mask.scatter_(1, targets.view(-1, 1), 1.0)
    
        pos_scale = self.s * F.relu(1 + self.m - inputs.detach())
        neg_scale = self.s * F.relu(inputs.detach() + self.m)
        scale_matrix = pos_scale * mask + neg_scale * (1 - mask)

        scores = (inputs - (1 - self.m) * mask - self.m * (1 - mask)) * scale_matrix
        
        loss = F.cross_entropy(scores, targets)

        return loss


class PairwiseCircleLoss(nn.Module):
    def __init__(self, scale=48, margin=0.35, **kwargs):
        super(PairwiseCircleLoss, self).__init__()
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

        pos_scale = self.s * F.relu(1 + self.m - similarities.detach())
        neg_scale = self.s * F.relu(similarities.detach() + self.m)
        scale_matrix = pos_scale * mask_pos + neg_scale * mask_neg

        scores = (similarities - self.m) * mask_neg + (1 - self.m - similarities) * mask_pos
        scores = scores * scale_matrix
        
        neg_scores_LSE = torch.logsumexp(scores*mask_neg - 99999999*(1-mask_neg), dim=1)
        pos_scores_LSE = torch.logsumexp(scores*mask_pos - 99999999*(1-mask_pos), dim=1)

        loss = F.softplus(neg_scores_LSE + pos_scores_LSE).mean()

        return loss
