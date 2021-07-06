import math
import torch
import torch.nn.functional as F
from torch import nn


class ContrastiveLoss(nn.Module):
    def __init__(self, scale=16, **kwargs):
        super(ContrastiveLoss, self).__init__()
        self.scale = scale

    def forward(self, inputs, targets):
        inputs = F.normalize(inputs, p=2, dim=1)
        similarities = torch.matmul(inputs, inputs.t()) * self.scale

        targets = targets.view(-1,1)
        mask = torch.eq(targets, targets.T).float().cuda()
        mask_self = torch.eye(targets.size(0)).float().cuda()
        mask_pos = mask - mask_self
        mask_neg = 1 - mask

        # compute log_prob
        exp_logits = torch.exp(similarities) * (1 - mask_self)
        # log_prob = similarities - torch.log(exp_logits.sum(1, keepdim=True))
        log_sum_exp_pos_and_all_neg = torch.log((exp_logits * mask_neg).sum(1, keepdim=True) + exp_logits)
        log_prob = similarities - log_sum_exp_pos_and_all_neg

        # compute mean of log-likelihood over positive
        loss = (mask_pos * log_prob).sum(1) / mask_pos.sum(1)

        loss = - loss.mean()

        return loss