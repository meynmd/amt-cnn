import torch
from torch import nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.):
        super(FocalLoss, self).__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, z, y):
        batch_size, w, h = z.shape
        p = torch.sigmoid(z)
        p_t = p*y + (1. - p)*(1 - y)
        # w = self.alpha*y + (1. - self.alpha)*(1 - y)
        w = w*torch.pow(1. - p_t, self.gamma)
        return F.binary_cross_entropy_with_logits(z, y, w)
