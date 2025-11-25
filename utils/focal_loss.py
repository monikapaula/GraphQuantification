import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    input: log_probs
    crossEntropy = nn_loss(log_softmax(x)
    Focal Loss for addressing class imbalance.
    Reference: https://arxiv.org/abs/1708.02002
    """

    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, log_probs, targets):
        ce_loss = F.nll_loss(log_probs, targets, reduction='none')
        prob_t = torch.exp(-ce_loss)
        focal_loss = (1-prob_t)** self.gamma * ce_loss

        if self.alpha is not None:
            if self.alpha.device != log_probs.device:
                self.alpha = self.alpha.to(log_probs.device)

            alpha = self.alpha.gather(0, targets.view(-1))

            focal_loss = alpha * focal_loss


        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss