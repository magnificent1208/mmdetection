import torch
import torch.nn as nn

from ..builder import LOSSES


def _neg_loss(pred, gt):
    ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w) => (batch, c, num_points)
      gt_regr (batch x c x h x w)
    '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()
    neg_weights = torch.pow(1 - gt, 4)
    loss = 0
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds
    num_pos  = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    count = int(num_pos.cpu().detach())
    if  count == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


@LOSSES.register_module()
class CenterFocalLoss(nn.Module):
    '''nn.Module warpper for focal loss'''
    def __init__(self, loss_weight=1.0):
        super(CenterFocalLoss, self).__init__()
        self.neg_loss = _neg_loss
        self.loss_weight = loss_weight

    def forward(self, out, target):
        return self.neg_loss(out, target) * self.loss_weight


def _reg_loss(regs, gt_regs, mask):
    mask = mask[:, :, None].expand_as(gt_regs).float()
    loss = sum(F.l1_loss(r * mask, gt_regs * mask, reduction='sum') /
               (mask.sum() + 1e-4) for r in regs)
    return loss / len(regs)


@LOSSES.register_module()
class CenterL1Loss(nn.Module):
    '''nn.Module warpper for focal loss'''
    def __init__(self, loss_weight=1.0):
        super(CenterL1Loss, self).__init__()
        self.reg_loss = _reg_loss
        self.loss_weight = loss_weight

    def forward(self, out, target):
        return self.reg_loss(out, target) * self.loss_weight
