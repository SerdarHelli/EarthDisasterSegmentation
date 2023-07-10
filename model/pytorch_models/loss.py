import torch

eps=1e-6

def soft_dice_loss(outputs, targets, per_image=False):
    batch_size = outputs.size()[0]
    if not per_image:
        batch_size = 1


    dice_target = targets.contiguous().view(batch_size, -1).float()
    dice_output = outputs.contiguous().view(batch_size, -1)
    intersection = torch.sum(dice_output * dice_target)
    union = torch.sum(dice_output) + torch.sum(dice_target) + eps
    loss = (1 - (2 * intersection + eps) / union).mean()
    return loss

def dice_torch(outputs, targets, empty_score=1.0):
    dice_output = outputs
    dice_target = targets
    intersection = torch.sum(torch.logical_and(dice_output, dice_target), dim=0)
    union = torch.sum(dice_output) + torch.sum(dice_target, dim=0) + eps
    dice_value = ((2 * intersection + eps) / union).mean()
    return 1-dice_value


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, per_image=False):
        super().__init__()
        self.size_average = size_average
        self.register_buffer('weight', weight)
        self.per_image = per_image

    def forward(self, input, target):
        return soft_dice_loss(input, target)

class FocalLoss2d(nn.Module):
    def __init__(self, gamma=2, ignore_index=255):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, outputs, targets):
        outputs = outputs.contiguous()
        targets = targets.contiguous()
        # eps = 1e-8
        non_ignored = targets.view(-1) != self.ignore_index
        targets = targets.view(-1)[non_ignored].float()
        outputs = outputs.contiguous().view(-1)[non_ignored]
        outputs = torch.clamp(outputs, eps, 1. - eps)
        targets = torch.clamp(targets, eps, 1. - eps)
        pt = (1 - targets) * (1 - outputs) + targets * outputs
        return (-(1. - pt) ** self.gamma * torch.log(pt)).mean()

class ComboLossDice(nn.Module):
    def __init__(self, per_image=False):
        super().__init__()
        self.weights = [1,8]
        self.dice = DiceLoss(per_image=False)
        self.focal = FocalLoss2d()
   
        self.values = {}

    def forward(self, outputs, targets):
        loss = 0
        outputs = torch.sigmoid(outputs)

        loss += self.weights[0] * self.dice(outputs, targets)
        loss += self.weights[1] * self.focal(outputs, targets)
        return loss

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



