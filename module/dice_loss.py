import torch
import torch.nn as nn
from torch.autograd import Function, Variable


class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)


class DiceCoefficientLF(nn.Module):

    def __init__(self, device):
        super(DiceCoefficientLF, self).__init__()
        self.device = device

    def forward(self, y_true, y_pred):
        _smooth = torch.tensor([0.0001]).to(self.device)
        return 1.0 - (2.0 * torch.sum(y_true * y_pred)) /(torch.sum(y_true) + torch.sum(y_pred) + _smooth)


class DiceCoefficientLF_rec(nn.Module):

    def __init__(self, device):
        super(DiceCoefficientLF_rec, self).__init__()
        self.device = device

    def forward(self, y_true, y_pred):
        _smooth = torch.tensor([0.0001]).to(self.device)
        return (2.0 * torch.sum(y_true * y_pred)) /(torch.sum(y_true) + torch.sum(y_pred) + _smooth)


class MSELF(nn.Module):

    def __init__(self, device):
        super(MSELF, self).__init__()
        self.device = device

    def forward(self, y_true, y_pred):
        _smooth = torch.tensor([0.0001]).to(self.device)
        print(torch.max(y_true))
        print(torch.min(y_true))

        print(torch.max(y_pred))
        print(torch.min(y_pred))
        return torch.sum(y_pred - y_true)
