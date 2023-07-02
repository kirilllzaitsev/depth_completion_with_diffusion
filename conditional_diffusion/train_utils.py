import torch
import torch.nn as nn
import torch.nn.functional as F


def pad_tensor(source, target):
    """
    Pad source tensor to match target tensor size

    :param source: tensor that need to get padding
    :param target: tensor of the desired shape
    :return: source tensor with shape equal to target
    """

    diff_y = target.size()[2] - source.size()[2]
    diff_x = target.size()[3] - source.size()[3]

    source = F.pad(
        source, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2]
    )
    return source


def concatenate_tensors(x1, x2):
    """
    Concatenate both tensors

    :param x1: first tensor to be concatenated
    :param x2: second tensor to be concatenated
    :return: concatenation of both tensors
    """

    x1 = pad_tensor(x1, x2)
    return torch.cat([x1, x2], dim=1)
