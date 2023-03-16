"""Covers losses in the self- completion"""

import torch
import torch.nn as nn
from utils import check_tensor_shape, check_tensor_shapes_match


class MaskedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        check_tensor_shapes_match(pred, target)
        valid_mask = (target > 0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = (diff**2).mean()
        return self.loss


class MaskedL1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, weight=None):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = diff.abs().mean()
        return self.loss


class PhotometricLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, target, recon, mask=None):
        """
        Photommetric loss between the original (adjacent) frame and the one reconstructed, i.e., obtained by warping the target frame.

        Args:
            target: I_t frame
            recon: I_tau frame, tau is the index of an adjacent frame
            mask: mask out pixels present in the sparse depth map
        """

        check_tensor_shapes_match(target, recon)
        diff = (target - recon).abs()
        diff = torch.sum(diff, dim=1)  # sum along the color channel

        # compare only pixels that are not black
        valid_mask = (torch.sum(recon, 1) > 0).float() * (
            torch.sum(target, 1) > 0
        ).float()
        if mask is not None:
            valid_mask = valid_mask * torch.squeeze(mask).float()
        valid_mask = valid_mask.bool().detach()
        if valid_mask.numel() > 0:
            diff = diff[valid_mask]
            if diff.nelement() > 0:
                self.loss = diff.mean()
            else:
                print(
                    "warning: diff.nelement()==0 in PhotometricLoss (this is expected during early stage of training, try larger batch size)."
                )
                self.loss = 0
        else:
            print("warning: 0 valid pixel in PhotometricLoss")
            self.loss = 0
        return self.loss


class SmoothnessLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, depth):
        """
        Should it be a second derivative or a first derivative?

        Args:
            depth: depth feature map
        """
        check_tensor_shape(depth, target_dim=4)

        self.loss = self.second_derivative(depth)
        return self.loss

    def second_derivative(self, x):
        horizontal = 2 * x[:, :, 1:-1, 1:-1] - x[:, :, 1:-1, :-2] - x[:, :, 1:-1, 2:]
        vertical = 2 * x[:, :, 1:-1, 1:-1] - x[:, :, :-2, 1:-1] - x[:, :, 2:, 1:-1]
        der_2nd = horizontal.abs() + vertical.abs()
        return der_2nd.mean()


class DepthLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, depth, target, mask):
        """
        Args:
            depth: predicted dense depth map
            target: sparse (input) depth map
            mask: mask that has zeros at non-zero pixels of the sparse depth map
        """
        check_tensor_shapes_match(depth, target)
        check_tensor_shapes_match(target, mask)
        diff = ((target - depth) * mask).square()
        self.loss = torch.sum(diff)

        return self.loss


if __name__ == "__main__":
    ytrue = torch.tensor([1.0])
    yhat = torch.tensor([2.0])
    input_img = torch.rand(1, 3, 256, 256)
    reconstructed_img = torch.rand(1, 3, 256, 256)
    # loss = custom_mse_loss(ytrue, yhat)
    photommetric_loss = PhotometricLoss()
    loss = photommetric_loss(input_img, reconstructed_img)
