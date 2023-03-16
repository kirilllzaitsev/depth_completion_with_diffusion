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
        check_tensor_shapes_match(pred, target)
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


class GlobalSmoothnessLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, depth):
        """
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


class PhotometricConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        target,
        recon,
        mask=None,
        w_co=0.15,
        w_st=0.95,
    ):
        """
        Photommetric loss between the original (adjacent) frame and the one reconstructed, i.e., obtained by warping the target frame.
        From the paper:
        For KITTI, we choose wph = 1, wco = 0.15, wst = 0.95, wsz = 0.6, and wsm = 0.04

        Args:
            target: I_t frame
            recon: I_tau frame, tau is the index of an adjacent frame
            mask: mask out pixels present in the sparse depth map
        """

        check_tensor_shapes_match(target, recon)
        ssim = SSIM()
        ssim_compound = w_st * (1 - ssim(recon, target))
        l1_compound = w_co * MaskedL1Loss()(recon, target, mask)
        self.loss = (l1_compound + ssim_compound).mean()
        return self.loss


class SparseDepthConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, src, tgt, w):
        """
        Computes the sparse depth consistency loss

        Arg(s):
            src : torch.Tensor[float32]
                N x 1 x H x W source depth
            tgt : torch.Tensor[float32]
                N x 1 x H x W target depth
            w : torch.Tensor[float32]
                N x 1 x H x W weights
        Returns:
            torch.Tensor[float32] : mean absolute difference between source and target depth
        """

        delta = torch.abs(tgt - src)
        loss = torch.sum(w * delta, dim=[1, 2, 3])

        return torch.mean(loss / torch.sum(w, dim=[1, 2, 3]))


class TotalLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.photometric_loss = PhotometricConsistencyLoss()
        self.smoothness_loss = LocalSmoothnessLoss()
        self.sparse_depth_loss = SparseDepthConsistencyLoss()

    def forward(
        self,
        target,
        recon,
        depth,
        sparse_depth,
        mask,
        w_ph=1,
        w_sz=0.6,
        w_sm=0.04,
    ):
        """
        Args:
            target: I_t frame
            recon: I_tau frame, tau is the index of an adjacent frame
            depth: predicted dense depth map
            sparse_depth: sparse (input) depth map
            mask: mask that has zeros at non-zero pixels of the sparse depth map
        """
        self.loss = (
            w_ph * self.photometric_loss(target, recon, mask)
            + w_sm * self.smoothness_loss(depth)
            + w_sz * self.sparse_depth_loss(depth, sparse_depth, mask)
        )
        return self.loss


class LocalSmoothnessLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predict, image):
        """
        Computes the local smoothness loss

        Arg(s):
            predict : torch.Tensor[float32]
                N x 1 x H x W predictions
            image : torch.Tensor[float32]
                N x 3 x H x W RGB image
        Returns:
            torch.Tensor[float32] : mean SSIM distance between source and target images
        """

        predict_dy, predict_dx = gradient_yx(predict)
        image_dy, image_dx = gradient_yx(image)

        # Create edge awareness weights
        weights_x = torch.exp(-torch.mean(torch.abs(image_dx), dim=1, keepdim=True))
        weights_y = torch.exp(-torch.mean(torch.abs(image_dy), dim=1, keepdim=True))

        smoothness_x = torch.mean(weights_x * torch.abs(predict_dx))
        smoothness_y = torch.mean(weights_y * torch.abs(predict_dy))

        return smoothness_x + smoothness_y


def gradient_yx(T):
    """
    Computes gradients in the y and x directions

    Arg(s):
        T : torch.Tensor[float32]
            N x C x H x W tensor
    Returns:
        torch.Tensor[float32] : gradients in y direction
        torch.Tensor[float32] : gradients in x direction
    """

    dx = T[:, :, :, :-1] - T[:, :, :, 1:]
    dy = T[:, :, :-1, :] - T[:, :, 1:, :]
    return dy, dx


class SSIM(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        """
        Args:
            x: predicted image
            y: target image
        """
        check_tensor_shapes_match(x, y)
        return self.ssim(x, y)

    def ssim(self, x, y):
        """
        Computes Structural Similarity Index distance between two images

        Arg(s):
            x : torch.Tensor[float32]
                N x 3 x H x W RGB image
            y : torch.Tensor[float32]
                N x 3 x H x W RGB image
        Returns:
            torch.Tensor[float32] : SSIM distance between two images
        """

        C1 = 0.01**2
        C2 = 0.03**2

        mu_x = torch.nn.AvgPool2d(3, 1)(x)
        mu_y = torch.nn.AvgPool2d(3, 1)(y)
        mu_xy = mu_x * mu_y
        mu_xx = mu_x**2
        mu_yy = mu_y**2

        sigma_x = torch.nn.AvgPool2d(3, 1)(x**2) - mu_xx
        sigma_y = torch.nn.AvgPool2d(3, 1)(y**2) - mu_yy
        sigma_xy = torch.nn.AvgPool2d(3, 1)(x * y) - mu_xy

        numer = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
        denom = (mu_xx + mu_yy + C1) * (sigma_x + sigma_y + C2)
        score = numer / denom

        return torch.clamp((1.0 - score) / 2.0, 0.0, 1.0)


if __name__ == "__main__":
    ytrue = torch.tensor([1.0])
    yhat = torch.tensor([2.0])
    input_img = torch.rand(1, 3, 256, 256)
    reconstructed_img = torch.rand(1, 3, 256, 256)
    # loss = custom_mse_loss(ytrue, yhat)
    photommetric_loss = PhotometricLoss()
    loss = photommetric_loss(input_img, reconstructed_img)
