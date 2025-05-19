import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .lpips import LPIPS


class RGBLpipsLoss(nn.Module):
    """
    Loss module that combines RGB reconstruction loss (MSE) and optional perceptual loss (LPIPS).

    Args:
        perceptual_weight (float): Weight for the perceptual loss.
        use_perceptual_loss (bool): Flag to determine whether perceptual loss is used.
        enable_perceptual_loss (bool): Initial state of perceptual loss usage.
    """

    def __init__(
        self,
        perceptual_weight=0.5,
        use_perceptual_loss=True,
        enable_perceptual_loss=True,
    ):
        super().__init__()

        # Initialize the perceptual loss (LPIPS) if enabled
        if enable_perceptual_loss:
            self.perceptual_loss = LPIPS().eval()
            for param in self.perceptual_loss.parameters():
                param.requires_grad = False

        self.perceptual_weight = perceptual_weight
        self.use_perceptual_loss = use_perceptual_loss
        self.enable_perceptual_loss = enable_perceptual_loss

    def set_perceptual_loss(self, enable=True):
        """
        Enable or disable the perceptual loss.

        Args:
            enable (bool): Whether to enable perceptual loss.
        """
        self.enable_perceptual_loss = enable and self.use_perceptual_loss

    def forward(self, rgb, targets):
        """
        Compute the RGB reconstruction loss and (optionally) perceptual loss.

        Args:
            rgb (Tensor): Predicted RGB values with shape (..., H, W, C).
            targets (Tensor): Ground truth RGB values with shape (..., H, W, C).

        Returns:
            dict: Dictionary containing 'rgb_loss' and optionally 'perceptual_loss'.
        """
        # Rearrange input tensors to the format (batch, channels, height, width)
        rgb = rearrange(rgb, "... h w c -> (...) c h w")
        targets = rearrange(targets, "... h w c -> (...) c h w")
        rgb_loss = F.mse_loss(rgb, targets)
        loss_dict = {"rgb_loss": rgb_loss}
        if self.enable_perceptual_loss:
            perceptual_loss = self.perceptual_weight * self.perceptual_loss(rgb, targets)
            loss_dict["perceptual_loss"] = perceptual_loss.mean()

        return loss_dict
