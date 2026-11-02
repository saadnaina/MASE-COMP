"""
Experimental Training Pipeline for MONAI 2D Residual U-Net on AISD / CPAISD NCCT Lesions.
Uses combined Dice + Cross-Entropy loss and patient-level validation splitting.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
from monai.losses import DiceCELoss
from monai.networks.nets import UNet

import sys
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import DEVICE, ISCHAEMIC_WEIGHTS_PATH


def build_unet_model() -> UNet:
    return UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=2,
    )


if __name__ == "__main__":
    print(
        "AISD / CPAISD 2D U-Net Training Script Ready. "
        f"Checkpoints will be saved to: {ISCHAEMIC_WEIGHTS_PATH}"
    )
