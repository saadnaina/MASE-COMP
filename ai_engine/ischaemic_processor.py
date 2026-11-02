"""
MODULE ONE (Part B): Experimental Ischaemic Stroke Lesion Segmentation Engine.
Uses a MONAI 2D Residual U-Net architecture designed for AISD / CPAISD NCCT datasets.
Includes Dice and IoU evaluation metrics.
"""
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from monai.networks.nets import UNet

from config import DEVICE, ISCHAEMIC_WEIGHTS_PATH
from preprocessing.dicom_utils import apply_hu_window


def compute_dice_and_iou(pred_mask: np.ndarray, true_mask: np.ndarray) -> Tuple[float, float]:
    """Calculates Dice similarity coefficient and Intersection over Union (IoU)."""
    pred_b = (pred_mask > 0).astype(np.uint8)
    true_b = (true_mask > 0).astype(np.uint8)
    intersection = float(np.sum(pred_b * true_b))
    pred_sum = float(np.sum(pred_b))
    true_sum = float(np.sum(true_b))
    union = pred_sum + true_sum - intersection

    dice = (2.0 * intersection + 1e-6) / (pred_sum + true_sum + 1e-6)
    iou = (intersection + 1e-6) / (union + 1e-6)
    return round(float(dice), 4), round(float(iou), 4)


class IschaemicSegmentationProcessor:
    """
    Experimental MONAI 2D U-Net for segmenting suspected ischaemic lesions on NCCT.
    Clearly reports whether trained AISD/CPAISD checkpoint weights are present.
    """

    def __init__(self, weights_path: Path = ISCHAEMIC_WEIGHTS_PATH) -> None:
        self.weights_path = Path(weights_path)
        self.device = torch.device(DEVICE)
        self.model: nn.Module = UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64, 128),
            strides=(2, 2, 2),
            num_res_units=2,
        )
        self.weights_loaded = False
        self.status_label = "EXPERIMENTAL_UNTRAINED_ARCHITECTURE"
        self._load_weights()
        self.model.to(self.device)
        self.model.eval()

    def _load_weights(self) -> None:
        if self.weights_path.exists():
            try:
                state = torch.load(str(self.weights_path), map_location=self.device)
                self.model.load_state_dict(
                    state.get("model_state_dict", state)
                )
                self.weights_loaded = True
                self.status_label = "TRAINED_AISD_UNET_LOADED"
            except Exception as exc:
                self.status_label = f"CHECKPOINT_LOAD_ERROR ({exc})"
        else:
            torch.manual_seed(2026)

    def segment_slice(self, hu_slice: np.ndarray) -> Dict[str, Any]:
        """Runs 2D U-Net segmentation on a brain-windowed CT slice and creates a mask overlay."""
        orig_h, orig_w = hu_slice.shape
        brain_norm = apply_hu_window(hu_slice, 80.0, 40.0)
        resized = cv2.resize(brain_norm, (256, 256), interpolation=cv2.INTER_LINEAR)

        tensor_in = (
            torch.from_numpy(resized).unsqueeze(0).unsqueeze(0).to(self.device)
        )
        with torch.no_grad():
            logits = self.model(tensor_in)
            prob_map = torch.sigmoid(logits).squeeze().cpu().numpy()

        prob_orig = cv2.resize(prob_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        # Restrict mask inside cranial parenchyma (15 <= HU <= 45)
        parenchyma_mask = (hu_slice >= 12.0) & (hu_slice <= 45.0)
        binary_mask = ((prob_orig >= 0.55) & parenchyma_mask).astype(np.uint8)

        parenchyma_pixels = max(int(np.sum(parenchyma_mask)), 1)
        lesion_burden_pct = round(
            float(np.sum(binary_mask)) / float(parenchyma_pixels) * 100.0, 2
        )

        brain_u8 = np.uint8(255 * brain_norm)
        brain_rgb = cv2.cvtColor(brain_u8, cv2.COLOR_GRAY2RGB)
        seg_overlay = brain_rgb.copy()
        # Highlight suspected region in cyan-teal [0, 220, 255]
        seg_overlay[binary_mask == 1] = [0, 212, 255]
        blended = cv2.addWeighted(brain_rgb, 0.68, seg_overlay, 0.32, 0)

        return {
            "binary_mask": binary_mask,
            "probability_map": prob_orig,
            "lesion_burden_pct": lesion_burden_pct,
            "segmentation_overlay_rgb": blended,
            "weights_loaded": self.weights_loaded,
            "status_label": self.status_label,
        }
