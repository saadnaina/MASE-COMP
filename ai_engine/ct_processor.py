"""
MODULE ONE: AI Brain CT Haemorrhage Classifier & Grad-CAM Explainability Engine.
Uses DenseNet121 for 5-subtype + overall Intracranial Haemorrhage classification
and generates genuine gradient-weighted class activation heatmaps from the last conv block.
"""
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet121

from config import DEVICE, HAEMORRHAGE_WEIGHTS_PATH, ICH_SUBTYPES
from preprocessing.dicom_utils import apply_hu_window, build_multiwindow_tensor


class DenseNet121HaemorrhageModel(nn.Module):
    """
    Multi-label DenseNet121 classifier outputting 6 logits:
    [intraparenchymal, intraventricular, subarachnoid, subdural, epidural, any_haemorrhage].
    Includes forward/backward hooks on features.denseblock4 for authentic Grad-CAM extraction.
    """

    def __init__(self, num_classes: int = 6) -> None:
        super().__init__()
        self.backbone = densenet121(weights=None)
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Linear(in_features, num_classes)
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None

    def _save_gradient(self, grad: torch.Tensor) -> None:
        self.gradients = grad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.features(x)
        self.activations = features
        if features.requires_grad:
            features.register_hook(self._save_gradient)
        out = F.relu(features, inplace=False)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        logits = self.backbone.classifier(out)
        return logits


class CTHaemorrhageProcessor:
    """Manages CT haemorrhage classification, uncertainty estimation, and Grad-CAM generation."""

    def __init__(self, weights_path: Path = HAEMORRHAGE_WEIGHTS_PATH) -> None:
        self.weights_path = Path(weights_path)
        self.device = torch.device(DEVICE)
        self.model = DenseNet121HaemorrhageModel(num_classes=len(ICH_SUBTYPES))
        self.weights_loaded = False
        self.model_status = "UNTRAINED_ARCHITECTURE_BASELINE"
        self._load_weights_if_available()
        self.model.to(self.device)
        self.model.eval()

    def _load_weights_if_available(self) -> None:
        if self.weights_path.exists():
            try:
                state = torch.load(str(self.weights_path), map_location=self.device)
                if "model_state_dict" in state:
                    self.model.load_state_dict(state["model_state_dict"])
                else:
                    self.model.load_state_dict(state)
                self.weights_loaded = True
                self.model_status = "TRAINED_RSNA_CHECKPOINT_LOADED"
            except Exception as exc:
                self.weights_loaded = False
                self.model_status = f"CHECKPOINT_LOAD_ERROR ({exc})"
        else:
            # Initialize deterministic weights so forward/backward Grad-CAM is reproducible
            torch.manual_seed(2026)
            for m in self.model.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                elif isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, -0.5)

    def analyse_slice(
        self, hu_slice: np.ndarray, target_class_idx: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Runs a forward and backward pass on a 2D HU CT slice to produce:
        - Subtype probability dictionary
        - Binary predictive entropy (uncertainty metric)
        - Genuine Grad-CAM heatmap and RGB overlay
        """
        multi_win = build_multiwindow_tensor(hu_slice, target_size=(224, 224))
        input_tensor = (
            torch.from_numpy(multi_win).unsqueeze(0).to(self.device).requires_grad_(True)
        )

        self.model.zero_grad()
        logits = self.model(input_tensor)
        probs = torch.sigmoid(logits).squeeze(0)

        prob_dict: Dict[str, float] = {
            subtype: round(float(probs[idx].item()), 4)
            for idx, subtype in enumerate(ICH_SUBTYPES)
        }
        overall_prob = prob_dict["any_haemorrhage"]

        # Select target class for Grad-CAM (default to highest subtype or any_haemorrhage)
        if target_class_idx is None:
            target_class_idx = int(torch.argmax(probs).item())

        score = logits[0, target_class_idx]
        score.backward(retain_graph=False)

        heatmap, overlay_rgb, brain_rgb = self._compute_gradcam_visuals(
            hu_slice=hu_slice,
            activations=self.model.activations,
            gradients=self.model.gradients,
        )

        # Compute predictive binary entropy across subtypes as an uncertainty measure
        p_np = probs.detach().cpu().numpy()
        p_clipped = np.clip(p_np, 1e-6, 1.0 - 1e-6)
        entropy = - (
            p_clipped * np.log2(p_clipped) + (1.0 - p_clipped) * np.log2(1.0 - p_clipped)
        )
        mean_uncertainty_entropy = round(float(np.mean(entropy)), 4)

        haemorrhage_detected = bool(overall_prob >= 0.50 or max(prob_dict.values()) >= 0.50)

        return {
            "subtype_probabilities": prob_dict,
            "overall_haemorrhage_prob": overall_prob,
            "haemorrhage_detected": haemorrhage_detected,
            "uncertainty_entropy": mean_uncertainty_entropy,
            "target_gradcam_subtype": ICH_SUBTYPES[target_class_idx],
            "model_weights_status": self.model_status,
            "weights_loaded": self.weights_loaded,
            "brain_window_rgb": brain_rgb,
            "gradcam_heatmap_rgb": heatmap,
            "gradcam_overlay_rgb": overlay_rgb,
        }

    def _compute_gradcam_visuals(
        self,
        hu_slice: np.ndarray,
        activations: Optional[torch.Tensor],
        gradients: Optional[torch.Tensor],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generates the original brain-window RGB, jet heatmap RGB, and blended overlay RGB."""
        h, w = hu_slice.shape
        brain_norm = apply_hu_window(hu_slice, 80.0, 40.0)
        brain_u8 = np.uint8(255 * brain_norm)
        brain_rgb = cv2.cvtColor(brain_u8, cv2.COLOR_GRAY2RGB)

        if activations is None or gradients is None:
            zeros = np.zeros_like(brain_rgb)
            return zeros, brain_rgb, brain_rgb

        pooled_grads = torch.mean(gradients, dim=[0, 2, 3])
        act = activations.detach().squeeze(0).clone()
        for i in range(act.shape[0]):
            act[i, :, :] *= pooled_grads[i]

        cam = torch.mean(act, dim=0).cpu().numpy()
        cam = np.maximum(cam, 0.0)
        if np.max(cam) > 1e-8:
            cam = cam / np.max(cam)

        cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC)
        cam_u8 = np.uint8(255 * cam_resized)
        heatmap_bgr = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

        overlay_rgb = cv2.addWeighted(brain_rgb, 0.62, heatmap_rgb, 0.38, 0)
        return heatmap_rgb, overlay_rgb, brain_rgb
