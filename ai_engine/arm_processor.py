"""
MODULE THREE: Arm Weakness & Pronator Drift Assessment Engine.
Combines interactive clinical instructions with optional MediaPipe Pose landmark tracking
to compare left vs. right wrist vertical elevation relative to shoulder landmarks.
"""
import io
from typing import Any, Dict

import cv2
import numpy as np
from PIL import Image

try:
    import mediapipe as mp
    HAS_MP_POSE = hasattr(mp, "solutions") and hasattr(mp.solutions, "pose")
except Exception:
    HAS_MP_POSE = False


class ArmWeaknessProcessor:
    """Evaluates bilateral arm elevation symmetry from a pose frame or manual observation."""

    def analyse_pose_image_bytes(self, image_bytes: bytes) -> Dict[str, Any]:
        try:
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            rgb = np.array(pil_img, dtype=np.uint8)
        except Exception as exc:
            return {
                "status": "INVALID_IMAGE",
                "drift_flagged": False,
                "wrist_height_diff_norm": None,
                "message": f"Invalid image: {exc}",
                "annotated_rgb": None,
            }

        if not HAS_MP_POSE:
            return {
                "status": "POSE_MODEL_UNAVAILABLE_USE_MANUAL",
                "drift_flagged": False,
                "wrist_height_diff_norm": None,
                "message": "MediaPipe Pose Landmarker unavailable in current runtime; record manual arm observation.",
                "annotated_rgb": rgb,
            }

        h, w, _ = rgb.shape
        annotated = rgb.copy()
        with mp.solutions.pose.Pose(
            static_image_mode=True, min_detection_confidence=0.5
        ) as pose:
            res = pose.process(rgb)

        if not res.pose_landmarks:
            return {
                "status": "NO_POSE_DETECTED",
                "drift_flagged": False,
                "wrist_height_diff_norm": None,
                "message": "Upper body and wrists not clearly visible in frame. Use manual observation.",
                "annotated_rgb": annotated,
            }

        lm = res.pose_landmarks.landmark
        # Landmarks: 11=left_shoulder, 12=right_shoulder, 15=left_wrist, 16=right_wrist
        l_sh = np.array([lm[11].x * w, lm[11].y * h])
        r_sh = np.array([lm[12].x * w, lm[12].y * h])
        l_wr = np.array([lm[15].x * w, lm[15].y * h])
        r_wr = np.array([lm[16].x * w, lm[16].y * h])

        shoulder_width = float(np.linalg.norm(r_sh - l_sh)) + 1e-6
        # Relative vertical drop between wrists normalized by shoulder span
        wrist_vertical_diff = abs(float(l_wr[1] - r_wr[1])) / shoulder_width
        drift_flagged = bool(wrist_vertical_diff >= 0.18)

        for pt, col in [(l_sh, (0, 180, 216)), (r_sh, (0, 180, 216)), (l_wr, (239, 68, 68)), (r_wr, (239, 68, 68))]:
            cv2.circle(annotated, (int(pt[0]), int(pt[1])), 6, col, -1)
        cv2.line(annotated, (int(l_sh[0]), int(l_sh[1])), (int(l_wr[0]), int(l_wr[1])), (0, 212, 255), 3)
        cv2.line(annotated, (int(r_sh[0]), int(r_sh[1])), (int(r_wr[0]), int(r_wr[1])), (0, 212, 255), 3)

        lower_side = "Left Arm" if l_wr[1] > r_wr[1] else "Right Arm"
        return {
            "status": "DRIFT_FLAGGED" if drift_flagged else "SYMMETRIC_ARM_ELEVATION",
            "drift_flagged": drift_flagged,
            "wrist_height_diff_norm": round(wrist_vertical_diff, 4),
            "lower_arm_observed": lower_side if drift_flagged else "None",
            "message": (
                f"Experimental pose analysis indicates {lower_side} is lower (normalized delta {wrist_vertical_diff:.2f}). "
                "Webcam pose estimation cannot clinically diagnose paresis; verify with manual 10-second hold test."
                if drift_flagged
                else f"Bilateral wrist elevation appears symmetric (delta {wrist_vertical_diff:.2f})."
            ),
            "annotated_rgb": annotated,
        }
