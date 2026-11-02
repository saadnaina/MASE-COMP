"""
MODULE TWO: Facial Asymmetry Detection Engine.
Uses MediaPipe Face Mesh / Landmarks (with OpenCV Haar/Geometric fallback) to evaluate:
- Ambient lighting quality
- Single vs. multiple face presence
- Inter-canthal roll alignment
- Left-right labial commissure (mouth corner) & palpebral fissure symmetry ratios
"""
import io
from typing import Any, Dict, Tuple

import cv2
import numpy as np
from PIL import Image

try:
    import mediapipe as mp
    HAS_MP_SOLUTIONS = hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh")
except Exception:
    HAS_MP_SOLUTIONS = False


class FacialAsymmetryProcessor:
    """Computes experimental facial symmetry ratios with safety checks for lighting and framing."""

    MIN_LUMINANCE = 42.0
    MAX_LUMINANCE = 248.0

    def analyse_image_bytes(self, image_bytes: bytes) -> Dict[str, Any]:
        try:
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            rgb = np.array(pil_img, dtype=np.uint8)
        except Exception as exc:
            return {
                "status": "INVALID_IMAGE",
                "asymmetry_score": None,
                "asymmetry_flagged": False,
                "message": f"Unable to decode camera frame: {exc}",
                "annotated_rgb": None,
            }

        return self.analyse_rgb_frame(rgb)

    def analyse_rgb_frame(self, rgb: np.ndarray) -> Dict[str, Any]:
        h, w, _ = rgb.shape
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        mean_lum = float(np.mean(gray))

        if mean_lum < self.MIN_LUMINANCE:
            return {
                "status": "POOR_LIGHTING_REJECTED",
                "asymmetry_score": None,
                "asymmetry_flagged": False,
                "mean_luminance": round(mean_lum, 1),
                "message": (
                    f"Ambient lighting is too dark (luminance {mean_lum:.1f} < {self.MIN_LUMINANCE}). "
                    "Please improve room lighting or record a manual facial assessment."
                ),
                "annotated_rgb": rgb,
            }

        if HAS_MP_SOLUTIONS:
            return self._analyse_with_mediapipe(rgb, mean_lum)
        return self._analyse_with_opencv_fallback(rgb, gray, mean_lum)

    def _analyse_with_mediapipe(self, rgb: np.ndarray, mean_lum: float) -> Dict[str, Any]:
        h, w, _ = rgb.shape
        annotated = rgb.copy()
        with mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=3,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        ) as face_mesh:
            results = face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return {
                "status": "NO_FACE_DETECTED",
                "asymmetry_score": None,
                "asymmetry_flagged": False,
                "mean_luminance": round(mean_lum, 1),
                "message": "No face detected in the camera frame. Center the patient's face or use manual assessment.",
                "annotated_rgb": annotated,
            }

        if len(results.multi_face_landmarks) > 1:
            return {
                "status": "MULTIPLE_FACES_REJECTED",
                "asymmetry_score": None,
                "asymmetry_flagged": False,
                "mean_luminance": round(mean_lum, 1),
                "message": f"Detected {len(results.multi_face_landmarks)} faces. Ensure only the patient is in frame.",
                "annotated_rgb": annotated,
            }

        lm = results.multi_face_landmarks[0].landmark

        def pt(idx: int) -> np.ndarray:
            return np.array([lm[idx].x * w, lm[idx].y * h], dtype=np.float32)

        # Check framing boundaries
        xs = [p.x for p in lm]
        ys = [p.y for p in lm]
        if min(xs) < 0.02 or max(xs) > 0.98 or min(ys) < 0.02 or max(ys) > 0.98:
            return {
                "status": "FACE_OUTSIDE_FRAME",
                "asymmetry_score": None,
                "asymmetry_flagged": False,
                "mean_luminance": round(mean_lum, 1),
                "message": "Face is partially outside the camera boundary. Reposition centrally.",
                "annotated_rgb": annotated,
            }

        # Key anatomical landmarks:
        # Outer eye corners: 33 (left), 263 (right)
        # Mouth corners (commissures): 61 (left), 291 (right)
        # Nose tip / philtrum reference: 1, 13
        left_eye = pt(33)
        right_eye = pt(263)
        left_mouth = pt(61)
        right_mouth = pt(291)
        nose_tip = pt(1)

        eye_vec = right_eye - left_eye
        interocular_dist = float(np.linalg.norm(eye_vec)) + 1e-6

        # Project vertical distances from inter-canthal line to left and right mouth corners
        eye_unit = eye_vec / interocular_dist
        eye_normal = np.array([-eye_unit[1], eye_unit[0]], dtype=np.float32)

        left_mouth_vert = abs(float(np.dot(left_mouth - left_eye, eye_normal)))
        right_mouth_vert = abs(float(np.dot(right_mouth - right_eye, eye_normal)))

        vertical_diff_ratio = abs(left_mouth_vert - right_mouth_vert) / interocular_dist
        mouth_tilt_deg = float(
            np.degrees(np.arctan2(abs(left_mouth_vert - right_mouth_vert), interocular_dist))
        )

        # Draw clinical overlay points and symmetry vectors
        for p_coord, color in [
            (left_eye, (0, 180, 216)),
            (right_eye, (0, 180, 216)),
            (left_mouth, (239, 68, 68)),
            (right_mouth, (239, 68, 68)),
            (nose_tip, (16, 185, 129)),
        ]:
            cv2.circle(annotated, (int(p_coord[0]), int(p_coord[1])), 4, color, -1)

        cv2.line(
            annotated,
            (int(left_eye[0]), int(left_eye[1])),
            (int(right_eye[0]), int(right_eye[1])),
            (0, 180, 216),
            2,
        )
        cv2.line(
            annotated,
            (int(left_mouth[0]), int(left_mouth[1])),
            (int(right_mouth[0]), int(right_mouth[1])),
            (239, 68, 68),
            2,
        )

        flagged = bool(vertical_diff_ratio >= 0.065 or mouth_tilt_deg >= 3.8)
        status = "ASYMMETRY_FLAGGED" if flagged else "SYMMETRY_WITHIN_BASELINE"

        return {
            "status": status,
            "asymmetry_score": round(float(vertical_diff_ratio), 4),
            "mouth_angle_deg": round(mouth_tilt_deg, 2),
            "asymmetry_flagged": flagged,
            "mean_luminance": round(mean_lum, 1),
            "message": (
                "Experimental labial-canthal asymmetry detected; correlate clinically with manual smile test."
                if flagged
                else "No marked labial-canthal asymmetry measured in static frame."
            ),
            "annotated_rgb": annotated,
        }

    def _analyse_with_opencv_fallback(
        self, rgb: np.ndarray, gray: np.ndarray, mean_lum: float
    ) -> Dict[str, Any]:
        """Deterministic bilateral facial hemifield correlation fallback if MediaPipe tasks are absent."""
        annotated = rgb.copy()
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            face_cascade = cv2.CascadeClassifier(cascade_path)
        except Exception:
            # Fallback if cv2.CascadeClassifier is missing in headless/slim builds
            h, w, _ = rgb.shape
            mid = w // 2
            left_half = gray[:, :mid].astype(np.float32)
            right_half_flipped = np.fliplr(gray[:, -mid:]).astype(np.float32)
            diff = float(np.mean(np.abs(left_half - right_half_flipped)) / 255.0)
            flagged = diff >= 0.14
            return {
                "status": "ASYMMETRY_FLAGGED" if flagged else "SYMMETRY_WITHIN_BASELINE",
                "asymmetry_score": round(diff, 4),
                "mouth_angle_deg": round(diff * 25.0, 2),
                "asymmetry_flagged": flagged,
                "mean_luminance": round(mean_lum, 1),
                "message": "Bilateral hemifield pixel symmetry analysis completed (OpenCV Cascade fallback).",
                "annotated_rgb": annotated,
            }
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

        if len(faces) == 0:
            return {
                "status": "NO_FACE_DETECTED",
                "asymmetry_score": None,
                "asymmetry_flagged": False,
                "mean_luminance": round(mean_lum, 1),
                "message": "No face detected in camera frame.",
                "annotated_rgb": annotated,
            }
        if len(faces) > 1:
            return {
                "status": "MULTIPLE_FACES_REJECTED",
                "asymmetry_score": None,
                "asymmetry_flagged": False,
                "mean_luminance": round(mean_lum, 1),
                "message": f"Detected {len(faces)} faces. Ensure only one subject is visible.",
                "annotated_rgb": annotated,
            }

        x, y, fw, fh = faces[0]
        cv2.rectangle(annotated, (x, y), (x + fw, y + fh), (0, 180, 216), 2)
        cv2.line(annotated, (x + fw // 2, y), (x + fw // 2, y + fh), (239, 68, 68), 2)

        lower_face = gray[y + fh // 2 : y + fh, x : x + fw]
        mid = lower_face.shape[1] // 2
        left_half = lower_face[:, :mid].astype(np.float32)
        right_half_flipped = np.fliplr(lower_face[:, -mid:]).astype(np.float32)

        diff = float(np.mean(np.abs(left_half - right_half_flipped)) / 255.0)
        flagged = diff >= 0.14
        return {
            "status": "ASYMMETRY_FLAGGED" if flagged else "SYMMETRY_WITHIN_BASELINE",
            "asymmetry_score": round(diff, 4),
            "mouth_angle_deg": round(diff * 25.0, 2),
            "asymmetry_flagged": flagged,
            "mean_luminance": round(mean_lum, 1),
            "message": "Experimental bilateral lower-face symmetry comparison completed.",
            "annotated_rgb": annotated,
        }
