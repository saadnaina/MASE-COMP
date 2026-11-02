"""
DICOM and Medical Image Preprocessing Pipeline for NeuroScan AI.
Handles Hounsfield Unit (HU) calibration, clinical windowing (Brain, Subdural, Bone),
3-channel tensor normalization, slice extraction, and safe rejection of corrupt files.
"""
import io
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
from PIL import Image

from config import CT_WINDOWS


class InvalidMedicalImageError(ValueError):
    """Raised when an uploaded image is corrupt, unreadable, or clinically invalid."""


def apply_hu_window(hu_array: np.ndarray, window_width: float, window_level: float) -> np.ndarray:
    """Applies radiological Hounsfield Unit windowing and scales to [0.0, 1.0]."""
    low = window_level - (window_width / 2.0)
    high = window_level + (window_width / 2.0)
    windowed = np.clip(hu_array, low, high)
    windowed = (windowed - low) / max(high - low, 1e-6)
    return windowed.astype(np.float32)


def build_multiwindow_tensor(hu_slice: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Constructs a 3-channel (3, H, W) float32 array from a 2D HU slice using:
    Channel 0: Brain Window (W=80, L=40)
    Channel 1: Subdural Window (W=200, L=80)
    Channel 2: Bone/Soft Tissue Window (W=380, L=40)
    """
    if hu_slice.ndim != 2:
        raise InvalidMedicalImageError(f"Expected 2D slice, got shape {hu_slice.shape}")

    c_brain = apply_hu_window(hu_slice, CT_WINDOWS["brain"]["W"], CT_WINDOWS["brain"]["L"])
    c_subdural = apply_hu_window(hu_slice, CT_WINDOWS["subdural"]["W"], CT_WINDOWS["subdural"]["L"])
    c_bone = apply_hu_window(hu_slice, CT_WINDOWS["bone_soft"]["W"], CT_WINDOWS["bone_soft"]["L"])

    resized = [
        cv2.resize(ch, target_size, interpolation=cv2.INTER_LINEAR)
        for ch in (c_brain, c_subdural, c_bone)
    ]
    stacked = np.stack(resized, axis=0).astype(np.float32)
    return stacked


def _extract_hu_from_dicom(ds: Dataset) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Converts raw DICOM pixel_array into calibrated Hounsfield Units."""
    try:
        pixel_array = ds.pixel_array.astype(np.float32)
    except Exception as exc:
        raise InvalidMedicalImageError(f"Unable to decode DICOM pixel data: {exc}") from exc

    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    hu_image = pixel_array * slope + intercept

    metadata = {
        "Modality": str(getattr(ds, "Modality", "CT")),
        "SliceThickness": str(getattr(ds, "SliceThickness", "Unknown")),
        "KVP": str(getattr(ds, "KVP", "Unknown")),
        "RescaleSlope": slope,
        "RescaleIntercept": intercept,
        "Rows": int(getattr(ds, "Rows", hu_image.shape[-2])),
        "Columns": int(getattr(ds, "Columns", hu_image.shape[-1])),
    }
    return hu_image, metadata


def load_ct_slice_or_volume(
    uploaded_files: List[Tuple[str, bytes]]
) -> Dict[str, Any]:
    """
    Loads one or multiple DICOM (.dcm) or standard raster (.png, .jpg) slices into
    a unified volumetric list of 2D slices for slice navigation and inference.
    """
    if not uploaded_files:
        raise InvalidMedicalImageError("No image files provided.")

    slices_hu: List[np.ndarray] = []
    slice_names: List[str] = []
    modality_format = "UNKNOWN"
    metadata: Dict[str, Any] = {}

    for filename, raw_bytes in uploaded_files:
        if not raw_bytes or len(raw_bytes) < 64:
            raise InvalidMedicalImageError(f"File '{filename}' is empty or truncated.")

        lower_name = filename.lower()
        if lower_name.endswith(".dcm") or lower_name.endswith(".dicom"):
            try:
                ds = pydicom.dcmread(io.BytesIO(raw_bytes), force=True)
                hu_arr, meta = _extract_hu_from_dicom(ds)
                modality_format = "DICOM"
                metadata = meta
            except Exception as exc:
                raise InvalidMedicalImageError(
                    f"Corrupt or unsupported DICOM file '{filename}': {exc}"
                ) from exc

            if hu_arr.ndim == 3:
                for idx in range(hu_arr.shape[0]):
                    slices_hu.append(hu_arr[idx])
                    slice_names.append(f"{filename} [Slice {idx + 1}]")
            elif hu_arr.ndim == 2:
                slices_hu.append(hu_arr)
                slice_names.append(filename)
            else:
                raise InvalidMedicalImageError(f"Unsupported DICOM dimensions: {hu_arr.shape}")

        elif lower_name.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
            try:
                pil_img = Image.open(io.BytesIO(raw_bytes)).convert("L")
                gray = np.array(pil_img, dtype=np.float32)
            except Exception as exc:
                raise InvalidMedicalImageError(
                    f"Invalid raster image file '{filename}': {exc}"
                ) from exc

            if gray.shape[0] < 32 or gray.shape[1] < 32:
                raise InvalidMedicalImageError(
                    f"Image '{filename}' resolution ({gray.shape}) is too small for CT analysis."
                )

            # Map [0, 255] grayscale image to approximate brain HU range [0, 80] for compatibility
            pseudo_hu = (gray / 255.0) * 80.0
            slices_hu.append(pseudo_hu)
            slice_names.append(filename)
            modality_format = "RASTER_GRAYSCALE"
            metadata = {
                "Modality": "Raster Image (Pseudo-HU Approximated)",
                "Rows": int(gray.shape[0]),
                "Columns": int(gray.shape[1]),
                "Note": "Standard image uploaded; windowing approximated from 8-bit grayscale.",
            }
        else:
            raise InvalidMedicalImageError(
                f"Unsupported file extension for '{filename}'. Allowed: .dcm, .png, .jpg, .jpeg"
            )

    # Assess basic image quality on middle slice
    mid_slice = slices_hu[len(slices_hu) // 2]
    std_val = float(np.std(mid_slice))
    quality_status = "VALID" if std_val > 2.5 else "LOW_CONTRAST_UNRELIABLE"

    return {
        "slices_hu": slices_hu,
        "slice_names": slice_names,
        "num_slices": len(slices_hu),
        "modality_format": modality_format,
        "image_quality_status": quality_status,
        "metadata": metadata,
    }


def generate_synthetic_demo_dicom(
    output_path: Path, scenario: str = "haemorrhage_positive"
) -> Path:
    """
    Generates a valid synthetic non-contrast Brain CT DICOM phantom in calibrated Hounsfield Units
    (Skull ~ +900 HU, Brain Parenchyma ~ +35 HU, CSF Ventricles ~ +8 HU, Acute Haemorrhage ~ +72 HU,
    or Early Ischaemic Hypodensity ~ +18 HU) so competition demonstrations work offline immediately.
    """
    size = 256
    yy, xx = np.mgrid[:size, :size]
    center = size // 2
    norm_x = (xx - center) / (size * 0.40)
    norm_y = (yy - center) / (size * 0.45)
    ellipse = norm_x**2 + norm_y**2

    # Air background (-1000 HU)
    hu = np.full((size, size), -1000.0, dtype=np.float32)

    # Cranial bone ring (+850 HU)
    skull_mask = (ellipse <= 1.0) & (ellipse >= 0.84)
    hu[skull_mask] = 850.0

    # Brain parenchyma (+36 HU with realistic Gaussian texture)
    rng = np.random.default_rng(42 if scenario == "haemorrhage_positive" else 99)
    brain_mask = ellipse < 0.84
    hu[brain_mask] = 36.0 + rng.normal(0.0, 3.2, size=(size, size))[brain_mask]

    # Lateral ventricles (+10 HU CSF)
    vent_left = ((xx - (center - 22)) / 12.0) ** 2 + ((yy - center) / 34.0) ** 2 <= 1.0
    vent_right = ((xx - (center + 22)) / 12.0) ** 2 + ((yy - center) / 34.0) ** 2 <= 1.0
    hu[vent_left | vent_right] = 10.0

    if scenario == "haemorrhage_positive":
        # Hyperdense Intraparenchymal Haemorrhage focus (+74 HU) in left hemisphere
        ich_region = ((xx - (center - 46)) / 22.0) ** 2 + ((yy - (center + 10)) / 28.0) ** 2 <= 1.0
        hu[ich_region] = 74.0 + rng.normal(0.0, 2.5, size=(size, size))[ich_region]
    elif scenario == "ischaemic_hypodense":
        # Subtle hypodense MCA territory lesion (+21 HU) in right hemisphere
        isc_region = ((xx - (center + 48)) / 26.0) ** 2 + ((yy - (center - 8)) / 32.0) ** 2 <= 1.0
        hu[isc_region] = 21.0 + rng.normal(0.0, 2.0, size=(size, size))[isc_region]

    # Shift by +1024 to store as unsigned uint16 with RescaleIntercept = -1024
    stored_pixels = np.clip(hu + 1024.0, 0, 4095).astype(np.uint16)

    file_meta = pydicom.dataset.FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(str(output_path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.Modality = "CT"
    ds.PatientName = f"FICTIONAL^{scenario.upper()}"
    ds.PatientID = f"DEMO-{scenario.upper()[:8]}"
    ds.Rows = size
    ds.Columns = size
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.RescaleSlope = 1.0
    ds.RescaleIntercept = -1024.0
    ds.WindowCenter = 40.0
    ds.WindowWidth = 80.0
    ds.PixelData = stored_pixels.tobytes()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(output_path), write_like_original=False)
    return output_path
