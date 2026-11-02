"""
Training Pipeline for RSNA Intracranial Hemorrhage Multi-Label Classification (DenseNet121).
Implements patient-level GroupShuffleSplit to prevent data leakage across slices from the same patient,
positive class-weighting via BCEWithLogitsLoss, and saves checkpoints to models/ct_haemorrhage/.
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import pydicom
from sklearn.model_selection import GroupShuffleSplit
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import sys
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ai_engine.ct_processor import DenseNet121HaemorrhageModel
from config import DEVICE, HAEMORRHAGE_WEIGHTS_PATH, ICH_SUBTYPES
from preprocessing.dicom_utils import build_multiwindow_tensor


class RSNADicomSliceDataset(Dataset):
    """Loads RSNA DICOM slices and returns 3-channel HU windowed tensors + 6 binary labels."""

    def __init__(self, df: pd.DataFrame, dicom_dir: Path) -> None:
        self.df = df.reset_index(drop=True)
        self.dicom_dir = Path(dicom_dir)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        dcm_path = self.dicom_dir / f"{row['slice_id']}.dcm"
        ds = pydicom.dcmread(str(dcm_path), force=True)
        hu = ds.pixel_array.astype(np.float32) * float(getattr(ds, "RescaleSlope", 1.0)) + float(
            getattr(ds, "RescaleIntercept", -1024.0)
        )
        x = build_multiwindow_tensor(hu, target_size=(224, 224))
        y = np.array([float(row[col]) for col in ICH_SUBTYPES], dtype=np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)


def train_rsna_model(csv_path: Path, dicom_dir: Path, epochs: int = 5, batch_size: int = 16, lr: float = 1e-4) -> None:
    """
    Expected CSV columns:
    slice_id, patient_id, intraparenchymal, intraventricular, subarachnoid, subdural, epidural, any_haemorrhage
    """
    df = pd.read_csv(csv_path)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=2026)
    train_idx, val_idx = next(splitter.split(df, groups=df["patient_id"]))

    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]
    print(f"Patient-level split complete: {len(train_df)} train slices, {len(val_df)} val slices.")

    train_loader = DataLoader(RSNADicomSliceDataset(train_df, dicom_dir), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(RSNADicomSliceDataset(val_df, dicom_dir), batch_size=batch_size, shuffle=False)

    device = torch.device(DEVICE)
    model = DenseNet121HaemorrhageModel(num_classes=len(ICH_SUBTYPES)).to(device)

    # Class imbalance handling via positive weights
    pos_counts = np.maximum(train_df[ICH_SUBTYPES].sum().values, 1.0)
    neg_counts = len(train_df) - pos_counts
    pos_weight = torch.tensor(neg_counts / pos_counts, dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_loss = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = model(x_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * len(x_b)

        train_loss = running_loss / max(len(train_df), 1)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_b, y_b in val_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                logits = model(x_b)
                val_loss += float(criterion(logits, y_b).item()) * len(x_b)
        val_loss /= max(len(val_df), 1)

        print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {"model_state_dict": model.state_dict(), "epoch": epoch, "val_loss": val_loss},
                str(HAEMORRHAGE_WEIGHTS_PATH),
            )
            print(f"Saved improved checkpoint to {HAEMORRHAGE_WEIGHTS_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RSNA Haemorrhage DenseNet121 Model")
    parser.add_argument("--csv", type=Path, required=True, help="Path to patient-grouped RSNA metadata CSV")
    parser.add_argument("--dicom-dir", type=Path, required=True, help="Directory containing .dcm slices")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    train_rsna_model(args.csv, args.dicom_dir, epochs=args.epochs, batch_size=args.batch_size)
