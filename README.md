# NeuroScan AI — GMU MASE 2026 AI Healthcare Competition
**An Integrated Multimodal AI System for Early Stroke Screening and CT-Based Stroke Analysis**

## 1. Medical Limitation & Safety Directive
NeuroScan AI is an **experimental research and educational prototype**. It is **NOT** a clinically validated diagnostic device and must never independently diagnose or rule out stroke.
- Any sudden stroke-like symptom (Facial droop, Arm drift, Speech difficulty) triggers immediate emergency medical advice (`IMMEDIATE_EMERGENCY_ESCALATION`).
- A negative CT haemorrhage classification **never** rules out acute ischaemic stroke.
- Normal blood pressure, heart rate, or oxygen saturation **never** rules out stroke.

## 2. Quick Start (Windows 10/11)
```powershell
# 1. Unzip NEUROSCAN_AI.zip and open PowerShell inside the NEUROSCAN_AI folder
cd NEUROSCAN_AI

# 2. Create and activate a Python 3.11 virtual environment
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Run the automated clinical safety & unit test suite
pytest -v

# 5. Launch the NeuroScan AI Hospital Dashboard
python run.py
```

## 3. Competition Demonstration Walkthrough (Scenarios A, B, and C)
1. **Seed Scenarios**: On **1. Main Dashboard**, open *Competition Demonstration Scenario Controls* and click **Load 3 Competition Demo Scenarios (A, B, C)**.
   - **Scenario A (`ASM-DEMOA001`)**: Haemorrhage-positive CT case with acute symptoms.
   - **Scenario B (`ASM-DEMOB002`)**: FAST-positive patient with a negative CT haemorrhage scan (demonstrating the mandatory acute ischaemic stroke safety override).
   - **Scenario C (`ASM-DEMOC003`)**: Patient with missing CT scan and degraded camera lighting where manual symptom screening still escalates safely.
2. **Test Live AI Brain CT Analysis**: Go to **3. Brain CT Analysis**, expand *Offline DICOM Phantom Generator*, generate a calibrated DICOM phantom (or upload your own `.dcm` slices), and click **Run AI Brain CT Analysis** to inspect the multi-window HU conversion, 5-subtype DenseNet121 predictions, Grad-CAM activation heatmaps, and experimental MONAI 2D U-Net segmentation.
3. **Inspect Multimodal Integration**: Go to **4. Integrated Multimodal Analysis** to review cross-modal conflict alerts, missing data flags, and data provenance (`USER_REPORTED`, `AI_DERIVED`, `SIMULATED_DEMO`).
4. **Export Clinical PDF Report**: Go to **5. Reports & Assessment History** and click **Generate & Download PDF Assessment Report**.
# MASE-COMP
