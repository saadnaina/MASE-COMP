"""
Centralized configuration, clinical safety constants, and file paths for NeuroScan AI.
GMU MASE 2026 AI Healthcare Competition Prototype.
"""
from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent
APP_DIR = BASE_DIR / "app"
AI_ENGINE_DIR = BASE_DIR / "ai_engine"
MODELS_DIR = BASE_DIR / "models"
DATABASE_DIR = BASE_DIR / "database"
REPORTS_DIR = BASE_DIR / "reports" / "generated"
ASSETS_DIR = BASE_DIR / "assets"
DEMO_CT_DIR = ASSETS_DIR / "demo_ct"

for directory in [
    MODELS_DIR / "ct_haemorrhage",
    MODELS_DIR / "ct_ischaemic",
    MODELS_DIR / "facial_analysis",
    MODELS_DIR / "speech_analysis",
    DATABASE_DIR,
    REPORTS_DIR,
    ASSETS_DIR,
    DEMO_CT_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

DB_PATH = DATABASE_DIR / "neuroscan_ai.db"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_LABEL = (
    f"GPU ({torch.cuda.get_device_name(0)})"
    if torch.cuda.is_available()
    else "CPU (Standard Local Execution)"
)

APP_NAME = "NeuroScan AI"
APP_SUBTITLE = "Integrated Multimodal AI System for Early Stroke Screening & CT Analysis"
APP_VERSION = "1.0.0-MASE2026"
COMPETITION_NAME = "GMU MASE 2026 AI Healthcare Competition"

MEDICAL_DISCLAIMER_SHORT = (
    "EXPERIMENTAL RESEARCH PROTOTYPE — NOT FOR CLINICAL DIAGNOSTIC USE. "
    "Suspected acute stroke symptoms require immediate emergency medical escalation (Call Emergency Services). "
    "Do not delay emergency care while waiting for AI or CT analysis. "
    "A negative CT haemorrhage result NEVER excludes acute ischaemic stroke."
)

MEDICAL_DISCLAIMER_FULL = (
    "IMPORTANT CLINICAL & REGULATORY LIMITATION: NeuroScan AI is an experimental research "
    "and educational software prototype developed for the GMU MASE 2026 competition. "
    "It is NOT a certified or clinically validated medical device and must never be used "
    "to independently diagnose, treat, or rule out stroke or any neurological pathology. "
    "1) Any sudden neurological symptom (Facial droop, Arm weakness, Speech abnormality) "
    "must trigger immediate emergency medical escalation regardless of AI output. "
    "2) Non-contrast CT (NCCT) sensitivity for hyperacute ischaemic stroke is limited; "
    "a negative CT haemorrhage result NEVER rules out ischaemic stroke. "
    "3) Normal vital signs (BP, HR, SpO2) never exclude acute cerebrovascular events."
)

HAEMORRHAGE_WEIGHTS_PATH = MODELS_DIR / "ct_haemorrhage" / "densenet121_rsna_ich.pth"
ISCHAEMIC_WEIGHTS_PATH = MODELS_DIR / "ct_ischaemic" / "unet_aisd_lesion.pth"

ICH_SUBTYPES = [
    "intraparenchymal",
    "intraventricular",
    "subarachnoid",
    "subdural",
    "epidural",
    "any_haemorrhage",
]

CT_WINDOWS = {
    "brain": {"W": 80.0, "L": 40.0},
    "subdural": {"W": 200.0, "L": 80.0},
    "bone_soft": {"W": 380.0, "L": 40.0},
}

PROVENANCE_TAGS = [
    "USER_REPORTED",
    "SENSOR_MEASURED",
    "AI_DERIVED",
    "SIMULATED_DEMO",
    "MISSING",
]

SPEECH_REFERENCE_SENTENCE = (
    "The quick brown fox jumps over the lazy dog near the quiet river bank."
)
