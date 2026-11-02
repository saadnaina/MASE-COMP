"""
Convenience launcher for NeuroScan AI.
Run via: python run.py
"""
import subprocess
import sys
from pathlib import Path


def main() -> None:
    app_path = Path(__file__).resolve().parent / "app" / "main.py"
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.headless=false",
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
