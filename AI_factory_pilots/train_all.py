
"""Quick script to train both pilots (PM + VQC) sequentially."""
import subprocess
import sys

print("Training Predictive Maintenance model...")
subprocess.run([sys.executable, "-u", "predictive_maintenance/train_pm.py"], check=True)
print("\nTraining Vision QC model (this may take a few minutes)...")
subprocess.run([sys.executable, "-u", "vision_quality_control/train_vqc.py"], check=True)
print("\nAll training finished. Check artifacts/ folders in each submodule.")
