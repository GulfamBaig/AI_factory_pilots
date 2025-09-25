
# AI Factory Pilots — Predictive Maintenance & Vision Quality Control

This repository contains two low-cost, edge-ready pilot projects suitable for a small factory:
1. `predictive_maintenance` — sensor-based anomaly detection/classification for machine health.
2. `vision_quality_control` — camera-based defect detection using a simple CNN.

**Purpose:** Provide end-to-end code (data generation, training, inference) so you can upload to GitHub and run locally.  
**Note:** This uses synthetic data so you can test the pipeline immediately. Replace synthetic data with your factory's real data for production.

## Structure
- predictive_maintenance/
  - data_generator.py
  - train_pm.py
  - infer_pm.py
  - requirements.txt
- vision_quality_control/
  - data_generator.py
  - train_vqc.py
  - infer_vqc.py
  - requirements.txt
- LICENSE
- .gitignore

## How to run (high level)
1. Create a Python virtual env (Python 3.9+ recommended).
2. Install requirements for each pilot (see each subfolder requirements.txt).
3. Run data generation and training scripts:
   - `python predictive_maintenance/train_pm.py`
   - `python vision_quality_control/train_vqc.py`
4. Run inference scripts to see example predictions:
   - `python predictive_maintenance/infer_pm.py`
   - `python vision_quality_control/infer_vqc.py`

The training scripts use synthetic data and are configured to reach >90% accuracy on the synthetic test sets. Real-world results depend on data quality.

Replace synthetic generators with your real sensor logs and camera images and re-train.

