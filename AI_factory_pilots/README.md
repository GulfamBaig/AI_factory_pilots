AI Factory Pilots — Predictive Maintenance & Vision Quality Control

This repository provides two low-cost, edge-ready AI pilot solutions designed for small-scale factories:

predictive_maintenance — Sensor-based anomaly detection and machine health classification.

vision_quality_control — Camera-based defect detection using a lightweight CNN.

🎯 Goal

Deliver complete, ready-to-run pipelines (data generation → training → inference) that you can test locally with synthetic data.
For real-world deployment, simply replace the synthetic data with your factory’s actual sensor logs and camera images.

📂 Project Structure
AI-Factory-Pilots/
│
├── predictive_maintenance/
│   ├── data_generator.py
│   ├── train_pm.py
│   ├── infer_pm.py
│   └── requirements.txt
│
├── vision_quality_control/
│   ├── data_generator.py
│   ├── train_vqc.py
│   ├── infer_vqc.py
│   └── requirements.txt
│
├── LICENSE
├── .gitignore
└── README.md

🚀 How to Run

Set up environment

Use Python 3.9+

Create a virtual environment and activate it

Install dependencies

pip install -r predictive_maintenance/requirements.txt
pip install -r vision_quality_control/requirements.txt


Generate data & train models

python predictive_maintenance/train_pm.py
python vision_quality_control/train_vqc.py


Run inference on test samples

python predictive_maintenance/infer_pm.py
python vision_quality_control/infer_vqc.py

📊 Accuracy

Both training pipelines are tuned to achieve 90%+ accuracy on synthetic test data.

Real-world accuracy will depend on the quality and representativeness of your factory’s data.

🔧 Customization

Replace data_generator.py logic with your own data loaders.

Adjust training hyperparameters in train_pm.py and train_vqc.py.

Retrain and deploy on edge devices for production use.
