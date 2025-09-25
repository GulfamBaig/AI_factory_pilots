
# Predictive Maintenance Pilot

Steps:
1. Install requirements: `pip install -r requirements.txt`
2. Generate synthetic data (optional): `python data_generator.py` (creates pm_synthetic.csv)
3. Train model: `python train_pm.py`
4. Run inference demo: `python infer_pm.py`

Model: RandomForestClassifier trained on synthetic sensor features. The synthetic dataset and model are configured so the test accuracy is typically >90% on the held-out synthetic set.

Replace `data_generator.generate()` with your real sensor CSV loader and re-run training.
