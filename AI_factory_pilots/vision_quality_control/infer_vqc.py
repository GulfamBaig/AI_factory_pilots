
"""Load the trained Keras model and run predictions on a few sample images."""
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

def main():
    model_path = "artifacts/vqc_model.h5"
    if not os.path.exists(model_path):
        print("Model not found. Run train_vqc.py first.")
        return
    model = load_model(model_path)
    # pick some sample images
    sample_dir = "vqc_data"
    paths = []
    for cls in ["defect","ok"]:
        cls_dir = os.path.join(sample_dir, cls)
        for i, fname in enumerate(os.listdir(cls_dir)[:5]):
            paths.append(os.path.join(cls_dir,fname))
    for p in paths:
        img = image.load_img(p, color_mode='grayscale', target_size=(64,64))
        arr = image.img_to_array(img)/255.0
        arr = np.expand_dims(arr, 0)
        pred = model.predict(arr)[0][0]
        label = "defect" if pred>0.5 else "ok"
        print(f"{os.path.basename(p)} -> pred:{pred:.3f} -> {label}")

if __name__ == '__main__':
    main()
