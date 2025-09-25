
"""Generate a synthetic image dataset for defect detection.
Creates simple grayscale images where defects are small dark spots/lines.
"""
import numpy as np
from PIL import Image
import os

def make_image(size=(64,64), defect=False, rng=None):
    if rng is None:
        rng = np.random.RandomState()
    base = rng.normal(loc=200, scale=20, size=size).astype(np.uint8)
    img = base.copy()
    if defect:
        # add 1-3 random dark blobs/lines
        n = rng.randint(1,4)
        for _ in range(n):
            x = rng.randint(0, size[0])
            y = rng.randint(0, size[1])
            w = rng.randint(2, 10)
            h = rng.randint(2, 10)
            img[max(0,x-w):min(size[0], x+w), max(0,y-h):min(size[1], y+h)] = rng.randint(0,60)
    return img

def generate_dataset(out_dir="vqc_data", n_pos=1500, n_neg=3500, size=(64,64)):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "defect"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "ok"), exist_ok=True)
    rng = np.random.RandomState(1)
    for i in range(n_pos):
        img = make_image(size=size, defect=True, rng=rng)
        Image.fromarray(img).save(os.path.join(out_dir, "defect", f"defect_{i}.png"))
    for i in range(n_neg):
        img = make_image(size=size, defect=False, rng=rng)
        Image.fromarray(img).save(os.path.join(out_dir, "ok", f"ok_{i}.png"))
    print("Saved synthetic images to", out_dir)

if __name__ == '__main__':
    generate_dataset()
