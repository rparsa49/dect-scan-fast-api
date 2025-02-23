import pydicom, cv2, json
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from pathlib import Path

DATA_DIR = Path("data")

def load_json(file_name):
    with open(DATA_DIR / file_name, "r") as file:
        return json.load(file)
    
CIRCLE_DATA = load_json("circles.json")

# table of truth values
true_rho = {
    "Lung": 0.46,
    "Adipose": 0.94,
    "Breast": 0.99,
    "True Water": 1,
    "Solid Water": 1.02,
    "Muscle": 1.05,
    "Brain": 1.05,
    "Liver": 1.09,
    "Inner Bone": 1.15,
    "Cortical Bone": 1.82
}

# table of HU ranges
hu = {
    "Cortical Bone": [1000, "inf"],
    "Brain": [30, 40],
    "Liver": [45, 50],
    "Lung": [-950, -650],
    "Muscle": [45, 50],
    "Adipose": [-150, -50],
    "True Water": [0, 0],
    "Solid Water": [0, 0],
    "Breast": [41, 134],
    "Air": [-1000, -1]
}

# Mapping HU Values to material names
materialsa = list(true_rho.keys())

# How do we determine b?
b = 0

# Search range and step
alpha_min, alpha_max = 0.0, 1.0

alpha_range = np.arange(alpha_min, alpha_max, 0.1)
best_alpha = None
best_r2 = -np.inf

# Get HU from image
high_path = '/Users/royaparsa/Downloads/high-low/1.3.12.2.1107.5.1.4.83775.30000024051312040257200015808/test.dcm'
low_path = '/Users/royaparsa/Downloads/high-low/1.3.12.2.1107.5.1.4.83775.30000024051312040257200019235/test.dcm'

# Obtain test data
dicom_datah = pydicom.dcmread(high_path)
dicom_datal = pydicom.dcmread(low_path)
high_image = dicom_datah.pixel_array
low_image = dicom_datal.pixel_array

high_hu, low_hu = [], []

saved_circles = CIRCLE_DATA["head"]

for circle in saved_circles:
    x, y, radius = circle["x"], circle["y"], circle["radius"]
    mask = np.zeros(high_image.shape, dtype=np.uint8)
    cv2.circle(mask, (x, y), int(radius * 1.0), 1, thickness=-1)
    high_pixel_values = high_image[mask == 1]
    low_pixel_values = low_image[mask == 1]

    high_hu.append(np.mean(high_pixel_values))
    low_hu.append(np.mean(low_pixel_values))

# Classify insert materials
mats_h = []
for i in high_hu: 
    for material, (min_hu, max_hu) in hu.items():
        if float(min_hu) <= float(i) <= float(max_hu):
            mats_h.append(material)
mats_l = []
for i in low_hu:
    for material, (min_hu, max_hu) in hu.items():
        if float(min_hu) <= float(i) <= float(max_hu):
            mats_h.mats_l(material)

# Search
for alpha in alpha_range:
    for HU_h, HU_l in zip(high_hu, low_hu):
        delta_HU = (1 + alpha) * HU_h - (alpha * HU_l) # compute delta HU
        rho_cal = alpha * (delta_HU / 1000) + b # compute rho_cal
        # TODO: compute R^2 for calculated rho and true rho
        