import pydicom
import cv2
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

DATA_DIR = Path("data")

def load_json(file_name):
    with open(DATA_DIR / file_name, "r") as file:
        return json.load(file)

# Saito 2017a eq. 1
def delta_HU(alpha, HU_H, HU_L):
    return (1 + alpha) * HU_H - (alpha * HU_L)

# Saito 2017a eq. 2
def rho_e(delta_HU):
    return delta_HU / 1000 + 1

CIRCLE_DATA = load_json("circles.json")

TRUE_RHO = {
    "LN-300": 0.282,
    "LN-450": 0.458,
    "Adipose": 0.949,
    "Breast": 0.970,
    "True Water": 1.00,
    "Solid Water": 0.999,
    "Brain": 1.024,
    "Liver": 1.054,
    "Inner Bone": 1.158,
    "30% CaC03": 1.268,
    "50% CaC03": 1.462,
    "Cortical Bone": 1.775,
    "Air": 0
}

HU_RANGES = {
    'LN-300': [-674.51, -646.29],
    'LN-450': [-501.61, -478.64],
    'Adipose': [-44.46, -40.0],
    'Brain': [31.01, 44.41],
    'Liver': [57.72, 69.84],
    'Inner Bone': [251.57, 444.36],
    'Cortical Bone': [1141.35, np.inf],
    'Breast': [-38.49, -28.54],
    '30% CaC03': [391.14, 590.57],
    '50% CaC03': [692.78, 1108.77]
}

# Function to minimize: difference between calculated rho_e and true rho_e
def optimize_alpha(HU_H, HU_L, true_rho):
    def objective(alpha):
        delta_hu = delta_HU(alpha, HU_H, HU_L)
        calculated_rho = rho_e(delta_hu)
        return abs(calculated_rho - true_rho)

    result = minimize_scalar(objective, bounds=(0, 1), method="bounded")
    return result.x  # Optimal alpha


# Load DICOM images
low_path = '/Users/royaparsa/Downloads/high-low/1.3.12.2.1107.5.1.4.83775.30000024051312040257200015808/test.dcm'  # 140 KVP
high_path = '/Users/royaparsa/Downloads/high-low/1.3.12.2.1107.5.1.4.83775.30000024051312040257200019235/test.dcm'  # 70 KVP

dicom_data_h = pydicom.dcmread(high_path)
dicom_data_l = pydicom.dcmread(low_path)

high_image = dicom_data_h.pixel_array
low_image = dicom_data_l.pixel_array
saved_circles = CIRCLE_DATA["head"]

calculated_rhos = []
true_rhos = []
materials_list = []
optimized_alphas = []

for circle in saved_circles:
    x, y, radius = circle["x"], circle["y"], circle["radius"]

    # Create mask for circle region
    mask = np.zeros(high_image.shape, dtype=np.uint8)
    cv2.circle(mask, (x, y), int(radius), 1, thickness=-1)

    high_pixel_values = high_image[mask == 1]
    low_pixel_values = low_image[mask == 1]

    mean_high_hu = np.mean(high_pixel_values)
    mean_low_hu = np.mean(low_pixel_values)
    
    mean_high_hu = mean_high_hu * dicom_data_h.RescaleSlope + dicom_data_h.RescaleIntercept
    mean_low_hu = mean_low_hu * dicom_data_l.RescaleSlope + dicom_data_l.RescaleIntercept

    # Identify material based on HU ranges
    identified_material = "Unknown"
    for material, (low, high) in HU_RANGES.items():
        if low <= mean_high_hu <= high:
            identified_material = material
            break

    if identified_material in TRUE_RHO:
        true_rho = TRUE_RHO[identified_material]

        # Find optimal alpha
        optimal_alpha = optimize_alpha(mean_high_hu, mean_low_hu, true_rho)
        optimized_alphas.append(optimal_alpha)

        # Calculate electron density
        delta_hu = delta_HU(optimal_alpha, mean_high_hu, mean_low_hu)
        calculated_rho = rho_e(delta_hu)

        calculated_rhos.append(calculated_rho)
        true_rhos.append(true_rho)
        materials_list.append(identified_material)

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(materials_list, true_rhos,
            label="True Electron Density", color='blue', marker='o')
plt.scatter(materials_list, calculated_rhos,
            label="Calculated Electron Density", color='red', marker='x')

plt.xlabel("Material")
plt.ylabel("Electron Density")
plt.title("Comparison of True and Calculated Electron Density for Each Insert")
plt.xticks(rotation=45)
plt.legend()
plt.grid()

# Show plot
plt.show()
