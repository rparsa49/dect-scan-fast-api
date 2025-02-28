import pydicom
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, minimize
from pathlib import Path

DATA_DIR = Path("data")

def load_json(file_name):
    with open(DATA_DIR / file_name, "r") as file:
        return json.load(file)

# Saito 2017a Eq. 1 - Calculate delta_HU
def delta_HU(alpha, HU_H, HU_L):
    return (1 + alpha) * HU_H - (alpha * HU_L)

# Saito 2017a Eq. 2 - Calculate electron density relative to water
def rho_e(delta_HU):
    return delta_HU / 1000 + 1

# Saito 2012 Eq. 4 - Calculate electron density with parameter fit
def rho_e_calc(delta_HU, a, b):
    return (a * (delta_HU / 1000) + b)

# Saito 2017a Eq. 4 - Reduced CT number
def reduce_ct(HU):
    return HU/1000 + 1

# Saito 2017a Eq. 8 - Effective Atomic Number
def zeff(gamma, ct, rho):
    return gamma * ((ct/rho) - 1)

# Load circle data
CIRCLE_DATA = load_json("circles.json")

# True electron densities (ρe) for materials
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
    "30% CaCO3": 1.268,
    "50% CaCO3": 1.462,
    "Cortical Bone": 1.775,
    "Lung": 0.296, 
    "Air": 0
}

# Optimize alpha to match true electron density using Saito 2017a formula
def optimize_alpha(HU_H, HU_L, true_rho):
    def objective(alpha):
        delta_hu = delta_HU(alpha, HU_H, HU_L)
        calculated_rho = rho_e(delta_hu)
        return abs(calculated_rho - true_rho)

    result = minimize_scalar(objective, bounds=(0, 1), method="bounded")
    return result.x  # Optimal alpha

# Optimize a and b to match true electron density using Saito 2012 formula
def optimize_a_b(delta_HU, true_rho):
    def objective(params):
        a, b = params
        calculated_rho = rho_e_calc(delta_HU, a, b)
        return abs(calculated_rho - true_rho)

    initial_guess = [1, 1]
    result = minimize(objective, initial_guess, method="Nelder-Mead")
    return result.x  # Optimized [a, b]

# Load DICOM images
low_path = '/Users/royaparsa/Downloads/high-low/1.3.12.2.1107.5.1.4.83775.30000024051312040257200015808/test.dcm'  # 140 KVP
high_path = '/Users/royaparsa/Downloads/high-low/1.3.12.2.1107.5.1.4.83775.30000024051312040257200019235/test.dcm'  # 70 KVP

dicom_data_h = pydicom.dcmread(high_path)
dicom_data_l = pydicom.dcmread(low_path)

high_image = dicom_data_h.pixel_array
low_image = dicom_data_l.pixel_array

# Process head phantom
saved_circles = CIRCLE_DATA["head"]

calculated_rhos = []
true_rhos = []
materials_list = []
optimized_alphas = []
optimized_as, optimized_bs = [], []

for circle in saved_circles:
    x, y, radius, material = circle["x"], circle["y"], circle["radius"], circle["material"]

    if material not in TRUE_RHO:
        print(f"Warning: Material '{material}' not found in TRUE_RHO.")
        continue

    true_rho = TRUE_RHO[material]

    # Mask for circular region
    mask = np.zeros(high_image.shape, dtype=np.uint8)
    cv2.circle(mask, (x, y), int(radius), 1, thickness=-1)

    high_pixel_values = high_image[mask == 1]
    low_pixel_values = low_image[mask == 1]

    mean_high_hu = np.mean(high_pixel_values) * \
        dicom_data_h.RescaleSlope + dicom_data_h.RescaleIntercept
    mean_low_hu = np.mean(low_pixel_values) * \
        dicom_data_l.RescaleSlope + dicom_data_l.RescaleIntercept

    # Step 1: Optimize alpha
    optimal_alpha = optimize_alpha(mean_high_hu, mean_low_hu, true_rho)
    optimized_alphas.append(optimal_alpha)

    # Step 2: Use optimal alpha to calculate delta_HU
    delta_hu = delta_HU(optimal_alpha, mean_high_hu, mean_low_hu)

    # Step 3: Optimize a and b
    optimized_a, optimized_b = optimize_a_b(delta_hu, true_rho)

    optimized_as.append(optimized_a)
    optimized_bs.append(optimized_b)

    # Step 4: Calculate rho_e using optimized a and b
    calculated_rho = rho_e_calc(delta_hu, optimized_a, optimized_b)

    calculated_rhos.append(calculated_rho)
    true_rhos.append(true_rho)
    materials_list.append(material)

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(materials_list, true_rhos,
            label="True Electron Density", color='blue', marker='o')
plt.scatter(materials_list, calculated_rhos,
            label="Calculated Electron Density (optimized a, b)", color='red', marker='x')

plt.xlabel("Material")
plt.ylabel("Electron Density")
plt.title("Comparison of True and Calculated Electron Density for Each Insert (With Optimized a, b)")
plt.xticks(rotation=45)
plt.legend()
plt.grid()

# Show plot
plt.show()

# Optional: Print optimized a and b for each material
for mat, a, b in zip(materials_list, optimized_as, optimized_bs):
    print(f"Material: {mat}, Optimized a: {a:.4f}, Optimized b: {b:.4f}")