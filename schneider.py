import json
from pathlib import Path
import pydicom
import numpy as np
import pandas as pd
import scipy as sp
import cv2 
from scipy.optimize import curve_fit

DATA_DIR = Path("data")

def load_json(file_name):
    with open(DATA_DIR / file_name, "r") as file:
        return json.load(file)

CIRCLE_DATA = load_json("circles.json")
MATERIAL_PROPERTIES = load_json("material_properties.json")
ELEMENTAL_PROPERTIES = load_json("element_properties.json")

# Calculate  HU according to Schneider 1996
def hounsfield_schneider(mew, mew_w):
    return 1000*mew/mew_w

# Calculate N_g for mew
def compute_Ng(material):
    N_A = sp.constants.Avogadro
    composition = MATERIAL_PROPERTIES[material]["composition"]
    
    sum_term = 0
    for element, weight_fraction in composition.items():
        Z_i = ELEMENTAL_PROPERTIES[element]["number"]
        A_i = ELEMENTAL_PROPERTIES[element]["mass"]
        sum_term += (weight_fraction * Z_i) / A_i
    
    return N_A * sum_term

# Calculate weighted Z
def compute_weighted_Z(material, exponent):
    composition = MATERIAL_PROPERTIES[material]["composition"]

    sum_term = 0
    N_g = compute_Ng(material)
    N_A = sp.constants.Avogadro

    for element, weight_fraction in composition.items():
        Z_i = ELEMENTAL_PROPERTIES[element]["number"]
        A_i = ELEMENTAL_PROPERTIES[element]["mass"]
        N_gi = N_A * (weight_fraction * Z_i) / A_i
        lambda_i = N_gi / N_g
        sum_term += lambda_i * (Z_i ** exponent)
    
    return (sum_term) ** (1 / exponent)

# Calculate electron density from Scheineider 1996
def compute_rhoe_schneider(material, water="True Water"):
    Ng = compute_Ng(material)
    Ng_w = compute_Ng(water)
    
    rho = MATERIAL_PROPERTIES[material]["density"]
    rho_w = MATERIAL_PROPERTIES[water]["density"]
    
    return (rho * Ng) / (rho_w * Ng_w)

# Mu Model Fit Function
def mu_model_fit(X, Kph, Kcoh, KKN):
    rhoNg, Zbar, Zhat = X
    return rhoNg * (Kph * Zbar ** 3.62 + Kcoh * Zhat ** 1.86 + KKN)

# Compute linear attenuation coeff. from Schenider 1996
# def mew_schneider(material):
#     rho = compute_rhoe_schneider(material)
#     Ng = compute_Ng(material)
    
#     Z_bar = compute_weighted_Z(material, 3.62)
#     Z_hat = compute_weighted_Z(material, 1.86)
    
#     # FIGURE OUT K'S    
#     return rho * Ng * (K_ph * Z_bar ** 3.62 + K_coh + Z_hat ** 1.86 + K_KN)


# def schneider(phantom_type):
def schneider(path, phantom_type, radii_ratio):
    dicom_data = pydicom.dcmread(path)
    
    image = dicom_data.pixel_array
    
    HU_List, materials_list, rhos, mews, sprs, mean_excitations = [], [], [], [], [], []
    
    SAVED_CIRCLES = CIRCLE_DATA[phantom_type]
    # measured_water_HU = 0
    for circle in SAVED_CIRCLES:
        x, y, radius, material = circle["x"], circle["y"], circle["radius"], circle["material"]
        if material == '50% CaCO3' or material == '30% CaCO3':
        # if material not in TRUE_RHO or material == '50% CaCO3' or material == '30% CaCO3':
            print(f"Warning: Material '{material}' not found in TRUE_RHO")
            continue
        
        # if circle["material"] == "True Water":
        #     x, y, radius = circle["x"], circle["y"], circle["radius"]
        #     mask = np.zeros(image.shape, dtype=np.uint8)
        #     cv2.circle(mask, (x, y), int(radius * radii_ratio), 1, thickness=-1)
        #     pixel_values = image[mask == 1]
        #     measured_water_HU = np.mean(pixel_values) * dicom_data.RescaleSlope + dicom_data.RescaleIntercept
        #     break

        
        # Obtain list of materials
        materials_list.append(material)
        
        mask = np.zeros(image.shape, dtype=np.uint8)
        cv2.circle(mask, (x, y), int(radius * radii_ratio), 1, thickness=-1)
        
        pixel_values = image[mask == 1]
        hu = np.mean(pixel_values) * \
            dicom_data.RescaleSlope + dicom_data.RescaleIntercept
        
        # HU from CT image
        HU_List.append(hu)
    
    # Calculate rho
    print("\n=== Electron Density Calculations ===")
    for material in materials_list:
        temp = compute_rhoe_schneider(material)
        print(f"{material:<15} | Electron Density: {temp}")
        rhos.append(compute_rhoe_schneider(material))
    
    # Formatted HU output
    print("\n=== Measured HU Values ===")
    for material, hu in zip(materials_list, HU_List):
        print(f"{material:<15} | HU: {hu}")

    # Prepare data for fitting
    rhoNg_list, Zbar_list, Zhat_list, mu_list = [], [], [], []
    mu_water = 0.2 / 1000
    for i, material in enumerate(materials_list):
        rho = MATERIAL_PROPERTIES[material]["density"]
        Ng = compute_Ng(material)
        rhoNg = rho * Ng

        Zbar = compute_weighted_Z(material, 3.62)
        Zhat = compute_weighted_Z(material, 1.86)

        measured_HU = HU_List[i]
        mu = mu_water * (measured_HU / 1000)

        rhoNg_list.append(rhoNg)
        Zbar_list.append(Zbar)
        Zhat_list.append(Zhat)
        mu_list.append(mu)
    
    # Formatted MU output
    print("\n=== Measured mu Values ===")
    for material, mu in zip(materials_list, mu_list):
        print(f"{material:<15} | mu: {mu}")
 
    X = np.array([rhoNg_list, Zbar_list, Zhat_list])
    y = np.array(mu_list)
    
    initial_guess = [1e-5, 4e-4, 0.5] # original from schneider
    
    popt, _ = curve_fit(mu_model_fit, X, y, p0=initial_guess)
    Kph, Kcoh, KKN = popt
    
    print("\n=== Fitted Coefficients ===")
    print(f"Kph: {Kph}")
    print(f"Kcoh: {Kcoh}")
    print(f"KKN: {KKN}")

    
schneider('/Users/royaparsa/Desktop/Gammex-Pelvis-1cm/CT1.3.12.2.1107.5.1.4.83775.30000024051312040257200013605.dcm', "body", 0.75)
# schneider("body")