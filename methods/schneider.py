import json
from pathlib import Path
import pydicom
import numpy as np
import pandas as pd
import scipy as sp
import cv2 
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.constants import physical_constants
from numpy.polynomial.polynomial import Polynomial
from sklearn.metrics import mean_squared_error

DATA_DIR = Path("data")

def load_json(file_name):
    with open(DATA_DIR / file_name, "r") as file:
        return json.load(file)

CIRCLE_DATA = load_json("circles.json")
MATERIAL_PROPERTIES = load_json("material_properties.json")
ELEMENTAL_PROPERTIES = load_json("element_properties.json")
ICRP_PROPERTIES = load_json("icrp.json")

def validate_spr_calibration_fit(material_list, spr_model, model_params, Kph, Kcoh, KKN):
    seen = set()
    unique_materials = []

    for m in material_list:
        if m not in seen:
            seen.add(m)
            unique_materials.append(m)

    mu_water = calculate_mu("True Water", Kph, Kcoh, KKN)
    HU_values, predicted_sprs, true_sprs = [], [], []

    for m in unique_materials:
        mu = calculate_mu(m, Kph, Kcoh, KKN)
        HU = hounsfield_schneider(mu, mu_water)
        HU_values.append(HU)

        # Predict SPR using model
        predicted_spr = spr_model(HU, *model_params)
        predicted_sprs.append(predicted_spr)

        # Calculate true SPR
        rhoe = compute_rhoe_schneider(m)
        I = compute_I(m)
        true_spr = calculate_spr(rhoe, I)
        true_sprs.append(true_spr)

    # Plotting
    x_fit = np.linspace(min(HU_values)-50, max(HU_values)+50, 500)
    y_fit = spr_model(x_fit, *model_params)

    plt.figure(figsize=(8, 6))
    plt.plot(x_fit, y_fit, 'r-', label='Calibration Curve (HU → SPR)')
    plt.scatter(HU_values, true_sprs, color='blue',
                label='True SPR (from composition)', marker='o')
    plt.scatter(HU_values, predicted_sprs, color='green',
                label='Predicted SPR (from HU)', marker='x')

    for i, mat in enumerate(unique_materials):
        plt.plot([HU_values[i], HU_values[i]], [true_sprs[i],
                 predicted_sprs[i]], 'gray', linestyle='--', linewidth=1)
        plt.annotate(mat, (HU_values[i], true_sprs[i]), fontsize=8, ha='right')

    plt.xlabel("Hounsfield Unit (HU)")
    plt.ylabel("Stopping Power Ratio (SPR)")
    plt.title("Calibration Curve Validation (Scan Materials, HU from Schneider μ)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Error metrics
    true_np = np.array(true_sprs)
    predicted_np = np.array(predicted_sprs)
    mae = np.mean(np.abs(true_np - predicted_np))
    rmse = np.sqrt(np.mean((true_np - predicted_np) ** 2))
    print(f"\nSPR Validation MAE: {mae:.4f}")
    print(f"SPR Validation RMSE: {rmse:.4f}")
    
def validate_ed_calibration_schneider_fit(material_list, a, b, Kph, Kcoh, KKN):
    """
    Computes HU from mu for given materials using Schneider model, then plots
    predicted ED (from HU) vs. true ED (from composition).
    """
    seen = set()
    unique_materials = []

    for m in material_list:
        if m not in seen:
            seen.add(m)
            unique_materials.append(m)

    # Calculate HU using Schneider-based μ values
    mu_water = calculate_mu("True Water", Kph, Kcoh, KKN)
    HU_values = []

    for m in unique_materials:
        mu = calculate_mu(m, Kph, Kcoh, KKN)
        HU = hounsfield_schneider(mu, mu_water)
        HU_values.append(HU)

    # Apply calibration model to get predicted ED
    predicted_ED = [a * hu + b for hu in HU_values]
    true_ED = [MATERIAL_PROPERTIES[m]["rho_e_w"] for m in unique_materials]

    # Fit curve line
    x_fit = np.linspace(min(HU_values) - 50, max(HU_values) + 50, 500)
    y_fit = a * x_fit + b

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(x_fit, y_fit, 'r-', label='Calibration Curve (HU → ED)')
    plt.scatter(HU_values, true_ED, color='blue',
                label='True ED (from composition)', marker='o')
    plt.scatter(HU_values, predicted_ED, color='green',
                label='Predicted ED (from HU)', marker='x')

    for i, mat in enumerate(unique_materials):
        plt.plot([HU_values[i], HU_values[i]], [true_ED[i],
                 predicted_ED[i]], 'gray', linestyle='--', linewidth=1)
        plt.annotate(mat, (HU_values[i], true_ED[i]), fontsize=8, ha='right')

    plt.xlabel("Hounsfield Unit (HU)")
    plt.ylabel("Electron Density (rho_e)")
    plt.title("Calibration Curve Validation (Scan Materials, HU from Schneider μ)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Error metrics
    true_ED_np = np.array(true_ED)
    predicted_ED_np = np.array(predicted_ED)
    mae = np.mean(np.abs(true_ED_np - predicted_ED_np))
    rmse = np.sqrt(np.mean((true_ED_np - predicted_ED_np) ** 2))
    print(f"\nMAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

def plot_true_vs_calculated_rhoe():
    true_rhoe = []
    calculated_rhoe = []
    labels = []

    for material in MATERIAL_PROPERTIES.keys():
        if material == '50% CaCO3' or material == '30% CaCO3':
            continue
        try:
            true_value = MATERIAL_PROPERTIES[material]["rho_e_w"]
            calc_value = compute_rhoe_schneider(material)
            true_rhoe.append(true_value)
            calculated_rhoe.append(calc_value)
            labels.append(material)
        except Exception as e:
            print(f"Error for {material}: {e}")

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.scatter(true_rhoe, calculated_rhoe)

    # Annotate each point with material name
    for i, label in enumerate(labels):
        plt.annotate(label, (true_rhoe[i], calculated_rhoe[i]), fontsize=8)

    # Plot y=x line for perfect agreement
    min_val = min(true_rhoe + calculated_rhoe)
    max_val = max(true_rhoe + calculated_rhoe)
    plt.plot([min_val, max_val], [min_val, max_val],
             'r--', label="Ideal (y=x)")

    plt.xlabel("True Electron Density (rho_e_w)")
    plt.ylabel("Calculated Electron Density (rho_e)")
    plt.title("True vs. Calculated Electron Density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_HU_and_mu(materials_list, HU_List, mu_list):
    materials_arr = np.array(materials_list)
    HU_arr = np.array(HU_List)
    mu_arr = np.array(mu_list)

    x = np.arange(len(materials_arr))  # numeric x for scatter plots

    # Plot HU values per material
    plt.figure(figsize=(10, 6))
    plt.scatter(x, HU_arr)
    plt.xticks(x, materials_arr, rotation=45, ha='right')
    plt.xlabel("Material")
    plt.ylabel("Measured HU")
    plt.title("Measured HU per Material")
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()

    # Plot mu values per material
    plt.figure(figsize=(10, 6))
    plt.scatter(x, mu_arr)
    plt.xticks(x, materials_arr, rotation=45, ha='right')
    plt.xlabel("Material")
    plt.ylabel("Calculated mu")
    plt.title("Calculated mu per Material")
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()
    
# Calculate  HU according to Schneider 1996
def hounsfield_schneider(mew, mew_w):
    # return 1000*mew/mew_w
    return ((mew / mew_w) - 1 ) * 1000

# Calculate N_g for mew
def compute_Ng(material, flag="Phantoms"):
    N_A = sp.constants.Avogadro
    composition = MATERIAL_PROPERTIES[material]["composition"] if flag == "Phantoms" else ICRP_PROPERTIES[material]["composition"]
    
    sum_term = 0
    for element, weight_fraction in composition.items():
        Z_i = ELEMENTAL_PROPERTIES[element]["number"]
        A_i = ELEMENTAL_PROPERTIES[element]["mass"]
        sum_term += (weight_fraction * Z_i) / A_i
    
    return N_A * sum_term

# Calculate weighted Z
def compute_weighted_Z(material, exponent, flag="Phantoms"):
    composition = MATERIAL_PROPERTIES[material]["composition"] if flag == "Phantoms" else ICRP_PROPERTIES[material]["composition"]

    sum_term = 0
    N_g = compute_Ng(material) if flag == "Phantoms" else compute_Ng(material, flag="ICRP")
    N_A = sp.constants.Avogadro

    for element, weight_fraction in composition.items():
        Z_i = ELEMENTAL_PROPERTIES[element]["number"]
        A_i = ELEMENTAL_PROPERTIES[element]["mass"]
        N_gi = N_A * (weight_fraction * Z_i) / A_i
        lambda_i = N_gi / N_g
        sum_term += lambda_i * (Z_i ** exponent)
    
    return (sum_term) ** (1 / exponent)

# Calculate electron density from Scheineider 1996
def compute_rhoe_schneider(material, water="True Water", flag="Phantoms"):
    Ng = compute_Ng(material) if flag == "Phantoms" else compute_Ng(material, "ICRP")
    Ng_w = compute_Ng(water)
    
    rho = MATERIAL_PROPERTIES[material]["density"] if flag == "Phantoms" else ICRP_PROPERTIES[material]["density"]
    rho_w = MATERIAL_PROPERTIES[water]["density"]
    
    return (rho * Ng) / (rho_w * Ng_w)

# Mu Model Fit Function
def mu_model_fit(X, Kph, Kcoh, KKN):
    rhoNg = X[:,0]
    Zbar = X[:,1]
    Zhat = X[:,2]
    return rhoNg * (Kph * Zbar ** 3.62 + Kcoh * Zhat ** 1.86 + KKN)

# Method for linear attenuation of a material
def linear_attenuation(material):
    rho = MATERIAL_PROPERTIES[material]["density"]
    composition = MATERIAL_PROPERTIES[material]["composition"]

    mu_total = 0.0
    for element, fraction in composition.items():
        # get elemental properties
        atomic_mass = ELEMENTAL_PROPERTIES[element]["mass"]
        atomic_number = ELEMENTAL_PROPERTIES[element]["number"]

        # number density of the element in the material
        N = (rho * fraction) / atomic_mass

        mu_a = atomic_number

        mu_total += mu_a * N
    return mu_total

# Calculate linear attenuation of a material using Eq. 8 and fitted K coefficients
def calculate_mu(material, Kph, Kcoh, KKN):
    Ng = compute_Ng(material)
    rho = MATERIAL_PROPERTIES[material]["density"] # g/cm^3
    rhoNg = (rho * Ng) / 1e23
    
    Zbar = compute_weighted_Z(material, 3.62)
    Zhat = compute_weighted_Z(material, 1.86)
    
    return rhoNg * (Kph * Zbar ** 3.62 + Kcoh * Zhat ** 1.86 + KKN) # cm^-1

# Calculate HU for tissues
def calculate_HU(tissues, Kph, Kcoh, KKN, flag="Phantoms"):
    mu_water = calculate_mu("True Water", Kph, Kcoh, KKN)
    res = []
    for tissue in tissues:
        Ng = compute_Ng(tissue) if flag == "Phantoms" else compute_Ng(tissue, "ICRP")
        rho = MATERIAL_PROPERTIES[tissue]["density"] if flag == "Phantoms" else ICRP_PROPERTIES[tissue]["density"]
        rhoNg = (rho * Ng) / 1e23
        
        Zbar = compute_weighted_Z(tissue, 3.62) if flag == "Phantoms" else compute_weighted_Z(tissue, 3.62, "ICRP")
        Zhat = compute_weighted_Z(tissue, 1.86) if flag == "Phantoms" else compute_weighted_Z(tissue, 1.86, "ICRP")
        
        mu = rhoNg * (Kph * Zbar ** 3.62 + Kcoh * Zhat ** 1.86 + KKN)  # cm^-1
        
        HU = hounsfield_schneider(mu, mu_water)
        res.append(HU)
        print(f"{tissue:<15} | Calculated HU: {HU:.2f}")
    return res

# Calculate mean excitation energy from eq 4
def compute_I(material, flag="Phantoms"):
    composition = MATERIAL_PROPERTIES[material]["composition"] if flag == "Phantoms" else ICRP_PROPERTIES[material]["composition"]
    
    num = 0.0
    den = 0.0 
    
    for element, weight_fraction in composition.items():
        Z = ELEMENTAL_PROPERTIES[element]["number"]
        A = ELEMENTAL_PROPERTIES[element]["mass"]
        I = ELEMENTAL_PROPERTIES[element]["ionization"]
        
        weight = (weight_fraction * Z) / A
        num += weight * np.log(I)
        den += weight
    
    return np.exp(num / den)

# Calculate beta proton speed fraction of light
def get_beta(kvp=219):
    kinetic_energy_mev = kvp / 1000
    proton_mass_mev = physical_constants['proton mass energy equivalent in MeV'][0]
    gamma = (proton_mass_mev + kinetic_energy_mev) / proton_mass_mev
    return np.sqrt(1 - (1 / gamma ** 2)) ** 2

# Calculate SPR using eq 1
def calculate_spr(rhoe, I, I_water=75):
    me = 9.10938356e-31
    c = 2.99792458e8
    beta = get_beta()
    
    numerator = (np.log(2*me * (c ** 2) * beta)) / (I*(1 - beta) - beta)
    denominator = (np.log(2*me * (c ** 2) * beta)) / (I_water*(1 - beta) - beta)
    return rhoe * (numerator / denominator)

def schneider(path, phantom_type, radii_ratio):
    dicom_data = pydicom.dcmread(path)
    
    image = dicom_data.pixel_array
    
    HU_List, materials_list, rhos, rhos_ICRP, mews, sprs, mean_excitations = [], [], [], [], [], [], []
    
    SAVED_CIRCLES = CIRCLE_DATA[phantom_type]
    for circle in SAVED_CIRCLES:
        x, y, radius, material = circle["x"], circle["y"], circle["radius"], circle["material"]
        if material == '50% CaCO3' or material == '30% CaCO3':
            print(f"Warning: Material '{material}' not found in TRUE_RHO")
            continue
        
        # Obtain list of materials
        if material not in materials_list:
            materials_list.append(material)
        
        mask = np.zeros(image.shape, dtype=np.uint8)
        cv2.circle(mask, (x, y), int(radius * radii_ratio), 1, thickness=-1)
        
        pixel_values = image[mask == 1]
        hu = np.mean(pixel_values) * \
            dicom_data.RescaleSlope + dicom_data.RescaleIntercept
        hu = (hu / 1000) + 1
        # HU from CT image
        HU_List.append(hu)
    
    # Calculate rho
    print("\n=== Electron Density Calculations ===")
    for material in materials_list:
        temp = compute_rhoe_schneider(material, flag="Phantoms")
        print(f"{material:<15} | Electron Density: {temp:.2f}")
        rhos.append(temp)
    
    # Formatted HU output
    print("\n=== Measured HU Values ===")
    for material, hu in zip(materials_list, HU_List):
        print(f"{material:<15} | HU: {hu:.2f}")

    # Prepare data for fitting
    rhoNg_list, Zbar_list, Zhat_list, mu_list = [], [], [], []
    mu_water = linear_attenuation("True Water")
    for i, material in enumerate(materials_list):
        rho = MATERIAL_PROPERTIES[material]["density"]
        Ng = compute_Ng(material)
        rhoNg = rho * Ng

        Zbar = compute_weighted_Z(material, 3.62)
        Zhat = compute_weighted_Z(material, 1.86)

        measured_HU = HU_List[i]
        mu = measured_HU * mu_water

        rhoNg_list.append(rhoNg)
        Zbar_list.append(Zbar)
        Zhat_list.append(Zhat)
        mu_list.append(mu)
        
    rhoNg_arr = np.array(rhoNg_list) / 1e23
    
    # Formatted MU output
    print("\n=== Measured mu Values ===")
    for material, mu in zip(materials_list, mu_list):
        print(f"{material:<15} | mu: {mu:.2f}")
 
    X = np.array([rhoNg_arr, Zbar_list, Zhat_list]).T  # transpose to shape (N, 3)
    y = np.array(mu_list)
    
    initial_guess = [1e-5, 4e-4, 0.5] # original from schneider
    bounds = ([0, 0, 0], [1e-4, 1e-3, 2])

    popt, _ = curve_fit(mu_model_fit, X, y, p0=initial_guess, bounds=bounds)
    Kph, Kcoh, KKN = popt
    
    print("\n=== Fitted Coefficients ===")
    print(f"Kph: {Kph}")
    print(f"Kcoh: {Kcoh}")
    print(f"KKN: {KKN}")

    # Check fit quality
    predicted_mu = mu_model_fit(X, *popt)
    residuals = y - predicted_mu
    rmse = np.sqrt(np.mean(residuals**2))
    print(f"RMSE of fit: {rmse}")
    
    # Step 4: Compute HU of ICRP tissues using eq. 5 and 8
    ICRP_Tissues = list(ICRP_PROPERTIES.keys())
    print("\n=== HU of ICRP Tissues ===")
    ICRP_HUs = calculate_HU(ICRP_Tissues, Kph, Kcoh, KKN, flag="ICRP")
    
    # Step 5: Compute electron density for ICRP tissues
    print("\n=== Electron Density of ICRP Tissues ===")
    for material in ICRP_Tissues:
        temp = compute_rhoe_schneider(material, flag="ICRP")
        print(f"{material:<15} | Electron Density: {temp:.3f}")
        rhos_ICRP.append(temp)
    
    # Step 6: Compute SPR for ICRP tissues
    print("\n=== SPR of ICRP Tissues ===")
    for i, material in enumerate(ICRP_Tissues):
        rhoe = rhos_ICRP[i]
        I = compute_I(material, flag="ICRP")
        spr = calculate_spr(rhoe, I)
        
        print(f"{material:<15} | SPR: {spr:.4f}")
        sprs.append(spr)
        
    # Step 7: Generate Calibration Curve
    
    # First, calculate the HU from the schneider method for our phantoms
    
    def model_func(HU, a, b):
        return a * HU + b
    
    params, _ = curve_fit(model_func, ICRP_HUs, rhos_ICRP)
    
    x_fit = np.linspace(min(ICRP_HUs), max(ICRP_HUs), 500)
    y_fit = model_func(x_fit, *params)
    
    plt.figure(figsize=(8, 5))
    plt.scatter(ICRP_HUs, rhos_ICRP, color='blue', label='Data Points')
    plt.plot(x_fit, y_fit, color='red', label='Calibration Curve')
    plt.xlabel("Hounsfield Unit (HU)")
    plt.ylabel("Electron Density")
    plt.title("Calibration Curve (ICRP)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    print("\n=== Validating ED Calibration ===")
    a, b = params
    validate_ed_calibration_schneider_fit(materials_list, a, b, Kph, Kcoh, KKN)

    def logarithmic_spr_model(HU, a, b, c):
        return a * np.log(HU + b) ** c
    
    params, _ = curve_fit(logarithmic_spr_model, ICRP_HUs, sprs, bounds = ([0, 1, 1], [10, 1000, 5]))

    x_fit = np.linspace(min(ICRP_HUs), max(ICRP_HUs), 500)
    y_fit = logarithmic_spr_model(x_fit, *params)
    
    plt.figure(figsize=(8, 5))
    plt.scatter(ICRP_HUs, sprs, color='blue', label='Data Points')
    plt.plot(x_fit, y_fit, color='red', label='Calibration Curve')
    plt.xlabel("Hounsfield Unit (HU)")
    plt.ylabel("Stopping Power")
    plt.title("Calibration Curve (ICRP)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Validate SPR Fit
    validate_spr_calibration_fit(
        material_list=materials_list,
        spr_model=logarithmic_spr_model,
        model_params=params,
        Kph=Kph,
        Kcoh=Kcoh,
        KKN=KKN
    )

# schneider('/Users/royaparsa/Desktop/Gammex-Pelvis-1cm/CT1.3.12.2.1107.5.1.4.83775.30000024051312040257200013605.dcm', "body", 0.75)
# plot_true_vs_calculated_rhoe()
