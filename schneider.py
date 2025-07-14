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

DATA_DIR = Path("data")

def load_json(file_name):
    with open(DATA_DIR / file_name, "r") as file:
        return json.load(file)

CIRCLE_DATA = load_json("circles.json")
MATERIAL_PROPERTIES = load_json("material_properties.json")
ELEMENTAL_PROPERTIES = load_json("element_properties.json")
ICRP_PROPERTIES = load_json("icrp.json")

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
    
    for tissue in tissues:
        Ng = compute_Ng(tissue) if flag == "Phantoms" else compute_Ng(tissue, "ICRP")
        rho = MATERIAL_PROPERTIES[tissue]["density"] if flag == "Phantoms" else ICRP_PROPERTIES[tissue]["density"]
        rhoNg = (rho * Ng) / 1e23
        
        Zbar = compute_weighted_Z(tissue, 3.62) if flag == "Phantoms" else compute_weighted_Z(tissue, 3.62, "ICRP")
        Zhat = compute_weighted_Z(tissue, 1.86) if flag == "Phantoms" else compute_weighted_Z(tissue, 1.86, "ICRP")
        
        mu = rhoNg * (Kph * Zbar ** 3.62 + Kcoh * Zhat ** 1.86 + KKN)  # cm^-1
        
        HU = hounsfield_schneider(mu, mu_water)
        print(f"{tissue:<15} | Calculated HU: {HU:.2f}")

# Calculate beta proton speed fraction of light
def beta(kvp=200):
    kinetic_energy_mev = kvp / 1000
    proton_mass_mev = physical_constants['proton mass energy equivalent in MeV'][0]
    gamma = (proton_mass_mev + kinetic_energy_mev) / proton_mass_mev
    return np.sqrt(1 - (1 / gamma ** 2)) ** 2



# def schneider(phantom_type):
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
    calculate_HU(ICRP_Tissues, Kph, Kcoh, KKN, flag="ICRP")
    
    # Step 5: Compute electron density for ICRP tissues
    print("\n=== Electron Density of ICRP Tissues ===")
    for material in ICRP_Tissues:
        temp = compute_rhoe_schneider(material, flag="ICRP")
        print(f"{material:<15} | Electron Density: {temp:.2f}")
        rhos_ICRP.append(temp)
    
    # Step 6: Compute SPR for ICRP tissues
    
schneider('/Users/royaparsa/Desktop/Gammex-Pelvis-1cm/CT1.3.12.2.1107.5.1.4.83775.30000024051312040257200013605.dcm', "body", 0.75)
# plot_true_vs_calculated_rhoe()
