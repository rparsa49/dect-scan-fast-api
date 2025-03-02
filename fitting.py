from scipy.optimize import minimize_scalar
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

# Saito 2017a Eq. 8 - LHS
def zeff_lhs(zeff):
    return (zeff / 7.45) ** 3.3

# Saito 2017a Eq. 8 - RHS
def zeff_rhs(gamma, ct, rho):
    return gamma * ((ct/rho) - 1)

# Hunemohr 2014 Eq. 21 - Effective Atomic Number
def zeff_hunemohr(n_i, Z_i, n=1):
    num = np.sum(n_i * (Z_i ** (n + 1)))
    den = np.sum(n_i * Z_i)
    return (num / den) ** (1 / n)

# True Mean Excitation Energy (Courtesy of Milo V.)
def i_truth(weight_fractions, Z_eff, A, I):
    return sum(weight_fractions * Z_eff / A * np.log(I)) / sum(weight_fractions * Z_eff / A)

# Tanaka 2020 Eq. 6 - Mean Excitation Energy
def i_tanaka(z_ratio, c0, c1):
    return c1 * (z_ratio - 1) - c0

# Tanaka 2020 Eq. 1 - Stopping Power
def spr_tanaka(rho, mean_excitation):
    '''
    rho: electron density ratio to water
    I, Iw: Mean excitation energies of the material and water
    me: rest electron mass
    c: speed of light in a vacuum
    beta: speed of the projectile proton relative to that of light
    '''
    pass

# Load circle data
CIRCLE_DATA = load_json("circles.json")
# Load material data
MATERIAL_PROPERTIES = load_json("material_properties.json")
# Loat atomic number data
ATOMIC_NUMBERS = load_json("atomic_numbers.json")
# Load elemental properties data
ELEMENTAL_PROPERTIES = load_json("element_properties.json")

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

# Optimize alpha to match true electron density using Saito 2017a eq. 1 and eq. 2
def optimize_alpha(HU_H, HU_L, true_rho):
    def objective(alpha):
        delta_hu = delta_HU(alpha, HU_H, HU_L)
        calculated_rho = rho_e(delta_hu)
        return abs(calculated_rho - true_rho)

    result = minimize_scalar(objective, bounds=(0, 1), method="bounded")
    return result.x  # Optimal alpha

# Optimize a and b to match true electron density using Saito 2012 eq. 4
def optimize_a_b(delta_HU, true_rho):
    def objective(params):
        a, b = params
        calculated_rho = rho_e_calc(delta_HU, a, b)
        return abs(calculated_rho - true_rho)

    initial_guess = [1, 1]
    result = minimize(objective, initial_guess, method="Nelder-Mead")
    return result.x  # Optimized [a, b]

# Calculate Z_eff using Hunemohr 2014 eq. 21
def calculate_z_eff_hunemohr(material):
    composition = MATERIAL_PROPERTIES[material]["composition"]

    elements = list(composition.keys())
    fractions = np.array([composition[el] for el in elements])
    atomic_numbers = np.array([ATOMIC_NUMBERS[el] for el in elements])

    z_eff = zeff_hunemohr(fractions, atomic_numbers)
    return z_eff

# Optimize gamma to match true effective atomic number using Saito 2017a eq. 8
def calculate_optimized_gamma(ct, rho, z_eff):
    lhs = zeff_lhs(z_eff)
    
    def objective(gamma):
        rhs = zeff_rhs(gamma, ct, rho)
        return (lhs - rhs)**2
    
    result = minimize_scalar(objective, bounds=(0,20), method="bounded")
    return result.x

# Minimize the difference between calculated and true Z_eff
def optimize_n_for_hunemohr(fractions, atomic_numbers, true_z_eff):
    def objective(n):
        calculated_z_eff = zeff_hunemohr(fractions, atomic_numbers, n)
        return (calculated_z_eff - true_z_eff)**2  # Squared error

    result = minimize_scalar(objective, bounds=(0.5, 3), method="bounded")

    optimal_n = result.x
    return zeff_hunemohr(fractions, atomic_numbers, optimal_n), optimal_n

def calculate_optimized_z_eff_hunemohr(material, true_z_eff):
    composition = MATERIAL_PROPERTIES[material]["composition"]

    elements = list(composition.keys())
    fractions = np.array([composition[el] for el in elements])
    atomic_numbers = np.array([ATOMIC_NUMBERS[el] for el in elements])

    z_eff, optimal_n = optimize_n_for_hunemohr(
        fractions, atomic_numbers, true_z_eff)
    return z_eff, optimal_n

# Optimize c0 and c1 to match the true mean excitation energies
def optimize_c(ionization, z_ratio):
    def objective(params):
        c0, c1 = params
        calc_i = i_tanaka(z_ratio, c0, c1)
        return (calc_i - ionization) ** 2
       
    initial_guess = [100, 50]
    result = minimize(objective, initial_guess, method = 'Nelder-Mead')
    return result.x

# Load DICOM images
low_path = '/Users/royaparsa/Downloads/high-low/1.3.12.2.1107.5.1.4.83775.30000024051312040257200015808/test.dcm'  # 140 KVP
high_path = '/Users/royaparsa/Downloads/high-low/1.3.12.2.1107.5.1.4.83775.30000024051312040257200019235/test.dcm'  # 70 KVP

dicom_data_h = pydicom.dcmread(high_path)
dicom_data_l = pydicom.dcmread(low_path)

high_image = dicom_data_h.pixel_array
low_image = dicom_data_l.pixel_array

# Process head phantom
saved_circles = CIRCLE_DATA["head"]
materials_list = []

calculated_rhos, true_rhos = [], []
optimized_alphas = []
optimized_as, optimized_bs = [], []
calculated_z_effs, true_z_effs = [], []
optimized_gammas = []
true_z_ratios, calculated_z_ratios = [], []
true_mean_excitation, calculated_mean_excitation = [], []

for circle in saved_circles:
    x, y, radius, material = circle["x"], circle["y"], circle["radius"], circle["material"]

    if material not in TRUE_RHO:
        print(f"Warning: Material '{material}' not found in TRUE_RHO.")
        continue

    true_rho = TRUE_RHO[material]
    true_z_eff = MATERIAL_PROPERTIES[material]["Z_eff"]
    
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
    
    # Step 5: Calculate optimized effective atomic numbers using material properties
    calculated_z_eff, optimal_n = calculate_optimized_z_eff_hunemohr(material, true_z_eff)
    calculated_z_effs.append(calculated_z_eff)
    true_z_effs.append(true_z_eff)

    print(f"Material: {material}, Optimal n: {optimal_n:.4f}, Calculated Z_eff: {calculated_z_eff:.4f}, True Z_eff: {true_z_eff:.4f}")

    # Step 6: Optimize gamma for Z_eff_w ratio
    true_z_ratio = zeff_lhs(calculated_z_eff)
    true_z_ratios.append(true_z_ratio)
    reduced_ct = reduce_ct(mean_low_hu)
    optimal_gamma = calculate_optimized_gamma(reduced_ct, calculated_rho, calculated_z_eff)
    optimized_gammas.append(optimal_gamma)
    
    z_ratio = zeff_rhs(optimal_gamma, reduced_ct, calculated_rho)
    calculated_z_ratios.append(z_ratio)
    
    # Step 7: Calculate mean excitation energy
    composition = MATERIAL_PROPERTIES[material]["composition"]
    elements = list(composition.keys())
    weight_fractions = np.array([composition[e] for e in elements])
    
    atomic_numbers = np.array([ELEMENTAL_PROPERTIES[e]["number"] for e in elements])
    atomic_masses = np.array([ELEMENTAL_PROPERTIES[e]["mass"] for e in elements])
    ionization_energies = np.array([ELEMENTAL_PROPERTIES[e]["ionization"] for e in elements])
    
    true_mean_i = i_truth(weight_fractions, atomic_numbers, atomic_masses, ionization_energies)
    
    true_mean_excitation.append(true_mean_i)
    
    # Step 8: Optimize c0 and c1 for mean excitation energy using Tanaka 2020 eq. 6
    c0, c1 = optimize_c(true_mean_i, z_ratio)
    i_tanaka_val = i_tanaka(z_ratio, c0, c1)
    calculated_mean_excitation.append(i_tanaka_val)
    
    # Step 9: Calculate Stopping Power!
    


# Plot Z_eff
# plt.figure(figsize=(12, 6))
# plt.scatter(materials_list, true_z_effs,
#             label="Hunemohr True Z_eff", color='green', marker='o')
# plt.scatter(materials_list, calculated_z_effs,
#             label="Calculated Z_eff", color='orange', marker='x')
# plt.xlabel("Material")
# plt.ylabel("Z_eff")
# plt.title("Comparison of True and Calculated Z_eff for Each Insert (Hunemohr)")
# plt.xticks(rotation=45)
# plt.legend()
# plt.grid()
# plt.show()

# # Convert to numpy arrays for plotting
# calculated_rhos = np.array(calculated_rhos)
# true_rhos = np.array(true_rhos)

# # Calculate (rho_e_measured / rho_e_truth) - 1
# rho_ratio_error = (calculated_rhos / true_rhos) - 1

# # Plot (rho_e_measured / rho_e_truth) - 1 vs rho_e_truth
# plt.figure(figsize=(10, 6))
# plt.scatter(true_rhos, rho_ratio_error, color='teal', marker='o')

# plt.axhline(0, color='gray', linestyle='--', linewidth=1)  # Reference line at zero
# plt.xlabel("True Electron Density (ρe)")
# plt.ylabel("(ρe_measured / ρe_truth) - 1")
# plt.title("Relative Error in Electron Density vs. True Electron Density")
# plt.grid(True)
# plt.show()

# # Convert to numpy arrays for plotting
# calculated_z_effs = np.array(calculated_z_effs)
# true_z_effs = np.array(true_z_effs)

# # Calculate (calculated_z_effs / true_z_effs) - 1
# z_eff_ratio_error = (calculated_z_effs / true_z_effs) - 1

# # Plot (z_eff_measured / z_eff_truth) - 1 vs z_eff_truth
# plt.figure(figsize=(10, 6))
# plt.scatter(true_z_effs, z_eff_ratio_error, color='teal', marker='o')

# plt.axhline(0, color='gray', linestyle='--', linewidth=1)  # Reference line at zero
# plt.xlabel("True Effective Atomic Number (Zeff)")
# plt.ylabel("(zeff_measured / zeff_truth) - 1")
# plt.title("Relative Error in Effective Atomic Number vs. True Effective Atomic Number")
# plt.grid(True)
# plt.show()

# # Convert to numpy arrays for plotting
# calculated_z_ratios = np.array(calculated_z_ratios)
# true_z_ratios = np.array(true_z_ratios)

# # Calculate (z_eff_ratio_measured / z_eff_ratio_truth) - 1
# z_ratio_error = (calculated_z_ratios / true_z_ratios) - 1

# # Plot (z_eff_ratio_measured / z_eff_ratio_truth) - 1 vs z_eff_ratio_truth
# plt.figure(figsize=(10, 6))
# plt.scatter(true_z_ratios, z_ratio_error, color='teal', marker='o')

# plt.axhline(0, color='gray', linestyle='--', linewidth=1)  # Reference line at zero
# plt.xlabel("True Z Ratio")
# plt.ylabel("(calculated_z_ratios / true_z_ratios) - 1")
# plt.title("Relative Error in Calculated Z Ratio vs True Z Ratio")
# plt.grid(True)
# plt.show()
# Plotting
# plt.figure(figsize=(12, 6))
# plt.scatter(materials_list, true_mean_excitation,label='True Mean Excitation Energy (I_truth)', color='green', marker='o')
# plt.scatter(materials_list, calculated_mean_excitation, label='Calculated Mean Excitation Energy (Tanaka)', color='orange', marker='x')

# plt.xlabel('Material')
# plt.ylabel('Mean Excitation Energy (eV)')
# plt.title('Comparison of True and Calculated Mean Excitation Energy')
# plt.xticks(rotation=45, ha='right')
# plt.legend()
# plt.grid(True)
# plt.show()

calculated_mean_excitation = np.array(calculated_mean_excitation)
true_mean_excitation = np.array(true_mean_excitation)
i_ratio = (calculated_mean_excitation / true_mean_excitation) - 1
plt.figure(figsize=(10, 6))
plt.scatter(true_mean_excitation, i_ratio, color='teal', marker='o')

plt.axhline(0, color='gray', linestyle='--', linewidth=1)  # Reference line at zero
plt.xlabel("True Mean Excitation")
plt.ylabel("(calculated_mean_excitation / true_mean_excitation) - 1")
plt.title("Relative Error in Calculated Mean Excitation vs True Mean Excitation")
plt.grid(True)
plt.show()

# Print optimized a and b
for mat, a, b in zip(materials_list, optimized_as, optimized_bs):
    print(f"Material: {mat}, Optimized a: {a:.4f}, Optimized b: {b:.4f}")
