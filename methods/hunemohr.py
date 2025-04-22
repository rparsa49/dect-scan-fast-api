import numpy as np
from pathlib import Path
import json
from scipy.optimize import minimize_scalar
import pydicom
import cv2
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from scipy.constants import physical_constants
from scipy.optimize import curve_fit

DATA_DIR = Path("data")

def load_json(file_name):
    with open(DATA_DIR / file_name, "r") as file:
        return json.load(file)

WATER_SPR = load_json("water_sp.json")
CIRCLE_DATA = load_json("circles.json")
MATERIAL_PROPERTIES = load_json("material_properties.json")
ATOMIC_NUMBERS = load_json("atomic_numbers.json")
ELEMENTAL_PROPERTIES = load_json("element_properties.json")  

# True electron densities and Zeffs for materials
TRUE_RHO = {mat: MATERIAL_PROPERTIES[mat]["rho_e_w"]for mat in MATERIAL_PROPERTIES}
TRUE_ZEFF = {mat: MATERIAL_PROPERTIES[mat]["Z_eff"]for mat in MATERIAL_PROPERTIES}

# Hunemohr Functions
def rho_e_hunemohr(HU_h, HU_l, c):
    '''
    Hunemohr 2014
    HU_h: CT Number at High Energy
    HU_l: CT Number at Low Energy
    c: calibration parameter
    '''
    return c * (HU_h/1000 + 1) + (1 - c) * (HU_l/1000) + 1

def z_eff_hunemohr(n_i, Z_i, n=3.1):
    num = np.sum(n_i * (Z_i ** (n + 1)))
    den = np.sum(n_i * Z_i)
    return (num / den) ** (1 / n)

# def spr_hunemohr(rho, Z, kvp):
#     '''
#     Hunemohr 2014
#     rho: Electron density ratio of the material to water
#     Z: Effective atomic number
#     '''
#     a = 0.125 if Z <= 8.5 else 0.098
#     b = 3.378 if Z <= 8.5 else 3.376
#     return (rho * ((12.77 - (a * Z + b)) / 8.45)) / WATER_SPR.get(str(kvp))

def beta(kvp=200):
    kinetic_energy_mev = kvp / 1000
    proton_mass_mev = physical_constants['proton mass energy equivalent in MeV'][0]
    gamma = (proton_mass_mev + kinetic_energy_mev) / proton_mass_mev
    return np.sqrt(1 - (1 / gamma ** 2)) ** 2

# Get I_material from ln I / Iw
def get_I(mean_exciation):
    return 75 * (np.e ** mean_exciation)

def spr_bethe(rho, I, beta):
    '''
    rho: electron density ratio to water
    I, Iw: Mean excitation energies of the material and water
    me: rest electron mass
    c: speed of light in a vacuum
    beta: speed of the projectile proton relative to that of light
    '''
    me = 9.10938356e-31
    c = 2.99792458e8
    Iw = 75
    
    term1 = (np.log(2*me * (c ** 2) * beta)) / (I - beta) - beta
    term2 = (np.log(2*me * (c ** 2) * beta)) / (Iw - beta) - beta
    return rho * (term1/term2)

# True Mean Excitation Energy (Courtesy of Milo V.)
def i_truth(weight_fractions, Num, A, I):
    return sum(weight_fractions * Num / A * np.log(I)) / sum(weight_fractions * Num / A)

# Fitting Functions
def optimize_c(HU_H_List, HU_L_List, true_rho_list, materials_list):
    def objective(c):
        errors = []
        for HU_H, HU_L, mat in zip(HU_H_List, HU_L_List, materials_list):
            if mat in true_rho_list:
               true_rho = true_rho_list[mat]
               estimated_rho = rho_e_hunemohr(HU_H, HU_L, c)
               errors.append(abs(estimated_rho - true_rho))
        return sum(errors)
    return minimize_scalar(objective, bounds=(0,1), method="bounded").x

'''
True values of Z_eff to use in fitting
'''
def calculate_zeff_hunemohr(material):
    composition = MATERIAL_PROPERTIES[material]["composition"]

    elements = list(composition.keys())
    fractions = np.array([composition[el] for el in elements])
    atomic_numbers = np.array([ATOMIC_NUMBERS[el] for el in elements])
    
    return z_eff_hunemohr(fractions, atomic_numbers, 3.1)

'''
Fitting Z_eff
'''
def z_eff_model(X, d_e, n=3.1):
    rho_e, zeff_w, x1, x2 = X.T
    factor = (rho_e) ** -1
    term1 = d_e * ((x1 / 1000) + 1)
    term2 = (zeff_w ** n -d_e) * ((x2 / 1000) + 1)
    inner = factor * (term1 + term2)
    return inner ** (1/n)

def fit_zeff(rho_e, zeff_w, true_zeff, x1, x2):
    rho_e = np.asarray(rho_e)
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    zeff_w_arr = np.full_like(rho_e, zeff_w)

    X = np.column_stack((rho_e, zeff_w_arr, x1, x2))
    popt, _ = curve_fit(lambda X, d_e: z_eff_model(X, d_e), X, true_zeff)
    return popt[0]

def calculate_zeff_optimized(rho_e, zeff_w, x1, x2, d_e, n=3.1):
    factor = (rho_e) ** -1
    term1 = d_e * ((x1 / 1000) + 1)
    term2 = (zeff_w ** n -d_e) * ((x2 / 1000) + 1)
    inner = factor * (term1 + term2)
    return inner ** (1/n)

def hunemohr(high_path, low_path, phantom_type, radii_ratios):
    dicom_data_h = pydicom.dcmread(high_path)
    dicom_data_l = pydicom.dcmread(low_path)
    
    high_image = dicom_data_h.pixel_array
    low_image = dicom_data_l.pixel_array
    
    HU_H_List, HU_L_List, materials_list = [], [], []
    calculated_rhos, calculated_zeffs = [], []
    optimized_zs = []
    sprs = []
    mean_excitations = []
    c = 0
    SAVED_CIRCLES = CIRCLE_DATA[phantom_type]

    for circle in SAVED_CIRCLES:
        x, y, radius, material = circle["x"], circle["y"], circle["radius"], circle["material"]
        if material not in TRUE_RHO or material == '50% CaCO3' or material == '30% CaCO3':
            print(f"Warning: Material '{material}' not found in TRUE_RHO.")
            continue
        
        materials_list.append(material)
        
        # Mask for circular region
        mask = np.zeros(high_image.shape, dtype=np.uint8)
        cv2.circle(mask, (x, y), int(radius * radii_ratios), 1, thickness=-1)

        high_pixel_values = high_image[mask == 1]
        low_pixel_values = low_image[mask == 1]

        mean_high_hu = np.mean(high_pixel_values) * \
            dicom_data_h.RescaleSlope + dicom_data_h.RescaleIntercept
        mean_low_hu = np.mean(low_pixel_values) * \
            dicom_data_l.RescaleSlope + dicom_data_l.RescaleIntercept

     # Create HU lists
        HU_H_List.append(mean_high_hu)
        HU_L_List.append(mean_low_hu)
    
    # Step 1: Optimize C for rho_e calculation
    c = optimize_c(HU_H_List, HU_L_List, TRUE_RHO, materials_list)
    print(f"Optimized C: {c}")
    
    print("\n")
    
    # Step 2: Calculate rho using c
    for hu_h, hu_l in zip(HU_H_List, HU_L_List):
        rho = rho_e_hunemohr(hu_h, hu_l, c)
        calculated_rhos.append(rho)
    
    # Step 3: Calculate true zeff
    for mat in materials_list:
        calculated_z = calculate_zeff_hunemohr(mat)
        calculated_zeffs.append(calculated_z)
    
    # Step 4: Calculate optimized zeff
    zeff_w = calculate_zeff_hunemohr("True Water")
    # zeff_w = 7
    d_e = fit_zeff(calculated_rhos, zeff_w, calculated_zeffs, HU_H_List, HU_L_List)
    for rhos, x1, x2 in zip(calculated_rhos, HU_H_List, HU_L_List):
        opt_z = calculate_zeff_optimized(rhos, zeff_w, x1,  x2, d_e)
        optimized_zs.append(opt_z)
    
    # Step 6: Calculate Mean Excitation Energy
    for mat in materials_list:
        comp = MATERIAL_PROPERTIES[mat]["composition"]
        elements = list(comp.keys())
        fraction = np.array([comp[e] for e in elements])

        atomic_numbers = np.array(
            [ELEMENTAL_PROPERTIES[e]["number"] for e in elements])
        atomic_masses = np.array(
            [ELEMENTAL_PROPERTIES[e]["mass"] for e in elements])
        ionization_energies = np.array(
            [ELEMENTAL_PROPERTIES[e]["ionization"] for e in elements])

        i = i_truth(fraction, atomic_numbers,
                    atomic_masses, ionization_energies)
        mean_excitations.append(i)
        
    # Step 4: Stopping power
    for i, rho, mat in zip(mean_excitations, calculated_rhos, materials_list):
        I = get_I(i)
        beta2 = beta()
        spr = spr_bethe(rho, I, beta2)
        sprs.append(spr)
    # for rho, z in zip(calculated_rhos, calculated_zeffs):
    #     spr = spr_hunemohr(rho, z, dicom_data_h.KVP)
    #     sprs.append(spr)
    
    for mat, rho in zip(materials_list, calculated_rhos):
        print(f"Material: {mat} with electron density of {rho}")
        
    print("\n")
    
    for mat, z in zip(materials_list, optimized_zs):
        print(f"Material: {mat} with Z of {z}")
    
    print("\n")
    
    for mat, spr in zip(materials_list, sprs):
        print(f"Material: {mat} with SPR of {spr}")  
    
    ground_rho = []
    for mat in materials_list:
        ground_rho.append(MATERIAL_PROPERTIES[mat]["rho_e_w"])        
    rmse_rho = mean_squared_error(ground_rho, calculated_rhos)
    r2_rho = r2_score(ground_rho, calculated_rhos)
    percent_rho = mean_absolute_percentage_error(ground_rho, calculated_rhos)

    ground_z = []
    for mat in materials_list:
        ground_z.append(MATERIAL_PROPERTIES[mat]["Z_eff"])    
    rmse_z = mean_squared_error(ground_z, optimized_zs)
    r2_z = r2_score(ground_z, optimized_zs)
    percent_z = mean_absolute_percentage_error(ground_z, optimized_zs)

    # Return JSON
    results = {
        "materials": materials_list,
        "calculated_rhos": calculated_rhos,
        "calculated_z_effs": optimized_zs,
        "stopping_power": sprs,
        "error_metrics": {
            "rho": {"RMSE": rmse_rho, "R2": r2_rho, "PercentError": percent_rho},
            "z": {"RMSE": rmse_z, "R2": r2_z, "PercentError": percent_z}
        }
    }
    
    return json.dumps(results, indent=4)
