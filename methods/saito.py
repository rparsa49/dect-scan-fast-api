import numpy as np
from pathlib import Path
import json
from scipy.optimize import minimize_scalar, minimize
import sys
import pydicom
import cv2
from scipy.constants import physical_constants
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

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
RHO_W = 3.342801000466205e+23

def delta_HU(alpha, high, low):
    '''
    Saito 2012
    alpha: weighing factor
    high: CT high
    low: CT low
    
    return: Delta HU
    '''
    return (1+alpha)*high-(alpha*low)

def rho_e_saito(HU, a, b):
    '''
    Saito 2012
    HU: Delta HU
    alpha: weighing factor
    b: line intercept
    
    return: calibrated rho (electron density)
    '''
    return a*(HU/1000)+b

def rho_e(HU):
    return HU / 1000 + 1

def reduce_ct(HU):
    '''
    Saito 2017
    HU: CT number at either high or low energy
    
    return: linear attenuation coefficient
    '''
    return HU/1000 + 1

# Saito 2017a Eq. 8 - LHS
def zeff_lhs(zeff):
    return (zeff / 7.45) ** 3.3

# Saito 2017a Eq. 8 - RHS
def zeff_rhs(gamma, ct, rho):
    return gamma * ((ct/rho) - 1)

# Tanaka 2020 Eq. 1 - Stopping Power
def spr_tanaka(rho, I, beta):
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

    term1 = np.log(I/Iw)
    term2 = np.log((2 * me * c ** 2 * beta ** 2) / (Iw * (1 - beta ** 2)))
    return rho * (1 - (term1 / (term2 - beta ** 2)))

# True Mean Excitation Energy (Courtesy of Milo V.)
def i_truth(weight_fractions, Num, A, I):
    return sum(weight_fractions * Num / A * np.log(I)) / sum(weight_fractions * Num / A)

def beta(kvp):
    kinetic_energy_mev = kvp / 1000
    proton_mass_mev = physical_constants['proton mass energy equivalent in MeV'][0]
    gamma = (proton_mass_mev + kinetic_energy_mev) /  proton_mass_mev
    return np.sqrt(1 - (1 / gamma ** 2))

# Get I_material from ln I / Iw
def get_I(mean_exciation):
    return 75 * (np.e ** mean_exciation)

# Fitting functions
def optimize_alpha(HU_H_LIST, HU_L_LIST, true_rho_list, materials_list):
    best_r2 = 0
    best_alpha = None
    best_a = None
    best_b = None

    alphas = np.linspace(0, 1, 10000)  # Fine granularity

    for alpha in alphas:
        true_rhos = []
        deltas = []

        for HU_H, HU_L, material in zip(HU_H_LIST, HU_L_LIST, materials_list):
            if material in true_rho_list:
                # delta_HU = ((1 + alpha) * HU_H) - (alpha * HU_L)
                delta = delta_HU(alpha, HU_H, HU_L)
                deltas.append(delta / 1000)  # acts as x
                true_rhos.append(true_rho_list[material])  # acts as y

        # Linear fit: rho_e_cal = a * (delta_HU / 1000) + b
        x = np.array(deltas).reshape(-1, 1)
        y = np.array(true_rhos)
        model = LinearRegression().fit(x, y)
        y_pred = model.predict(x)
        r2 = r2_score(y, y_pred)

        if r2 > best_r2:
            best_r2 = r2
            best_alpha = alpha
            best_a = model.coef_[0]
            best_b = model.intercept_

    return best_alpha, best_a, best_b, best_r2

def optimize_gamma(zeff_list, ct_list, rho_list):
    def objective(gamma):
        errors = []
        for zeff, ct, rho in zip(zeff_list, ct_list, rho_list):
            lhs = zeff_lhs(zeff)
            rhs = zeff_rhs(gamma, ct, rho)
            errors.append(abs(lhs - rhs))
        return sum(errors)

    result = minimize_scalar(objective, bounds=(0, 10), method="bounded")
    return result.x

def saito(high_path, low_path, phantom_type, radii_ratios):
    dicom_data_h = pydicom.dcmread(high_path)
    dicom_data_l = pydicom.dcmread(low_path)

    high_image = dicom_data_h.pixel_array
    low_image = dicom_data_l.pixel_array

    HU_H_List, HU_L_List, materials_list = [], [], []
    calculated_rhos = []
    mean_excitations = []
    sprs = []
    deltas = []
    alpha, a, b = 0, 0, 0
    saved_circles = CIRCLE_DATA[phantom_type]
    
    for circle in saved_circles:
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
        
    print(f"High HU List: {HU_H_List}\n")
    print(f"Low HU List: {HU_L_List}\n")

    # Step 1: Optimize alpha, a, b for delta HU        
    alpha, a, b, r = optimize_alpha(HU_H_List, HU_L_List, TRUE_RHO, materials_list)
    print(f"Alpha: {alpha}\n a: {a}\n b: {b}\n r: {r}\n")
    
    deltas = []
    for HU_H, HU_L in zip(HU_H_List, HU_L_List):
        delta = ((1 + alpha) * HU_H) - (alpha * HU_L)
        deltas.append(delta)
    
    # Step 2: Calculate rho
    for delta in deltas:
        rho = rho_e_saito(delta, a, b)
        calculated_rhos.append(rho)
        
    for mat, rho in zip(materials_list, calculated_rhos):
        print(f"Material: {mat} with electron density of {rho}")
    
    # Step 3: Calculate reduced CT
    reduced_ct = [reduce_ct(hl) for hl in HU_L_List]
    
    # Step 4: Optimize gamma using true Zeff and estimated rho
    zeff_list = [TRUE_ZEFF[mat] for mat in materials_list]
    gamma = optimize_gamma(zeff_list, reduced_ct, calculated_rhos)
        
    # Step 5: Calculate estimated Zeff
    calculated_zeffs = [(np.abs(zeff_rhs(gamma, ct, rho)) ** (1/3.3) * 7.45)for ct, rho in zip(reduced_ct, calculated_rhos)]

    for mat, z in zip(materials_list, calculated_zeffs):
        print(f"Material: {mat}'s calculated Z: {z}")
    
    # Step 6: Calculate Mean Excitation Energy
    for mat in materials_list:
        comp = MATERIAL_PROPERTIES[mat]["composition"]
        elements = list(comp.keys())
        fraction = np.array([comp[e] for e in elements])
        
        atomic_numbers = np.array([ELEMENTAL_PROPERTIES[e]["number"] for e in elements])
        atomic_masses = np.array([ELEMENTAL_PROPERTIES[e]["mass"] for e in elements])
        ionization_energies = np.array([ELEMENTAL_PROPERTIES[e]["ionization"] for e in elements])

        i = i_truth(fraction, atomic_numbers, atomic_masses, ionization_energies)
        mean_excitations.append(i)

    # Step 7: Calculate SPR
    for i, rho, mat in zip(mean_excitations, calculated_rhos, materials_list):
        I = get_I(i)
        beta2 = beta(200)
        spr = spr_tanaka(rho, I, beta2)
        sprs.append(spr)
        
    # Step 8: Calculate error
    ground_rho = []
    for mat in materials_list:
        ground_rho.append(MATERIAL_PROPERTIES[mat]["rho_e_w"])
    rmse_rho = mean_squared_error(ground_rho, calculated_rhos)
    r2_rho = r2_score(ground_rho, calculated_rhos)
    print(f"RMSE for Rho: {rmse_rho}")
    
    ground_z = []
    for mat in materials_list:
        ground_z.append(MATERIAL_PROPERTIES[mat]["Z_eff"])
    rmse_z = mean_squared_error(ground_z, calculated_zeffs)
    r2_z = r2_score(ground_z, calculated_zeffs)
    print(f"RMSE for Z: {rmse_z}")
    
    print(f"R2 for lin reg {r}\n\n")
    
    # Return JSON
    results = {
        "materials": materials_list,
        "calculated_rhos": calculated_rhos,
        "calculated_z_effs": calculated_zeffs,
        "stopping_power": sprs,
        "alpha": alpha,
        "a": a,
        "b": b,
        "r": r,
        "gamma": gamma,
        "error_metrics": {
            "rho": {"RMSE": rmse_rho, "R2": r2_rho},
            "z": {"RMSE": rmse_z, "R2": r2_z}
        }
    }     
    
    # return results
    
    return json.dumps(results, indent=4)
