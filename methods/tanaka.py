from scipy.optimize import minimize_scalar
import pydicom
import cv2
import json
import numpy as np
from scipy.optimize import minimize_scalar, minimize
from pathlib import Path
from scipy.constants import physical_constants
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

DATA_DIR = Path("data")

def load_json(file_name):
    with open(DATA_DIR / file_name, "r") as file:
        return json.load(file)

# Load circle data
CIRCLE_DATA = load_json("circles.json")
# Load material data
MATERIAL_PROPERTIES = load_json("material_properties.json")
# Loat atomic number data
ATOMIC_NUMBERS = load_json("atomic_numbers.json")
# Load elemental properties data
ELEMENTAL_PROPERTIES = load_json("element_properties.json")

# True electron densities (ρe) for materials
TRUE_RHO = {mat: MATERIAL_PROPERTIES[mat]["rho_e_w"]
            for mat in MATERIAL_PROPERTIES}
TRUE_ZEFF = {mat: MATERIAL_PROPERTIES[mat]["Z_eff"]
             for mat in MATERIAL_PROPERTIES}

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
    v = gamma * ((ct/rho) - 1)
    print(v)
    return gamma * ((ct/rho) - 1)

# Hunemohr 2014 Eq. 21 - Effective Atomic Number
def zeff_hunemohr(n_i, Z_i, n=3.1):
    num = np.sum(n_i * (Z_i ** (n + 1)))
    den = np.sum(n_i * Z_i)
    return (num / den) ** (1 / n)

# True Mean Excitation Energy (Courtesy of Milo V.)
def i_truth(weight_fractions, Num, A, I):
    return sum(weight_fractions * Num / A * np.log(I)) / sum(weight_fractions * Num / A)

# Tanaka 2020 Eq. 6 - Mean Excitation Energy
def i_tanaka(z_ratio, c0, c1):
    return c1 * (z_ratio - 1) - c0

# Get I_material from ln I / Iw
def get_I(mean_exciation):
    return 75 * (np.e ** mean_exciation)

# Beta Proton (Courtesy of Milo)
def beta(kvp):
    kinetic_energy_mev = kvp / 1000
    proton_mass_mev = physical_constants['proton mass energy equivalent in MeV'][0]
    gamma = (proton_mass_mev + kinetic_energy_mev) /  proton_mass_mev
    return np.sqrt(1 - (1 / gamma ** 2))

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

# Optimize alpha to match true electron density using Saito 2017a eq. 1 and eq. 2
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

# Calculate Z_eff using Hunemohr 2014 eq. 21
def calculate_z_eff_hunemohr(material):
    composition = MATERIAL_PROPERTIES[material]["composition"]

    elements = list(composition.keys())
    fractions = np.array([composition[el] for el in elements])
    atomic_numbers = np.array([ATOMIC_NUMBERS[el] for el in elements])

    z_eff = zeff_hunemohr(fractions, atomic_numbers)
    return z_eff

# Optimize gamma to match true effective atomic number using Saito 2017a eq. 8
def calculate_optimized_gamma(ct_list, rho_list, z_eff_list):
    def objective(gamma):
        errors = [(zeff_lhs(z) - zeff_rhs(gamma, ct, rho)) ** 2 for ct, rho, z in zip(ct_list, rho_list, z_eff_list)]
        return sum(errors)
    
    result = minimize_scalar(objective, bounds=(0, 15), method="bounded")
    return result.x

# Minimize the difference between calculated and true Z_eff
def optimize_n_for_hunemohr(fractions, atomic_numbers, true_z_eff):
    def objective(n):
        calculated_z_eff = zeff_hunemohr(fractions, atomic_numbers, n)
        return (calculated_z_eff - true_z_eff)**2  # Squared error

    result = minimize_scalar(objective, bounds=(0.5, 3), method="bounded")

    optimal_n = 3
    return zeff_hunemohr(fractions, atomic_numbers, optimal_n)

def calculate_optimized_z_eff_hunemohr(material, true_z_eff_list):
    composition = MATERIAL_PROPERTIES[material]["composition"]
    true_z_eff = true_z_eff_list[material]
    elements = list(composition.keys())
    fractions = np.array([composition[el] for el in elements])
    atomic_numbers = np.array([ATOMIC_NUMBERS[el] for el in elements])

    z_eff = optimize_n_for_hunemohr(
        fractions, atomic_numbers, true_z_eff)
    return z_eff

# Optimize c0 and c1 to match the true mean excitation energies
def optimize_c(ionization_list, z_ratio_list):
    def objective(params):
        c0, c1 = params
        for i, z in zip(ionization_list, z_ratio_list):
            calc_i_list = [i_tanaka(z, c0, c1)]
            vals = [(calc_i - i) ** 2 for calc_i in calc_i_list]
            return sum(vals)
       
    initial_guess = [100, 50]
    result = minimize(objective, initial_guess, method = 'Nelder-Mead')
    return result.x

# Get SPR from Tanaka
def get_t_spr(material):
    if material == 'Lung':
        return 0.280
    if material == 'Adipose':
        return 0.947
    if material == 'Breast':
        return 0.973
    if material == 'Solid Water':
        return 0.997
    if material == 'Water':
        return 1.000
    if material == 'Brain':
        return 1.064
    if material == 'Liver':
        return 1.070
    if material == 'Inner Bone':
        return 1.088
    if material == '30% CaCO3':
        return 1.261
    if material == '50% CaCO3':
        return 1.427
    if material == 'Cortical Bone':
        return 1.621

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

def tanaka(high_path, low_path, phantom_type, radii_ratios):
    dicom_data_h = pydicom.dcmread(high_path)
    dicom_data_l = pydicom.dcmread(low_path)

    high_image = dicom_data_h.pixel_array
    low_image = dicom_data_l.pixel_array

    # Process head phantom
    saved_circles = CIRCLE_DATA[phantom_type]
    materials_list = []

    calculated_rhos = []
    calculated_z_effs = []
    true_z_ratios, calculated_z_ratios = [], []
    true_mean_excitation, calculated_mean_excitation = [], []
    sprs = []
    t_sprs = []

    reduced_cts = []

    alpha = 0
    a = 0
    b = 0
    gamma = 0
    c0, c1 = 0, 0

    HU_H_List, HU_L_List, delta_HU_list = [], [], []

    for circle in saved_circles:
        x, y, radius, material = circle["x"], circle["y"], circle["radius"], circle["material"]

        if material not in TRUE_RHO or material == '50% CaCO3' or material == '30% CaCO3':
            print(f"Warning: Material '{material}' not found in TRUE_RHO.")
            continue
        print(material)
    
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
    
    # Step 1: Get optimized alpha
    alpha, a, b, r = optimize_alpha(HU_H_List, HU_L_List, TRUE_RHO, materials_list)
    print(f"Alpha: {alpha}\n a: {a}\n b: {b}\n r: {r}\n")
    
    deltas = []
    for HU_H, HU_L in zip(HU_H_List, HU_L_List):
        delta = ((1 + alpha) * HU_H) - (alpha * HU_L)
        deltas.append(delta)

    # Step 2: Calculate rho
    for delta in deltas:
        rho = rho_e_calc(delta, a, b)
        calculated_rhos.append(rho)

    for mat, rho in zip(materials_list, calculated_rhos):
        print(f"Material: {mat} with electron density of {rho}")
    
    # Step 3: Calculate reduced CT
    reduced_ct = [reduce_ct(hl) for hl in HU_L_List]
    
    # Step 4: Optimize Gamma using true Z_Eff and estimated rho
    zeff_list = [TRUE_ZEFF[mat] for mat in materials_list]
    gamma = optimize_gamma(zeff_list, reduced_ct, calculated_rhos)
    print(f"Gamma is: {gamma}")
    
    # Step 5: Calculate estimated Z ratios
    calculated_z_ratios = [(zeff_rhs(gamma, ct, rho)) for ct, rho in zip(reduced_ct, calculated_rhos)]

    # Convert ratios to Z_eff
    calculated_z_effs = [(np.abs(zeff_rhs(gamma, ct, rho)) ** (1/3.3) * 7.45)for ct, rho in zip(reduced_ct, calculated_rhos)]

    # Step 5: Calculate True Mean Excitation Energy
    for mat in materials_list:
        comp = MATERIAL_PROPERTIES[mat]["composition"]
        elements = list(comp.keys())
        fraction = np.array([comp[e] for e in elements])
    
        atomic_numbers = np.array([ELEMENTAL_PROPERTIES[e]["number"] for e in elements])
        atomic_masses = np.array([ELEMENTAL_PROPERTIES[e]["mass"] for e in elements])
        ionization_energies = np.array([ELEMENTAL_PROPERTIES[e]["ionization"] for e in elements])

        i = i_truth(fraction, atomic_numbers, atomic_masses, ionization_energies)
        true_mean_excitation.append(i)

    print(f"True I: {true_mean_excitation}")
    print(f"Calculated Zs: {calculated_z_ratios}")

    # Step 6: Optimize c0 and c1 for mean excitation energy using Tanaka 2020 eq. 6
    c0, c1 = optimize_c(true_mean_excitation, calculated_z_ratios)

    print(f"C0: {c0} \nC1: {c1}")

    for z_ratio in calculated_z_ratios:
        i_tanaka_val = i_tanaka(z_ratio, c0, c1)
        calculated_mean_excitation.append(i_tanaka_val)

    print(f"Tanaka I: {calculated_mean_excitation}")
    # Step 7: Calculate stopping power
    for t, rho, mat in zip(calculated_mean_excitation, calculated_rhos, materials_list):
        I = get_I(t)
        beta2 = beta(dicom_data_l.KVP)
        spr = spr_tanaka(rho, I, beta2)
        sprs.append(spr) 
    
        tanaka_spr = get_t_spr(mat)
        t_sprs.append(tanaka_spr)
    
    ground_rho = []    
    for mat in materials_list:
        ground_rho.append(MATERIAL_PROPERTIES[mat]["rho_e_w"])
    rmse_rho = mean_squared_error(ground_rho, calculated_rhos)
    r2_rho = r2_score(ground_rho, calculated_rhos)
    
    ground_z = []
    for mat in materials_list:
        ground_z.append(MATERIAL_PROPERTIES[mat]["Z_eff"])
    rmse_z = mean_squared_error(ground_z, calculated_z_effs)
    r2_z = r2_score(ground_z, calculated_z_effs)

    # Return JSON
    results = {
        "materials": materials_list,
        "calculated_rhos": calculated_rhos,
        "calculated_z_effs": calculated_z_effs,
        "mean_excitations": calculated_mean_excitation,
        "stopping_power": sprs,
        "tanaka_stopping_power": t_sprs,
        "alpha": alpha,
        "a": a,
        "b": b,
        "r": r,
        "gamma": gamma,
        "c0": c0,
        "c1": c1, 
        "error_metrics": {
            "rho": {"RMSE": rmse_rho, "R2": r2_rho},
            "z": {"RMSE": rmse_z, "R2": r2_z}
        }
    }

    return json.dumps(results, indent=4)
