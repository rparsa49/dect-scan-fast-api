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
TRUE_RHO = {mat: MATERIAL_PROPERTIES[mat]
            ["rho_e_w"]for mat in MATERIAL_PROPERTIES}

# Hunemohr Functions


def rho_e_hunemohr(HU_h, HU_l, c):
    '''
    Hunemohr 2014
    HU_h: CT Number at High Energy
    HU_l: CT Number at Low Energy
    c: calibration parameter
    '''
    return c * (HU_h/1000 + 1) + (1 - c) * ((HU_l/1000) + 1)


def z_eff_hunemohr(n_i, Z_i, n=3.1):
    num = np.sum(n_i * (Z_i ** (n + 1)))
    den = np.sum(n_i * Z_i)
    return (num / den) ** (1 / n)


def beta(kvp=200):
    kinetic_energy_mev = kvp / 1000
    proton_mass_mev = physical_constants['proton mass energy equivalent in MeV'][0]
    gamma = (proton_mass_mev + kinetic_energy_mev) / proton_mass_mev
    return np.sqrt(1 - (1 / gamma ** 2)) ** 2

# Reference I values (Bär 2018)


def ref_I(material):
    '''
    Bragg Additivity Rule for the mean excitation energy of a compound
    ln I_med = Sum_j lambda_med, i, j * ln I_elem, j
    lambda_med, i, j = w_med,i,j * (Z_j / A_j) / (Z/A)_med,i
    (Z/A)_med,i = sum_j w_men,i,j * (Z_j / A_j)
    '''
    icru_key = "I_ICRU1993_eV"
    comp = MATERIAL_PROPERTIES[material]["composition"]
    elements = list(comp.keys())
    w = np.array([comp[e] for e in elements])

    # normalize
    w_sum = w.sum()
    w = w / w_sum

    # gather elemental data
    Z = np.array([ELEMENTAL_PROPERTIES[e]["number"] for e in elements])
    A = np.array([ELEMENTAL_PROPERTIES[e]["mass"] for e in elements])
    I_elem = np.array([ELEMENTAL_PROPERTIES[e][icru_key] for e in elements])

    # (Z/A)_med
    ZA_med = float(np.sum(w * (Z / A)))

    # Electron-fraction weights lambda_j
    lambda_arr = (w * (Z / A)) / ZA_med

    # BAR estimator for I_med
    ln_I_med = float(np.sum(lambda_arr * np.log(I_elem)))
    return float(np.exp(ln_I_med))

# Fit I


def fit_I(Zs, Is):
    Z_arr = np.array(Zs)
    lnI_arr = np.log(np.array(Is))

    # linear regression for coefficients
    coeffs = np.polyfit(Z_arr, lnI_arr, 1)
    a, b = coeffs[0], coeffs[1]

    return a, b

# Hunemohr 2014 eq. 2


def hunemohr_I(a, b, Z):
    return a * Z + b

# Hunemohr 2014 eq. 3


def spr_hunemohr(rho, I):
    return rho * ((12.77 - I) / 8.45)

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
    return minimize_scalar(objective, bounds=(0, 1), method="bounded").x


'''
True values of Z_eff to use in fitting
'''


def calculate_ref_zeff_hunemohr(material):
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
    term2 = (zeff_w ** n - d_e) * ((x2 / 1000) + 1)
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
    term2 = (zeff_w ** n - d_e) * ((x2 / 1000) + 1)
    inner = factor * (term1 + term2)
    return inner ** (1/n)


def hunemohr_test(high_path, low_path, phantom_type, radii_ratios, a, b, c):
    dicom_data_h = pydicom.dcmread(high_path)
    dicom_data_l = pydicom.dcmread(low_path)

    high_image = dicom_data_h.pixel_array
    low_image = dicom_data_l.pixel_array

    HU_H_List, HU_L_List, materials_list = [], [], []
    calculated_rhos, calculated_zeffs = [], []
    reference_I, calculated_I = [], []
    optimized_zs = []
    sprs = []
    SAVED_CIRCLES = CIRCLE_DATA[phantom_type]

    for circle in SAVED_CIRCLES:
        x, y, radius, material = circle["x"], circle["y"], circle["radius"], circle["material"]
        if material not in TRUE_RHO or material in materials_list:
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
    # Step 2: Calculate rho using c
    for hu_h, hu_l in zip(HU_H_List, HU_L_List):
        rho = rho_e_hunemohr(hu_h, hu_l, c)
        calculated_rhos.append(rho)

    # Step 3: Calculate true zeff
    for mat in materials_list:
        calculated_z = calculate_ref_zeff_hunemohr(mat)
        calculated_zeffs.append(calculated_z)

    # Step 4: Calculate optimized zeff
    zeff_w = calculate_ref_zeff_hunemohr("True Water")
    # zeff_w = 7
    d_e = fit_zeff(calculated_rhos, zeff_w,
                   calculated_zeffs, HU_H_List, HU_L_List)
    for rhos, x1, x2 in zip(calculated_rhos, HU_H_List, HU_L_List):
        opt_z = calculate_zeff_optimized(rhos, zeff_w, x1,  x2, d_e)
        optimized_zs.append(opt_z)

    # Step 5: Calculate Reference I and get fitted a, b
    for material in materials_list:
        I_ref = ref_I(material)
        reference_I.append(I_ref)

    # Step 6: Calculate Mean Excitation Energy
    for Z in optimized_zs:
        I_optimized = hunemohr_I(a, b, Z)
        calculated_I.append(I_optimized)

    # Step 4: Stopping power
    for i, rho in zip(calculated_I, calculated_rhos):
        spr_calc = spr_hunemohr(rho, i)
        sprs.append(spr_calc)

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


def hunemohr(high_path, low_path, phantom_type, radii_ratios):
    dicom_data_h = pydicom.dcmread(high_path)
    dicom_data_l = pydicom.dcmread(low_path)

    high_image = dicom_data_h.pixel_array
    low_image = dicom_data_l.pixel_array

    HU_H_List, HU_L_List, materials_list = [], [], []
    calculated_rhos, calculated_zeffs = [], []
    reference_I, calculated_I = [], []
    a = 0
    b = 0
    optimized_zs = []
    sprs = []
    c = 0
    SAVED_CIRCLES = CIRCLE_DATA[phantom_type]

    for circle in SAVED_CIRCLES:
        x, y, radius, material = circle["x"], circle["y"], circle["radius"], circle["material"]
        if material not in TRUE_RHO or material in materials_list:
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
        calculated_z = calculate_ref_zeff_hunemohr(mat)
        calculated_zeffs.append(calculated_z)

    # Step 4: Calculate optimized zeff
    zeff_w = calculate_ref_zeff_hunemohr("True Water")
    # zeff_w = 7
    d_e = fit_zeff(calculated_rhos, zeff_w,
                   calculated_zeffs, HU_H_List, HU_L_List)
    for rhos, x1, x2 in zip(calculated_rhos, HU_H_List, HU_L_List):
        opt_z = calculate_zeff_optimized(rhos, zeff_w, x1,  x2, d_e)
        optimized_zs.append(opt_z)

    # Step 5: Calculate Reference I and get fitted a, b
    for material in materials_list:
        I_ref = ref_I(material)
        reference_I.append(I_ref)

    a, b = fit_I(optimized_zs, reference_I)

    # Step 6: Calculate Mean Excitation Energy
    for Z in optimized_zs:
        I_optimized = hunemohr_I(a, b, Z)
        calculated_I.append(I_optimized)

    # Step 4: Stopping power
    for i, rho in zip(calculated_I, calculated_rhos):
        spr_calc = spr_hunemohr(rho, i)
        sprs.append(spr_calc)

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
        "a": a,
        "b": b,
        "c": c,
        "error_metrics": {
            "rho": {"RMSE": rmse_rho, "R2": r2_rho, "PercentError": percent_rho},
            "z": {"RMSE": rmse_z, "R2": r2_z, "PercentError": percent_z}
        }
    }

    return json.dumps(results, indent=4)
