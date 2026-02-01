import os
import json
import logging
import pydicom
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.constants import physical_constants

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = Path("data")

def load_json(file_name):
    with open(DATA_DIR / file_name, "r") as file:
        return json.load(file)

WATER_SPR = load_json("water_sp.json")
CIRCLE_DATA = load_json("circles.json")
MATERIAL_PROPERTIES = load_json("material_properties.json")
ATOMIC_NUMBERS = load_json("atomic_numbers.json")
ELEMENTAL_PROPERTIES = load_json("element_properties.json")

# True electron densities
TRUE_RHO = {mat: MATERIAL_PROPERTIES[mat]["rho_e_w"]
            for mat in MATERIAL_PROPERTIES}

TRUE_SPR_VALUES = {
    "Cortical Bone": 1.711, "Breast": 0.979, "Adipose": 0.963,
    "30% CaCO3": 1.254, "LN-350": 0.280, "LN-450": 0.435,
    "Liver": 1.062, "Inner Bone": 1.152, "50% CaCO3": 1.429,
    "Brain": 1.032, "Solid Water": 0.997, "Water": 1.000
}

def calculate_bethe_constants(energy_mev=175):
    proton_mass_mev = physical_constants['proton mass energy equivalent in MeV'][0]
    gamma = (proton_mass_mev + energy_mev) / proton_mass_mev
    beta_sq = 1 - (1 / gamma ** 2)

    me_c2_ev = 0.51099895e6
    I_water_ev = 75.3

    argument = (2 * me_c2_ev * beta_sq) / (1 - beta_sq)
    K = np.log(argument) - beta_sq
    B_w = K - np.log(I_water_ev)

    return K, B_w

CONST_K, CONST_BW = calculate_bethe_constants(175)

# --- Hunemohr Functions ---
def rho_e_hunemohr(HU_h, HU_l, c):
    # Eq 14: Linear combination of High/Low Reduced CT numbers
    return c * (HU_h/1000 + 1) + (1 - c) * ((HU_l/1000) + 1)

def z_eff_hunemohr(n_i, Z_i, n=3.1):
    num = np.sum(n_i * (Z_i ** (n + 1)))
    den = np.sum(n_i * Z_i)
    return (num / den) ** (1 / n)

def spr_hunemohr_corrected(rho, ln_I):
    return rho * ((CONST_K - ln_I) / CONST_BW)

# --- Fitting Functions ---
def optimize_c(HU_H_List, HU_L_List, true_rho_list, materials_list):
    for i, mat in enumerate(materials_list):
        if mat in true_rho_list:
            hu_h = HU_H_List[i]
            hu_l = HU_L_List[i]

            # Calculate terms
            term_h = (hu_h / 1000) + 1
            term_l = (hu_l / 1000) + 1

            # Check denominator to avoid singularity
            if abs(term_h - term_l) > 1e-5:
                true_rho = true_rho_list[mat]
                # Solve algebraically
                c_val = (true_rho - term_l) / (term_h - term_l)
                return c_val

    # Fallback if no suitable material found
    return 0.5

def calculate_ref_zeff_hunemohr(material):
    composition = MATERIAL_PROPERTIES[material]["composition"]
    elements = list(composition.keys())
    fractions = np.array([composition[el] for el in elements])
    atomic_numbers = np.array([ATOMIC_NUMBERS[el] for el in elements])
    return z_eff_hunemohr(fractions, atomic_numbers, 3.1)

def z_eff_model(X, d_e, n=3.1):
    rho_e, zeff_w, x1, x2 = X.T
    factor = np.where(rho_e != 0, rho_e ** -1, 0)
    term1 = d_e * ((x1 / 1000) + 1)
    term2 = (zeff_w ** n - d_e) * ((x2 / 1000) + 1)
    inner = factor * (term1 + term2)
    return np.abs(inner) ** (1/n)

def fit_zeff(rho_e, zeff_w, true_zeff, x1, x2):
    rho_e = np.asarray(rho_e)
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    true_zeff = np.asarray(true_zeff)

    # Filter Mask: Strict cutoff for Lung/Air
    mask = rho_e > 0.6

    rho_fit = rho_e[mask]
    x1_fit = x1[mask]
    x2_fit = x2[mask]
    zeff_fit = true_zeff[mask]
    zeff_w_fit = np.full_like(rho_fit, zeff_w)

    if len(rho_fit) == 0:
        return 1.0

    X = np.column_stack((rho_fit, zeff_w_fit, x1_fit, x2_fit))
    try:
        popt, _ = curve_fit(lambda X, d_e: z_eff_model(
            X, d_e), X, zeff_fit, maxfev=5000)
        return popt[0]
    except:
        return 1.0

def calculate_zeff_optimized(rho_e, zeff_w, x1, x2, d_e, n=3.1):
    if rho_e == 0:
        return 0
    factor = (rho_e) ** -1
    term1 = d_e * ((x1 / 1000) + 1)
    term2 = (zeff_w ** n - d_e) * ((x2 / 1000) + 1)
    inner = factor * (term1 + term2)
    return np.abs(inner) ** (1/n)

def get_t_spr(material):
    return TRUE_SPR_VALUES.get(material, 1.000)

def predict_ln_I_biological(Z_eff):
    if Z_eff <= 8.5:
        return 0.1234 * Z_eff + 3.376
    else:
        return 0.09800 * Z_eff + 3.379

# --- Main Batch Function ---
def hunemohr_test(high_folder, low_folder, phantom_type, radii_ratios, c, d_e):
    high_files = sorted([f for f in os.listdir(high_folder)
                        if f.lower().endswith(('.dcm', '.ima'))])
    low_files = sorted([f for f in os.listdir(low_folder)
                       if f.lower().endswith(('.dcm', '.ima'))])
    count = min(len(high_files), len(low_files))

    saved_circles = CIRCLE_DATA[phantom_type]
    saved_materials = [c["material"]
                       for c in saved_circles if c["material"] in TRUE_RHO and c["material"] not in ["LN-450", "LN-350"]]

    aggregated_sprs = {mat: [None] * count for mat in saved_materials}

    # Initialize storage for plotting data
    aggregated_plot_data = {mat: {'z': [], 'lni': []}
                            for mat in saved_materials}

    for i in range(count):
        high_path = os.path.join(high_folder, high_files[i])
        low_path = os.path.join(low_folder, low_files[i])

        try:
            dicom_data_h = pydicom.dcmread(high_path)
            dicom_data_l = pydicom.dcmread(low_path)
            high_image = dicom_data_h.pixel_array
            low_image = dicom_data_l.pixel_array

            HU_H_List, HU_L_List, materials_list = [], [], []

            for circle in saved_circles:
                x, y, radius, material = int(circle["x"]), int(
                    circle["y"]), circle["radius"], circle["material"]
                if material not in TRUE_RHO:
                    continue
                materials_list.append(material)

                mask = np.zeros(high_image.shape, dtype=np.uint8)
                cv2.circle(mask, (x, y), int(
                    radius * (radii_ratios / 100)), 1, thickness=-1)

                mean_high_hu = np.mean(
                    high_image[mask == 1]) * dicom_data_h.RescaleSlope + dicom_data_h.RescaleIntercept
                mean_low_hu = np.mean(
                    low_image[mask == 1]) * dicom_data_l.RescaleSlope + dicom_data_l.RescaleIntercept

                HU_H_List.append(mean_high_hu)
                HU_L_List.append(mean_low_hu)

            if not HU_H_List:
                continue

            # --- Calculate Rho ---
            calculated_rhos = []
            for hu_h, hu_l in zip(HU_H_List, HU_L_List):
                rho = rho_e_hunemohr(hu_h, hu_l, c)
                calculated_rhos.append(rho)
                
            # --- Fit Zeff Parameter For Water ---
            zeff_w = calculate_ref_zeff_hunemohr("Solid Water")

            # --- Calculate Optimized Zeff ---
            optimized_zs = []
            for rhos, x1, x2 in zip(calculated_rhos, HU_H_List, HU_L_List):
                if rhos < 0.6:
                    optimized_zs.append(0.0)
                else:
                    opt_z = calculate_zeff_optimized(rhos, zeff_w, x1, x2, d_e)
                    optimized_zs.append(opt_z)

            # --- 5. Calculate SPR (Using Rho only for Lung) ---
            for z, rho, mat in zip(optimized_zs, calculated_rhos, materials_list):

                # Calculate Ln I for all materials for plotting purposes
                current_ln_I = predict_ln_I_biological(z)

                if rho < 0.6:
                    # Direct approximation for Lung
                    spr_val = rho * ((12.77 - (0.125 * 7.5 + 3.378)) / 8.45)
                else:
                    # Standard calculation for tissues/bone
                    spr_val = spr_hunemohr_corrected(rho, current_ln_I)

                if mat in aggregated_sprs:
                    aggregated_sprs[mat][i] = spr_val

                # Aggregate data for plotting
                if mat in aggregated_plot_data:
                    aggregated_plot_data[mat]['z'].append(z)
                    aggregated_plot_data[mat]['lni'].append(current_ln_I)

        except Exception as e:
            logger.error(f"Error processing slice {i}: {e}")
            continue

    batch_results = []

    for mat, values in aggregated_sprs.items():
        valid_values = [v for v in values if v is not None]

        if not valid_values:
            mean_spr = 0
            std_dev = 0
        else:
            mean_spr = np.mean(valid_values)
            std_dev = np.std(valid_values)

        true_spr = get_t_spr(mat)

        diff = mean_spr - true_spr
        percent_diff = (diff / true_spr) * 100 if true_spr != 0 else 0

        slice_pct_diffs = []
        for v in values:
            if v is None:
                slice_pct_diffs.append(None)
            else:
                d = (v - true_spr) / true_spr * 100 if true_spr != 0 else 0
                slice_pct_diffs.append(float(d))

        batch_results.append({
            "material": mat,
            "mean_spr": float(mean_spr),
            "calculated_spr": float(mean_spr),
            "std_dev": float(std_dev),
            "true_spr": float(true_spr),
            "percent_diff": float(percent_diff),
            "count": len(valid_values),
            "slice_sprs": values,
            "slice_percent_diffs": slice_pct_diffs
        })

    all_pct_diffs = [r["percent_diff"]
                     for r in batch_results if r["count"] > 0]
    global_mean_pct_diff = np.mean(all_pct_diffs) if all_pct_diffs else 0
    global_std_pct_diff = np.std(all_pct_diffs) if all_pct_diffs else 0
    global_mean_abs_pct_diff = np.mean(
        np.abs(all_pct_diffs)) if all_pct_diffs else 0

    results = {
        "material_details": batch_results,
        "error_metrics": {
            "spr": {
                "mean_percent_diff": global_mean_pct_diff,
                "std_percent_diff": global_std_pct_diff,
                "mean_abs_percent_diff": global_mean_abs_pct_diff
            }
        },
        "last_fit_params": {
            "c": float(c) if 'c' in locals() else 0,
            "d_e": float(d_e) if 'd_e' in locals() else 0
        }
    }
    
    print(results)

    return json.dumps(results, indent=4)

def hunemohr(high_path, low_path, phantom_type, radii_ratios):
    return json.dumps({"error": "Use batch mode"})

def hunemohr_batch(high_folder, low_folder, phantom_type, radii_ratios):
    high_files = sorted([f for f in os.listdir(high_folder)
                        if f.lower().endswith(('.dcm', '.ima'))])
    low_files = sorted([f for f in os.listdir(low_folder)
                       if f.lower().endswith(('.dcm', '.ima'))])
    count = min(len(high_files), len(low_files))

    saved_circles = CIRCLE_DATA[phantom_type]
    saved_materials = [c["material"]for c in saved_circles if c["material"] in TRUE_RHO]

    aggregated_sprs = {mat: [None] * count for mat in saved_materials}

    # Initialize storage for plotting data
    aggregated_plot_data = {mat: {'z': [], 'lni': []}
                            for mat in saved_materials}

    for i in range(count):
        high_path = os.path.join(high_folder, high_files[i])
        low_path = os.path.join(low_folder, low_files[i])

        try:
            dicom_data_h = pydicom.dcmread(high_path)
            dicom_data_l = pydicom.dcmread(low_path)
            high_image = dicom_data_h.pixel_array
            low_image = dicom_data_l.pixel_array

            HU_H_List, HU_L_List, materials_list = [], [], []

            for circle in saved_circles:
                x, y, radius, material = int(circle["x"]), int(
                    circle["y"]), circle["radius"], circle["material"]
                if material not in TRUE_RHO:
                    continue
                materials_list.append(material)

                mask = np.zeros(high_image.shape, dtype=np.uint8)
                cv2.circle(mask, (x, y), int(
                    radius * (radii_ratios / 100)), 1, thickness=-1)

                mean_high_hu = np.mean(
                    high_image[mask == 1]) * dicom_data_h.RescaleSlope + dicom_data_h.RescaleIntercept
                mean_low_hu = np.mean(
                    low_image[mask == 1]) * dicom_data_l.RescaleSlope + dicom_data_l.RescaleIntercept

                HU_H_List.append(mean_high_hu)
                HU_L_List.append(mean_low_hu)

            if not HU_H_List:
                continue

            # --- 1. Calculate C ---
            c = optimize_c(HU_H_List, HU_L_List, TRUE_RHO, materials_list)

            # --- 2. Calculate Rho ---
            calculated_rhos = []
            for hu_h, hu_l in zip(HU_H_List, HU_L_List):
                rho = rho_e_hunemohr(hu_h, hu_l, c)
                calculated_rhos.append(rho)

            # --- 3. Fit Zeff Parameter (Excluding Lung) ---
            calculated_ref_zeffs = []
            for mat, rho in zip(materials_list, calculated_rhos):
                if rho < 0.6:
                    calculated_ref_zeffs.append(0.0)
                else:
                    calculated_ref_zeffs.append(
                        calculate_ref_zeff_hunemohr(mat))

            zeff_w = calculate_ref_zeff_hunemohr("Solid Water")

            # fit_zeff internally filters out rho < 0.6
            d_e = fit_zeff(calculated_rhos, zeff_w,
                           calculated_ref_zeffs, HU_H_List, HU_L_List)

            # --- 4. Calculate Optimized Zeff ---
            optimized_zs = []
            for rhos, x1, x2 in zip(calculated_rhos, HU_H_List, HU_L_List):
                if rhos < 0.6:
                    optimized_zs.append(0.0)
                else:
                    opt_z = calculate_zeff_optimized(rhos, zeff_w, x1, x2, d_e)
                    optimized_zs.append(opt_z)

            # --- 5. Calculate SPR (Using Rho only for Lung) ---
            for z, rho, mat in zip(optimized_zs, calculated_rhos, materials_list):

                # Calculate Ln I for all materials for plotting purposes
                current_ln_I = predict_ln_I_biological(z)

                if rho < 0.6:
                    # Direct approximation for Lung
                    spr_val = rho * ((12.77 - (0.125 * 7.5 + 3.378)) / 8.45)
                else:
                    # Standard calculation for tissues/bone
                    spr_val = spr_hunemohr_corrected(rho, current_ln_I)

                if mat in aggregated_sprs:
                    aggregated_sprs[mat][i] = spr_val

                # Aggregate data for plotting
                if mat in aggregated_plot_data:
                    aggregated_plot_data[mat]['z'].append(z)
                    aggregated_plot_data[mat]['lni'].append(current_ln_I)

        except Exception as e:
            logger.error(f"Error processing slice {i}: {e}")
            continue

    batch_results = []

    for mat, values in aggregated_sprs.items():
        valid_values = [v for v in values if v is not None]

        if not valid_values:
            mean_spr = 0
            std_dev = 0
        else:
            mean_spr = np.mean(valid_values)
            std_dev = np.std(valid_values)

        true_spr = get_t_spr(mat)

        diff = mean_spr - true_spr
        percent_diff = (diff / true_spr) * 100 if true_spr != 0 else 0

        slice_pct_diffs = []
        for v in values:
            if v is None:
                slice_pct_diffs.append(None)
            else:
                d = (v - true_spr) / true_spr * 100 if true_spr != 0 else 0
                slice_pct_diffs.append(float(d))

        batch_results.append({
            "material": mat,
            "mean_spr": float(mean_spr),
            "calculated_spr": float(mean_spr),
            "std_dev": float(std_dev),
            "true_spr": float(true_spr),
            "percent_diff": float(percent_diff),
            "count": len(valid_values),
            "slice_sprs": values,
            "slice_percent_diffs": slice_pct_diffs
        })

    all_pct_diffs = [r["percent_diff"]
                     for r in batch_results if r["count"] > 0]
    global_mean_pct_diff = np.mean(all_pct_diffs) if all_pct_diffs else 0
    global_std_pct_diff = np.std(all_pct_diffs) if all_pct_diffs else 0
    global_mean_abs_pct_diff = np.mean(
        np.abs(all_pct_diffs)) if all_pct_diffs else 0

    results = {
        "material_details": batch_results,
        "error_metrics": {
            "spr": {
                "mean_percent_diff": global_mean_pct_diff,
                "std_percent_diff": global_std_pct_diff,
                "mean_abs_percent_diff": global_mean_abs_pct_diff
            }
        },
        "last_fit_params": {
            "c": float(c) if 'c' in locals() else 0,
            "d_e": float(d_e) if 'd_e' in locals() else 0
        }
    }

    return json.dumps(results, indent=4)
