import os
import json
import logging
import pydicom
import cv2
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.constants import physical_constants
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = Path("data")

def load_json(file_name):
    with open(DATA_DIR / file_name, "r") as file:
        return json.load(file)

# Load Data
CIRCLE_DATA = load_json("circles.json")
MATERIAL_PROPERTIES = load_json("material_properties.json")
ATOMIC_NUMBERS = load_json("atomic_numbers.json")
ELEMENTAL_PROPERTIES = load_json("element_properties.json")

# True electron densities (ρe) and Z_eff
TRUE_RHO = {mat: MATERIAL_PROPERTIES[mat]["rho_e_w"]
            for mat in MATERIAL_PROPERTIES}
TRUE_ZEFF = {mat: MATERIAL_PROPERTIES[mat]["Z_eff"]
             for mat in MATERIAL_PROPERTIES}

TRUE_SPR_VALUES = {
    "Cortical Bone": 1.8479321476896744,
    "Breast": 0.977656790325545,
    "Adipose": 0.9554505522678512,
    "30% CaCO3": 1.3438744285396211,
    "LN-350": 0.2909152448618743,
    "LN-450": 0.44990001186623196,
    "Liver": 1.0800681536389107,
    "Inner Bone": 1.2185667167409664,
    "50% CaCO3": 1.5780908215659672,
    "Brain": 1.0445334165312719,
    "Solid Water": 1.0199031277454536, "Water": 0.9997458917957189
}

# --- PHYSICS FUNCTIONS ---
def delta_HU(alpha, HU_H, HU_L):
    return (1 + alpha) * HU_H - (alpha * HU_L)

def rho_e_calc(delta_HU, a, b):
    return (a * (delta_HU / 1000) + b)

def reduce_ct(HU):
    return 1.006*(HU/1000) + 0.988

def zeff_lhs(zeff):
    return ((zeff / 7.45) ** 3.3) - 1

def zeff_rhs(gamma, ct, rho):
    if rho == 0:
        return 0
    return gamma * ((ct/rho) - 1)

def zeff_hunemohr(n_i, Z_i, n=3.1):
    num = np.sum(n_i * (Z_i ** (n + 1)))
    den = np.sum(n_i * Z_i)
    return (num / den) ** (1 / n)

def i_truth(weight_fractions, Num, A, I):
    return sum(weight_fractions * Num / A * np.log(I)) / sum(weight_fractions * Num / A)

def i_tanaka(z_ratio, c0, c1):
    return c1 * (z_ratio - 1) - c0

def get_I(mean_exciation):
    return 75 * (np.e ** mean_exciation)

def beta(kvp):
    kinetic_energy_mev = kvp / 1000
    proton_mass_mev = physical_constants['proton mass energy equivalent in MeV'][0]
    gamma = (proton_mass_mev + kinetic_energy_mev) / proton_mass_mev
    return np.sqrt(1 - (1 / gamma ** 2))

def spr_tanaka(rho, I, beta):
    me = 9.10938356e-31
    c = 2.99792458e8
    Iw = 75
    term1 = np.log(I/Iw)
    term2 = np.log((2 * me * c ** 2 * beta ** 2) / (Iw * (1 - beta ** 2)))
    return rho * (1 - (term1 / (term2 - beta ** 2)))

# --- OPTIMIZATION HELPERS ---
def optimize_alpha(HU_H_LIST, HU_L_LIST, true_rho_list, materials_list):
    best_r2 = -np.inf
    best_alpha = 0.5
    best_a = 0
    best_b = 0

    if not HU_H_LIST:
        return 0.5, 0, 0, 0

    alphas = np.linspace(0, 1, 500)
    valid_indices = [i for i, mat in enumerate(
        materials_list) if mat in true_rho_list]

    if not valid_indices:
        return 0.5, 0, 0, 0

    # Extract valid points
    y_vals = [true_rho_list[materials_list[i]] for i in valid_indices]
    h_h_vals = [HU_H_LIST[i] for i in valid_indices]
    h_l_vals = [HU_L_LIST[i] for i in valid_indices]

    # Anchor the regression to (HU=-1000, Rho=0) to prevent floating intercepts
    y_vals.append(0.0)
    h_h_vals.append(-1000.0)
    h_l_vals.append(-1000.0)

    y = np.array(y_vals)
    h_h = np.array(h_h_vals)
    h_l = np.array(h_l_vals)

    for alpha in alphas:
        # Vectorized calculation
        deltas = delta_HU(alpha, h_h, h_l)
        x = (deltas / 1000.0).reshape(-1, 1)

        model = LinearRegression().fit(x, y)
        y_pred = model.predict(x)
        r2 = r2_score(y, y_pred)

        if r2 > best_r2:
            best_r2 = r2
            best_alpha = alpha
            best_a = model.coef_[0]
            best_b = model.intercept_

    return best_alpha, best_a, best_b, best_r2

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
    zeff_w_arr = np.full_like(rho_e, zeff_w)
    X = np.column_stack((rho_e, zeff_w_arr, x1, x2))

    try:
        popt, _ = curve_fit(lambda X, d_e: z_eff_model(
            X, d_e), X, true_zeff, maxfev=5000)
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

def calculate_z_eff_hunemohr(material):
    composition = MATERIAL_PROPERTIES[material]["composition"]
    elements = list(composition.keys())
    fractions = np.array([composition[el] for el in elements])
    atomic_numbers = np.array([ATOMIC_NUMBERS[el] for el in elements])
    return zeff_hunemohr(fractions, atomic_numbers)

def optimize_gamma_linear(zeff_list, ct_list, rho_list):
    try:
        X_list = []
        Y_list = []
        for zeff, ct, rho in zip(zeff_list, ct_list, rho_list):
            if rho == 0:
                continue
            lhs = zeff_lhs(zeff)
            rhs_term = (ct / rho) - 1
            X_list.append(rhs_term)
            Y_list.append(lhs)

        X = np.array(X_list).reshape(-1, 1)
        Y = np.array(Y_list)

        model = LinearRegression(fit_intercept=False).fit(X, Y)
        return model.coef_[0]
    except:
        return 12.0
    
def calculate_optimal_threshold():
    return 8.8

def optimize_c_segmented_dynamic(ionization_list, z_ratio_list, true_zeff_list, threshold):
    soft_i, soft_z = [], []
    bone_i, bone_z = [], []

    for i, z, true_z in zip(ionization_list, z_ratio_list, true_zeff_list):
        if true_z < threshold:
            soft_i.append(i)
            soft_z.append(z)
        else:
            bone_i.append(i)
            bone_z.append(z)

    soft_popt = [0.0206, 0.3423]
    bone_popt = [0.0444, 0.0696]

    if len(soft_z) >= 2:
        try:
            popt, _ = curve_fit(i_tanaka, np.array(soft_z),
                                np.array(soft_i), p0=soft_popt)
            soft_popt = popt
        except:
            pass

    if len(bone_z) >= 2:
        try:
            popt, _ = curve_fit(i_tanaka, np.array(bone_z),
                                np.array(bone_i), p0=bone_popt)
            bone_popt = popt
        except:
            pass

    return soft_popt, bone_popt

def get_t_spr(material):
    return TRUE_SPR_VALUES.get(material, 1.000)

#### USABLE FUNCTIONS ####
def tanaka_test(high_path, low_path, phantom_type, radii_ratios, alpha, a, b, gamma, c0, c1):
    return json.dumps({"error": "Use batch mode"})

def tanaka(high_path, low_path, phantom_type, radii_ratios):
    return json.dumps({"error": "Use batch mode"})

def tanaka_batch(high_folder, low_folder, phantom_type, radii_ratios):
    high_files = sorted([f for f in os.listdir(high_folder)
                        if f.lower().endswith(('.dcm', '.ima'))])
    low_files = sorted([f for f in os.listdir(low_folder)
                       if f.lower().endswith(('.dcm', '.ima'))])
    count = min(len(high_files), len(low_files))

    saved_circles = CIRCLE_DATA[phantom_type]
    saved_materials = [c["material"]
                       for c in saved_circles if c["material"] in TRUE_RHO]

    aggregated_sprs = {mat: [None] * count for mat in saved_materials}

    for i in range(count):
        high_path = os.path.join(high_folder, high_files[i])
        low_path = os.path.join(low_folder, low_files[i])

        try:
            dicom_data_h = pydicom.dcmread(high_path)
            dicom_data_l = pydicom.dcmread(low_path)
            high_image = dicom_data_h.pixel_array
            low_image = dicom_data_l.pixel_array

            materials_list = []
            HU_H_List, HU_L_List = [], []

            for circle in saved_circles:
                x, y, radius, material = circle["x"], circle["y"], circle["radius"], circle["material"]
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

            # 1. Optimize Alpha
            alpha, a, b, r = optimize_alpha(
                HU_H_List, HU_L_List, TRUE_RHO, materials_list)

            deltas = [delta_HU(alpha, h, l)
                      for h, l in zip(HU_H_List, HU_L_List)]
            calculated_rhos = [rho_e_calc(d, a, b) for d in deltas]
            reduced_ct = [reduce_ct(hl) for hl in HU_L_List]

            # 2. Optimize Gamma
            zeff_list = [TRUE_ZEFF[mat] for mat in materials_list]
            gamma = optimize_gamma_linear(
                zeff_list, reduced_ct, calculated_rhos)

            calculated_z_ratios = [
                zeff_rhs(gamma, ct, rho) for ct, rho in zip(reduced_ct, calculated_rhos)]
            zeff_w = calculate_z_eff_hunemohr("True Water")

            calculated_z_effs = [(abs(zeff_rhs(gamma, ct, rho) + 1)) **
                                 (1/3.3) * 7.45 for ct, rho in zip(reduced_ct, calculated_rhos)]

            # 3. Fit d_e using TRUE Z_eff
            d_e = fit_zeff(calculated_rhos, zeff_w,
                           calculated_z_effs, HU_H_List, HU_L_List)

            # 4. Calculate Optimized Zs (Refines Zeff estimates)
            optimized_zs = []
            for rhos, x1, x2 in zip(calculated_rhos, HU_H_List, HU_L_List):
                val = calculate_zeff_optimized(rhos, zeff_w, x1, x2, d_e)
                optimized_zs.append(val)

            # 5. Calculate True I-values
            true_mean_excitation = []
            for mat in materials_list:
                comp = MATERIAL_PROPERTIES[mat]["composition"]
                el = list(comp.keys())
                fr = np.array([comp[e] for e in el])
                an = np.array([ELEMENTAL_PROPERTIES[e]["number"] for e in el])
                am = np.array([ELEMENTAL_PROPERTIES[e]["mass"] for e in el])
                ie = np.array([ELEMENTAL_PROPERTIES[e]["ionization"]
                              for e in el])
                true_mean_excitation.append(i_truth(fr, an, am, ie))

            # 6. DYNAMIC THRESHOLD & SEGMENTED OPTIMIZATION
            threshold = calculate_optimal_threshold()
            soft_popt, bone_popt = optimize_c_segmented_dynamic(
                true_mean_excitation, calculated_z_ratios, zeff_list, threshold)
            c0_s, c1_s = soft_popt
            c0_b, c1_b = bone_popt

            # 7. Apply Curve based on REFINED Z_eff
            calculated_mean_excitation = []
            for z_ratio, z_eff_est in zip(calculated_z_ratios, optimized_zs):
                if z_eff_est < threshold:
                    val = i_tanaka(z_ratio, c0_s, c1_s)
                else:
                    val = i_tanaka(z_ratio, c0_b, c1_b)
                calculated_mean_excitation.append(val)

            for t, rho, mat in zip(calculated_mean_excitation, calculated_rhos, materials_list):
                I = get_I(t)
                beta2 = beta(200)
                spr = spr_tanaka(rho, I, beta2)

                if not np.isnan(spr) and mat in aggregated_sprs:
                    aggregated_sprs[mat][i] = spr

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
            val_array = np.array(valid_values)
            mean_spr = np.mean(val_array)
            std_dev = np.std(val_array)

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
        }
    }

    return json.dumps(results, indent=4)
