import json
from pathlib import Path
import pydicom
import numpy as np
import scipy as sp
import cv2
from scipy.optimize import curve_fit
from scipy.constants import physical_constants

DATA_DIR = Path("data")

def load_json(file_name):
    path = DATA_DIR / file_name
    if not path.exists():
        path = Path("..") / "data" / file_name
    with open(path, "r") as file:
        return json.load(file)
try:
    CIRCLE_DATA = load_json("circles.json")
    MATERIAL_PROPERTIES = load_json("material_properties.json")
    ELEMENTAL_PROPERTIES = load_json("element_properties.json")
    ICRP_PROPERTIES = load_json("icrp.json")
except Exception as e:
    print(f"Warning: Could not load JSON data in schneider.py: {e}")
    CIRCLE_DATA, MATERIAL_PROPERTIES, ELEMENTAL_PROPERTIES, ICRP_PROPERTIES = {}, {}, {}, {}

def compute_Ng(material, flag="Phantoms"):
    N_A = sp.constants.Avogadro
    composition = MATERIAL_PROPERTIES[material]["composition"] if flag == "Phantoms" else ICRP_PROPERTIES[material]["composition"]
    sum_term = 0
    for element, weight_fraction in composition.items():
        Z_i = ELEMENTAL_PROPERTIES[element]["number"]
        A_i = ELEMENTAL_PROPERTIES[element]["mass"]
        sum_term += (weight_fraction * Z_i) / A_i
    return N_A * sum_term

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

def compute_rhoe_schneider(material, water="True Water", flag="Phantoms"):
    Ng = compute_Ng(material) if flag == "Phantoms" else compute_Ng(
        material, "ICRP")
    Ng_w = compute_Ng(water)
    rho = MATERIAL_PROPERTIES[material]["density"] if flag == "Phantoms" else ICRP_PROPERTIES[material]["density"]
    rho_w = MATERIAL_PROPERTIES[water]["density"]
    return (rho * Ng) / (rho_w * Ng_w)

def mu_model_fit(X, Kph, Kcoh, KKN):
    rhoNg = X[:, 0]
    Zbar = X[:, 1]
    Zhat = X[:, 2]
    return rhoNg * (Kph * Zbar ** 3.62 + Kcoh * Zhat ** 1.86 + KKN)

def linear_attenuation(material):
    rho = MATERIAL_PROPERTIES[material]["density"]
    composition = MATERIAL_PROPERTIES[material]["composition"]
    mu_total = 0.0
    for element, fraction in composition.items():
        atomic_mass = ELEMENTAL_PROPERTIES[element]["mass"]
        atomic_number = ELEMENTAL_PROPERTIES[element]["number"]
        N = (rho * fraction) / atomic_mass
        mu_a = atomic_number
        mu_total += mu_a * N
    return mu_total

def calculate_mu(material, Kph, Kcoh, KKN):
    Ng = compute_Ng(material)
    rho = MATERIAL_PROPERTIES[material]["density"]
    rhoNg = (rho * Ng) / 1e23
    Zbar = compute_weighted_Z(material, 3.62)
    Zhat = compute_weighted_Z(material, 1.86)
    return rhoNg * (Kph * Zbar ** 3.62 + Kcoh * Zhat ** 1.86 + KKN)

def hounsfield_schneider(mew, mew_w):
    return ((mew / mew_w) - 1) * 1000

def calculate_HU(tissues, Kph, Kcoh, KKN, flag="Phantoms"):
    mu_water = calculate_mu("True Water", Kph, Kcoh, KKN)
    res = []
    for tissue in tissues:
        Ng = compute_Ng(tissue) if flag == "Phantoms" else compute_Ng(tissue, "ICRP")
        rho = MATERIAL_PROPERTIES[tissue]["density"] if flag == "Phantoms" else ICRP_PROPERTIES[tissue]["density"]
        rhoNg = (rho * Ng) / 1e23
        Zbar = compute_weighted_Z(tissue, 3.62) if flag == "Phantoms" else compute_weighted_Z(tissue, 3.62, "ICRP")
        Zhat = compute_weighted_Z(tissue, 1.86) if flag == "Phantoms" else compute_weighted_Z(tissue, 1.86, "ICRP")
        mu = rhoNg * (Kph * Zbar ** 3.62 + Kcoh * Zhat ** 1.86 + KKN)
        HU = hounsfield_schneider(mu, mu_water)
        res.append(HU)
    return res

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

def get_beta(kvp=120):
    kinetic_energy_mev = kvp / 1000
    proton_mass_mev = physical_constants['proton mass energy equivalent in MeV'][0]
    gamma = (proton_mass_mev + kinetic_energy_mev) / proton_mass_mev
    return np.sqrt(1 - (1 / gamma ** 2)) ** 2

def calculate_spr(rhoe, I, I_water=75):
    me = 9.10938356e-31
    c = 2.99792458e8
    beta = get_beta()
    numerator = (np.log(2*me * (c ** 2) * beta)) / (I*(1 - beta) - beta)
    denominator = (np.log(2*me * (c ** 2) * beta)) / \
        (I_water*(1 - beta) - beta)
    return rhoe * (numerator / denominator)

def linear_fit(x, m, c):
    return m * x + c

def perform_segmented_fit(x_data, y_data, split_point=100):
    """
    Splits data into 'Soft' (<= split_point) and 'Bone' (> split_point)
    and fits two separate linear lines.
    Returns [m_soft, c_soft, m_bone, c_bone, split_point]
    """
    x_data = np.array(x_data)
    y_data = np.array(y_data)

    # 1. Soft Tissue Segment
    mask_soft = x_data <= split_point
    if np.sum(mask_soft) > 1:
        popt_soft, _ = curve_fit(
            linear_fit, x_data[mask_soft], y_data[mask_soft])
    else:
        # Fallback default (water-like)
        popt_soft = [1e-3, 1.0]

    # 2. Bone Segment
    mask_bone = x_data > split_point
    if np.sum(mask_bone) > 1:
        popt_bone, _ = curve_fit(
            linear_fit, x_data[mask_bone], y_data[mask_bone])
    else:
        # Fallback: use soft params if no bone data
        popt_bone = popt_soft

    return [popt_soft[0], popt_soft[1], popt_bone[0], popt_bone[1], split_point]

def predict_segmented(hu_val, params):
    """
    Predicts Y based on HU using the segmented parameters.
    params: [m_soft, c_soft, m_bone, c_bone, split_point]
    """
    m_soft, c_soft, m_bone, c_bone, split_point = params

    if hu_val <= split_point:
        return m_soft * hu_val + c_soft
    else:
        return m_bone * hu_val + c_bone

def schneider(path, phantom_type, radii_ratio):
    """
    Generates calibration parameters using segmented fitting.
    Returns lists compatible with JSON serialization.
    """
    dicom_data = pydicom.dcmread(path)
    image = dicom_data.pixel_array

    HU_List, materials_list, rhos_ICRP, sprs = [], [], [], []

    if phantom_type not in CIRCLE_DATA:
        raise ValueError(f"Phantom type '{phantom_type}' not found.")

    SAVED_CIRCLES = CIRCLE_DATA[phantom_type]

    # 1. Measure HU from Phantom
    for circle in SAVED_CIRCLES:
        x, y, radius, material = circle["x"], circle["y"], circle["radius"], circle["material"]
        if material == '50% CaCO3' or material == '30% CaCO3':
            continue

        if material not in materials_list:
            materials_list.append(material)

        mask = np.zeros(image.shape, dtype=np.uint8)
        cv2.circle(mask, (x, y), int(radius * radii_ratio), 1, thickness=-1)
        pixel_values = image[mask == 1]

        # Standard HU
        hu = np.mean(pixel_values) * dicom_data.RescaleSlope + \
            dicom_data.RescaleIntercept
        HU_List.append(hu)

    # 2. Fit K parameters
    rhoNg_list, Zbar_list, Zhat_list, mu_list = [], [], [], []

    mu_water = linear_attenuation("True Water")

    for i, material in enumerate(materials_list):
        rho = MATERIAL_PROPERTIES[material]["density"]
        Ng = compute_Ng(material)
        rhoNg = rho * Ng
        Zbar = compute_weighted_Z(material, 3.62)
        Zhat = compute_weighted_Z(material, 1.86)

        measured_HU = HU_List[i]

        # Invert Standard HU to get mu
        mu = mu_water * ((measured_HU / 1000.0) + 1.0)

        rhoNg_list.append(rhoNg)
        Zbar_list.append(Zbar)
        Zhat_list.append(Zhat)
        mu_list.append(mu)

    rhoNg_arr = np.array(rhoNg_list) / 1e23
    X = np.array([rhoNg_arr, Zbar_list, Zhat_list]).T
    y = np.array(mu_list)

    initial_guess = [1e-5, 4e-4, 0.5]
    bounds = ([0, 0, 0], [1e-3, 1e-2, 5])

    try:
        popt, _ = curve_fit(mu_model_fit, X, y, p0=initial_guess, bounds=bounds)
        Kph, Kcoh, KKN = popt
        print(f"Fitted K-params: {popt}")
    except Exception as e:
        print(f"Curve fit failed, using defaults: {e}")
        Kph, Kcoh, KKN = 1e-5, 4e-4, 0.5

    # 3. Simulate ICRP Tissues
    ICRP_Tissues = list(ICRP_PROPERTIES.keys())

    ICRP_HUs = calculate_HU(ICRP_Tissues, Kph, Kcoh, KKN, flag="ICRP")

    for material in ICRP_Tissues:
        rhos_ICRP.append(compute_rhoe_schneider(material, flag="ICRP"))

    for i, material in enumerate(ICRP_Tissues):
        rhoe = rhos_ICRP[i]
        I = compute_I(material, flag="ICRP")
        sprs.append(calculate_spr(rhoe, I))

    # 4. Generate SEGMENTED Calibration Curves
    split_hu = 100.0

    # Fit ED (Segmented Linear)
    ed_params = perform_segmented_fit(
        ICRP_HUs, rhos_ICRP, split_point=split_hu)

    # Fit SPR (Segmented Linear)
    spr_params = perform_segmented_fit(ICRP_HUs, sprs, split_point=split_hu)

    return list(ed_params), list(spr_params)


def test_schneider(path, phantom_type, radii_ratio, ed_params, spr_params):
    """
    Applies the SEGMENTED calibration to a test scan.
    """
    dicom_data = pydicom.dcmread(path)
    image = dicom_data.pixel_array

    results = {"materials": {}}

    if phantom_type not in CIRCLE_DATA:
        return results

    SAVED_CIRCLES = CIRCLE_DATA[phantom_type]

    for circle in SAVED_CIRCLES:
        x, y, radius, material = circle["x"], circle["y"], circle["radius"], circle["material"]

        if material == '50% CaCO3' or material == '30% CaCO3':
            continue

        mask = np.zeros(image.shape, dtype=np.uint8)
        cv2.circle(mask, (x, y), int(radius * radii_ratio), 1, thickness=-1)

        pixel_values = image[mask == 1]

        raw_hu = np.mean(pixel_values) * \
            dicom_data.RescaleSlope + dicom_data.RescaleIntercept

        # Predict using Segmented Logic
        pred_rho = predict_segmented(raw_hu, ed_params)
        pred_spr = predict_segmented(raw_hu, spr_params)
        
        Zbar = compute_weighted_Z(material, 3.62)
        # Zhat = compute_weighted_Z(material, 1.86)

        results["materials"][material] = {
            "mean_hu": float(raw_hu),
            "predicted_rho": float(pred_rho),
            "predicted_spr": float(pred_spr),
            "z_eff": float(Zbar)
        }

    return results
