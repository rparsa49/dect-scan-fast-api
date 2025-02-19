import shutil, math, os, json, cv2, logging, pydicom, numpy as np
from fastapi import FastAPI, Form, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from methods.saito import (
    alpha_saito, hu_saito, rho_e_calibrated_saito, rho_e_saito, mew_saito, z_eff_saito
)
from methods.hunemohr import rho_e_hunemohr, z_eff_hunemohr, spr_hunemohr
from methods.spr import sp_truth
from methods.true_attenuation import linear_attenuation
from methods.ln_mean_excitation_potential import ln_mean_excitation_potential
from dect_processing.dect import (
    load_dicom_image,
    categorize_hu_value, save_dicom_as_png, process_and_save_circles, find_dicom_files
)
from dect_processing.organize import convert_numpy

logging.basicConfig(level=logging.INFO)
app = FastAPI()

app.mount("/processed_images",
          StaticFiles(directory="processed_images"), name="processed_images")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

IMAGES_DIR = "processed_images"
DICOM_DIR = "uploaded_dicoms"

Path(IMAGES_DIR).mkdir(exist_ok=True)

# Load JSON data files
DATA_DIR = Path("data")

def load_json(file_name):
    with open(DATA_DIR / file_name, "r") as file:
        return json.load(file)

SUPPORTED_MODELS = load_json("supported_models.json")
HU_CATEGORIES = load_json("hu_categories.json")
ELEMENT_PROPERTIES = load_json("element_properties.json")
MATERIAL_PROPERTIES = load_json("material_properties.json")
CIRCLE_DATA = load_json("circles.json")
WATER_ATTENUATION = load_json("water_att.json")
ELEMENT_ATOMIC_NUMBERS = load_json("atomic_numbers.json")
'''
API CALLS
'''
@app.post("/upload-scan")
async def upload_scan(files: List[UploadFile] = File(...)):
    dicom_files = []
    for file in files:
        file_path = os.path.join(IMAGES_DIR, file.filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(await file.read())
        dicom_files.append(file_path)

    # Separate files by high and low kVp
    high_kvp_files = []
    low_kvp_files = []
    slice_thickness = None
    ref_dcm = pydicom.dcmread(dicom_files[0])
    ref_kvp = ref_dcm.get("KVP")

    for dicom_file in dicom_files:
        dicom_data = pydicom.dcmread(dicom_file)
        kvp = dicom_data.get("KVP")

        if slice_thickness is None:
            slice_thickness = dicom_data.get("SliceThickness")
        
        if kvp >= ref_kvp:
            high_kvp_files.append(dicom_file)
        else:
            low_kvp_files.append(dicom_file)

    # Convert DICOM files to PNG and store paths
    high_kvp_image_paths = []
    low_kvp_image_paths = []

    for i, high_file in enumerate(high_kvp_files):
        high_img_path = os.path.join(IMAGES_DIR, f"high_kvp_{i+1}.png")
        save_dicom_as_png(high_file, high_img_path)
        high_kvp_image_paths.append(f"/get-image/high_kvp_{i+1}.png")

    for i, low_file in enumerate(low_kvp_files):
        low_img_path = os.path.join(IMAGES_DIR, f"low_kvp_{i+1}.png")
        save_dicom_as_png(low_file, low_img_path)
        low_kvp_image_paths.append(f"/get-image/low_kvp_{i+1}.png")

    return {
        "high_kvp_images": high_kvp_image_paths,
        "low_kvp_images": low_kvp_image_paths,
        "slice_thickness": slice_thickness
    }

# Return image
@app.get("/get-image/{image_name}")
async def get_image(image_name: str):
    image_path = os.path.join(IMAGES_DIR, image_name)
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(image_path)

# Update images with insert notations
@app.post("/update-circles")
async def update_circles(request: Request):
    data = await request.json()
    percentage = data.get("radius", 100)  # Radius as a percentage
    phantom_type = data.get("phantom_type")
    high_kvp_images = data.get("high_kvp_images", [])
    low_kvp_images = data.get("low_kvp_images", [])

    if not phantom_type:
        raise HTTPException(status_code=400, detail="Missing phantom_type.")

    # Get circle locations for the specified phantom type
    if phantom_type not in CIRCLE_DATA:
        raise HTTPException(status_code=400, detail="Invalid phantom type.")
    circles_data = CIRCLE_DATA[phantom_type]

    updated_high_images = []
    updated_low_images = []

    # Process high kVp images
    for image_path in high_kvp_images:
        # Remove the "/get-image/" prefix to get the actual file name
        image_name = image_path.split("/get-image/")[-1]
        input_path = os.path.join(IMAGES_DIR, image_name)
        output_path = os.path.join(IMAGES_DIR, f"circled_{image_name}")
        process_and_save_circles(
            input_path, percentage, circles_data, output_path)
        updated_high_images.append(f"/get-image/circled_{image_name}")

    # Process low kVp images
    for image_path in low_kvp_images:
        # Remove the "/get-image/" prefix to get the actual file name
        image_name = image_path.split("/get-image/")[-1]
        input_path = os.path.join(IMAGES_DIR, image_name)
        output_path = os.path.join(IMAGES_DIR, f"circled_{image_name}")
        process_and_save_circles(
            input_path, percentage, circles_data, output_path)
        updated_low_images.append(f"/get-image/circled_{image_name}")

    return JSONResponse({
        "updated_high_kvp_images": updated_high_images,
        "updated_low_kvp_images": updated_low_images
    })

# Return supported models
@app.get("/get-supported-models")
async def get_supported_models():
    return JSONResponse(SUPPORTED_MODELS)

@app.post("/analyze-inserts")
async def analyze_inserts(request: Request):
    data = await request.json()
    radii_ratios = data.get("radius", [1.0])
    radii_ratios = int(radii_ratios) / 100
    phantom_type = data.get("phantom")
    saved_circles = CIRCLE_DATA[phantom_type]
    # Assume that we pass in the method type
    method_type = data.get("model")

    # dicom_file_path = find_first_dicom_file()
    high_path, low_path = find_dicom_files()
    
    if not high_path or not low_path:
        raise HTTPException(
            status_code=404, detail="No DICOM files found in the directory structure")
    
    # Load the DICOM image and convert pixel values to HU values
    high_image, dicom_datah = load_dicom_image(high_path)
    low_image, dicom_datal = load_dicom_image(low_path)

    high_hu = []
    low_hu = []
    for circle in saved_circles:
        x, y, radius = circle["x"], circle["y"], circle["radius"]
        mask = np.zeros(high_image.shape, dtype=np.uint8)
        cv2.circle(mask, (x, y), int(radius * radii_ratios), 1, thickness=-1)
        high_pixel_values = high_image[mask == 1]
        low_pixel_values = low_image[mask == 1]

        high_hu.append(np.mean(high_pixel_values))
        low_hu.append(np.mean(low_pixel_values))
    
    mean_hu_values = high_hu
    
    # mean_hu_values = []
    # for a, b in zip(high_hu, low_hu):
    #     mean_hu_values.append((a+b)/2)

    results = []
    # From here, everything should be done in a switch-case
    if method_type == 'Saito':
        iter = 0
        for i, hu in enumerate(mean_hu_values):
            # Calculate uncalibrated electron density
            rho_e = rho_e_saito(hu)
        
            # Use Saito's methods to compute linear attenuation coefficients
            mu_l = mew_saito(low_hu)
            mu_h = mew_saito(high_hu)
            for i in range(len(mu_h)):
                mu_h[i] += 1

            # Get the attenuation coefficient of water at the energy spectra
            water_mu_h = WATER_ATTENUATION.get(str(dicom_datah.KVP))
            water_mu_l = WATER_ATTENUATION.get(str(dicom_datal.KVP))

            # Calculate alpha
            alpha = alpha_saito(mu_l, mu_h, water_mu_l, water_mu_h, rho_e)
            
            # Calibrated rho calculation
            beta = float(data.get("beta"))
            rho_e_cal = rho_e_calibrated_saito(hu, alpha[iter], beta)
            
            # Effective atomic number calculation
            lam = 10.98
            z_eff = z_eff_saito(mu_h, lam, rho_e_cal)
        
            # Categorize materials based on HU
            material = categorize_hu_value(hu)
            
            # Calculate stopping power
            spr = None
            if material in MATERIAL_PROPERTIES:
                material_info = MATERIAL_PROPERTIES[material]
                a = sum(
                    fraction * ELEMENT_PROPERTIES[element]["mass"]
                    for element, fraction in material_info["composition"].items()
                )
                ln_i_m = ln_mean_excitation_potential(z_eff)
                
                # Calculate beta2 (velocity / speed of sound)
                velocity = 299792458 * math.sqrt(1 - ((0.511 / (dicom_datah.KVP + 0.511)) ** 2))
                beta2 = velocity / 299792458
                
                spr = sp_truth(z_eff, a, ln_i_m, beta2, dicom_datah.KVP)
            
            # Append results
            results.append({
            "material": material,
            "rho_e": rho_e,
            "z_eff": z_eff,
            "stopping_power": spr,
            "alpha": alpha[iter],
            "beta": beta,
            "lambda": lam
        })
            iter += 1
    elif method_type == 'Hunemohr':
        c = float(data.get('beta'))
        for i, (hu_h, hu_l) in enumerate(zip(high_hu, low_hu)):
            # Ensure HU values are scalars
            hu_h = float(hu_h)  # Convert NumPy type to Python float
            hu_l = float(hu_l)

            # Compute electron density using Hunemohr equation
            rho_e = rho_e_hunemohr(hu_h, hu_l, c)
            
            # Identify material from HU
            material = categorize_hu_value(hu_h)
            logging.info(f"Material Identified: {material}")
            # Compute Z_eff using Hunemohr's method
            if material in MATERIAL_PROPERTIES:
                material_info = MATERIAL_PROPERTIES[material]
                n_i = np.array(list(material_info["composition"].values()))
                Z_i = np.array([ELEMENT_ATOMIC_NUMBERS[element] for element in material_info["composition"]])

                Z_eff = z_eff_hunemohr(n_i, Z_i)
                logging.info(f"Effective atomic number for {material} is {Z_eff}")
                # Compute stopping power ratio
                spr = spr_hunemohr(rho_e, Z_eff, dicom_datah.KVP)
                logging.info(f"Stopping power for {material} is {spr}")
            else:
                logging.info("No material found")
                Z_eff, spr = None, None
            
            results.append({
                "material": material,
                "rho_e": rho_e,
                "z_eff": Z_eff,
                "stopping_power": spr
            })
            
    return JSONResponse({"results": [{k: convert_numpy(v) for k, v in res.items()} for res in results]})

# Test calibration
@app.post("/test-calibration")
async def test_calibration(
    alpha: str = Form(...),
    beta: str = Form(...),
    lambda_val: str = Form(...),
    method: str = Form(...),
    files: List[UploadFile] = File(...)
):
    # Create test upload directory
    test_folder = os.path.join(IMAGES_DIR, "test_calibration")
    os.makedirs(test_folder, exist_ok=True)

    # Save uploaded files
    for file in files:
        file_path = os.path.join(test_folder, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

    # **Parse JSON strings into lists of floats**
    try:
        # Convert string to a list by splitting on commas and stripping spaces
        alpha_list = [float(a.strip()) for a in alpha.split(",")]
        beta_list = [float(b.strip()) for b in beta.split(",")]
        lambda_list = [float(l.strip()) for l in lambda_val.split(",")]
    except (json.JSONDecodeError, ValueError) as e:
        raise HTTPException(
            status_code=400, detail=f"Invalid JSON format: {str(e)}")

    # Run calibration test logging using parsed parameters
    new_results = perform_test_calibration(
        test_folder, alpha_list, beta_list, lambda_list, method)

    return {"new_table": new_results}


def perform_test_calibration(folder_path, alpha, beta, lambda_val, method):
    """
    Perform test calibration using the given method, calibration parameters, and DICOM scan.
    """
    high_path, low_path = find_dicom_files(folder_path)

    if not high_path or not low_path:
        raise HTTPException(
            status_code=404, detail="No DICOM files found in the directory structure.")

    # Load DICOM images and convert pixel values to HU values
    high_image, dicom_datah = load_dicom_image(high_path)
    low_image, dicom_datal = load_dicom_image(low_path)

    # Extract mean HU values from defined circular regions
    saved_circles = CIRCLE_DATA.get("head")  # Adjust based on phantom type
    radii_ratios = 1.0  # Assume default ratio for testing

    high_hu, low_hu = [], []
    for circle in saved_circles:
        x, y, radius = circle["x"], circle["y"], circle["radius"]
        mask = np.zeros(high_image.shape, dtype=np.uint8)
        cv2.circle(mask, (x, y), int(radius * radii_ratios), 1, thickness=-1)
        high_pixel_values = high_image[mask == 1]
        low_pixel_values = low_image[mask == 1]

        high_hu.append(np.mean(high_pixel_values))
        low_hu.append(np.mean(low_pixel_values))

    mean_hu_values = high_hu  # Assuming using high kVp HU for material classification

    results = []

    # Determine calculation method
    if method == "Saito":
        iter = 0
        for i, hu in enumerate(mean_hu_values):
            rho_e = rho_e_saito(hu)
            mu_h = mew_saito(high_hu[i]) + 1  # Offset by 1 for correction

            # Calculate calibrated rho_e
            rho_e_cal = rho_e_calibrated_saito(hu, alpha[iter], beta[iter])

            # Calculate z_eff
            z_eff = z_eff_saito(mu_h, lambda_val[iter], rho_e_cal)

            # Identify material
            material = categorize_hu_value(hu)

            # Calculate stopping power ratio
            spr = None
            if material in MATERIAL_PROPERTIES:
                material_info = MATERIAL_PROPERTIES[material]
                a = sum(
                    fraction * ELEMENT_PROPERTIES[element]["mass"]
                    for element, fraction in material_info["composition"].items()
                )
                ln_i_m = ln_mean_excitation_potential(z_eff)

                velocity = 299792458 * \
                    math.sqrt(1 - ((0.511 / (dicom_datah.KVP + 0.511)) ** 2))
                beta2 = velocity / 299792458

                spr = sp_truth(z_eff, a, ln_i_m, beta2, dicom_datah.KVP)

            # Store results
            results.append({
                "material": material,
                "rho_e": rho_e,
                "z_eff": z_eff,
                "stopping_power": spr,
                "alpha": alpha[iter],
                "beta": beta[iter],
                "lambda": lambda_val[iter]
            })
            iter += 1
    elif method == "BVM":
        # Placeholder for BVM calculations
        logging.info("BVM method selected but not implemented.")

    else:
        raise HTTPException(
            status_code=400, detail=f"Method {method} not supported.")

    return [{"material": res["material"], "rho_e": res["rho_e"], "z_eff": res["z_eff"], "stopping_power": res["stopping_power"]} for res in results]