import shutil
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from pathlib import Path
import os
from fastapi.staticfiles import StaticFiles
import numpy as np
from fastapi.responses import FileResponse, JSONResponse
import json
import cv2
import logging
import pydicom
from methods.saito import (
    alpha_saito, hu_saito, rho_e_saito, mew_saito, z_eff_saito
)
from methods.spr import sp_truth
from methods.true_attenuation import linear_attenuation
from methods.ln_mean_excitation_potential import ln_mean_excitation_potential
from dect_processing.dect import (
    load_dicom_image,
    categorize_hu_value, save_dicom_as_png, process_and_save_circles, find_first_dicom_file
)

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

    for dicom_file in dicom_files:
        dicom_data = pydicom.dcmread(dicom_file)
        kvp = dicom_data.get("KVP")
        if slice_thickness is None:
            slice_thickness = dicom_data.get("SliceThickness")

        if kvp > 80:  # Assuming high kVp > 80
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

    dicom_file_path = find_first_dicom_file()
    if not dicom_file_path:
        raise HTTPException(
            status_code=404, detail="No DICOM files found in the directory structure")

    # Load the DICOM image and convert pixel values to HU values
    image, dicom_data = load_dicom_image(dicom_file_path)
    mean_hu_values = []
    materials = []
    for circle in saved_circles:
        x, y, radius = circle["x"], circle["y"], circle["radius"]
        mask = np.zeros(image.shape, dtype=np.uint8)
        cv2.circle(mask, (x, y), int(radius * radii_ratios), 1, thickness=-1)
        pixel_values = image[mask == 1]
        mean_hu = np.mean(pixel_values)
        mean_hu_values.append(mean_hu)
    

    results = []
    for hu in mean_hu_values:
        rho_e = rho_e_saito(hu)
        materials.append(categorize_hu_value(hu))
        
    for material in materials:
        # Estimate effective atomic number (Z_eff) and stopping power for each material
        material_results = []
        logging.info(material)

        if material in MATERIAL_PROPERTIES:
            material_info = MATERIAL_PROPERTIES[material]
            
            # linear attenuation of the material
            mu_m = linear_attenuation(material_info)
            # alpha and z_eff for the material
            alpha = alpha_saito(mu_m, rho_e)
            z_eff = z_eff_saito(alpha, [ELEMENT_PROPERTIES[element]["number"] for element in material_info["composition"]])
            
            # calculate stopping power
            a = sum(fraction * ELEMENT_PROPERTIES[element]["mass"] for element, fraction in material_info["composition"].items())
            ln_i_m = ln_mean_excitation_potential(z_eff)
            beta2 = 0.01 # modify if known
            sp_ratio = sp_truth(z_eff, a, ln_i_m, beta2)
            
            # Append the calculated data for this material insert
            results.append({
                "mean_hu_value": hu,
                "material": material,
                "rho_e": rho_e,
                "z_eff": z_eff,
                "stopping_power": sp_ratio
            })
        else:
            # If material is unknown, append the HU and indicate that details are unknown
            results.append({
                "mean_hu_value": hu,
                "material": "unknown",
                "rho_e": None,
                "z_eff": None,
                "stopping_power": None
            })

    return JSONResponse({"results": results})

# Remove processed images folder on shutdown
@app.on_event("shutdown")
def cleanup_processed_images():
    try:
        if os.path.exists(IMAGES_DIR):
            shutil.rmtree(IMAGES_DIR)
            logging.info(f"Deleted the '{IMAGES_DIR}' folder successfully.")
    except Exception as e:
        logging.error(f"Error deleting the '{IMAGES_DIR}' folder: {e}")
