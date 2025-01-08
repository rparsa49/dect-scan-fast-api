from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from pathlib import Path
import os
from PIL import Image
from fastapi.staticfiles import StaticFiles
import pydicom
import numpy as np
from fastapi.responses import FileResponse, JSONResponse
import json
import cv2
import logging
import scipy.constants as const
import math 

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

'''
Below are functions for the various mathematical models.
'''

# Method for linear attenuation of a material
def linear_attenuation(material_info):
    rho = material_info["density"]
    composition = material_info["composition"]
    
    mu_total = 0.0
    for element, fraction in composition.items():
        # get elemental properties
        atomic_mass = ELEMENT_PROPERTIES[element]["mass"]
        atomic_number = ELEMENT_PROPERTIES[element]["number"]
        
        # number density of the element in the material
        N = (rho * fraction) / atomic_mass
        
        mu_a = atomic_number
        
        mu_total += mu_a * N
    return mu_total

# ln of mean excitation potential
def ln_mean_excitation_potential(z_eff):
    if z_eff > 8.5:
        a = 0.098
        b = 3.376
    else:
        a = 0.125
        b = 3.378
    
    return a * z_eff + b

# Saito's methods
def alpha_saito(mew_l, mew_h, mew_lw, mew_hw, rho):
    '''
    Saito 2012
    mew_l: linear attenuation coeff. at low energy
    mew_h: linear attenuation coeff. at high energy
    mew_lw: linear attenuation coeff. of water at low energy
    mew_hw: linear attenuation coeff. of water at high energy
    rho: electron density
    '''
    numerator = 1
    denominator = ((mew_l / mew_lw) - rho) / ((mew_h / mew_hw) - rho) - 1
    return numerator/denominator

def hu_saito(alpha, high, low):
    '''
    Saito 2012
    alpha: weighing factor
    high: CT high
    low: CT low
    '''
    return (1+alpha)*high-(alpha*low)

def rho_e_saito(HU):
    '''
    Saito 2012
    '''
    return (HU/1000) + 1

def mew_saito(HU):
    '''
    Saito 2017
    '''
    return HU/1000 + 1

def z_eff_saito(mew, lam, rho):
    '''
    Saito 2017 
    mew: linear attenuation coefficient (high or low)
    lam: 1/Q(E) is a material independent proportionality constant
    rho: electron density
    '''
    return lam*(mew/rho - 1) + 1

# Stopping power
def sp_truth(z, a, ln_i_m, beta2):
    return 0.307075*(z/a)/beta2*(math.log(2 * 511000.0 * beta2 / (1 - beta2)) - beta2 - ln_i_m)


'''
Below are DECT Processing Functions
'''
# Load circle locations from circle.json
with open("circles.json") as f:
    circle_data = json.load(f)

# Load DICOM image
def load_dicom_image(dicom_path):
    dicom_data = pydicom.dcmread(dicom_path)
    image = dicom_data.pixel_array
    return image, dicom_data

# Convert pixel data to HU  values
def helper_apply_modality_lut(image, dicom_data):
    slope = dicom_data.RescaleSlope
    intercept = dicom_data.RescaleIntercept
    hu_image = image * slope + intercept
    return hu_image

# Calculate mean HU in each circle
def calculate_mean_pixel_value(image, circle):
    x, y, radius = circle
    mask = np.zeros(image.shape, dtype=np.uint8)
    cv2.circle(mask, (x, y), radius, 1, thickness=-1)
    pixel_values = image[mask == 1]
    mean_value = np.mean(pixel_values)
    return mean_value

# Categorize HU value
def categorize_hu_value(hu_value):
    for material, (min_hu, max_hu) in HU_CATEGORIES.items():
        if min_hu <= hu_value <= max_hu:
            return material
    return "unknown"

# Saves DICOM file as png
def save_dicom_as_png(dicom_path: str, save_path: str):
    image, _ = load_dicom_image(dicom_path)
    img = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
    img = img.astype(np.uint8)
    pil_image = Image.fromarray(img)
    pil_image.save(save_path, "PNG")

def process_and_save_circles(image_path, percentage, circles_data, output_path):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Convert to 8-bit image and draw circles
    image_8bit = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    for circle in circles_data:
        center_x, center_y, default_radius = circle["x"], circle["y"], circle["radius"]
        # Scale the radius based on the percentage provided
        final_radius = int(default_radius * (int(percentage) / 100))
        cv2.circle(image_8bit, (center_x, center_y),
                   final_radius, (255, 0, 0), 2)

    # Save the processed image
    cv2.imwrite(str(output_path), image_8bit)

def draw_and_calculate_circles(image, saved_circles, radii_ratios, dicom_data):
    hu_image = helper_apply_modality_lut(image, dicom_data)
    logging.info(f"HU IMAGE: {hu_image}")
    mean_hu_values = []
    logging.info(f"RATIOS: {radii_ratios}")
    for circle in saved_circles:
        x, y, radius = circle["x"], circle["y"], circle["radius"]
        logging.info(f"Circle data \n X: {circle['x']} \n Y: {circle['y']}\n Radius: {circle['radius']}")
        new_radius = int(radius * radii_ratios)
        logging.info(f"New radius: {new_radius}")
        mean_hu_value = calculate_mean_pixel_value(hu_image, (x, y, new_radius))
        mean_hu_values.append(mean_hu_value)

    return mean_hu_values

# Find DICOM files from uploaded directry
def find_first_dicom_file():
    processed_images_dir = Path(IMAGES_DIR)
    for user_folder in processed_images_dir.iterdir():
        if user_folder.is_dir():
            for subfolder in user_folder.iterdir():
                if subfolder.is_dir():
                    dicom_files = sorted(subfolder.glob("*.dcm"))
                    if dicom_files:
                        # Return the first DICOM file found
                        return str(dicom_files[0])
    return None

# Determine material category from HU
def determine_materials(hu_values):
    materials = []
    for hu in hu_values:
        material = "unknown"
        for mat, (min_hu, max_hu) in HU_CATEGORIES.items():
            if min_hu <= hu <= max_hu:
                material = mat
                break
        materials.append(material)
    return materials

# Alternate HU calculation
def calculate_hu(material_mu, water_mu):
    return ((material_mu - water_mu) / water_mu) * 100


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
    if phantom_type not in circle_data:
        raise HTTPException(status_code=400, detail="Invalid phantom type.")
    circles_data = circle_data[phantom_type]

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
    saved_circles = circle_data[phantom_type]

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
