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

supported_models = {
    "Saito": {
        "description": "Saito's model for calculating stopping power",

    },
    "Hunemohr": {
        "description": "Hunemohr's model for calculating stopping power",
    }
}

# Define HU material categories (Greenway K, Campos A, Knipe H, et al. Hounsfield unit. Reference article, Radiopaedia.org (Accessed on 12 Nov 2024) https://doi.org/10.53347/rID-38181)
HU_CATEGORIES = {
    "Cortical Bone": (1000, float("inf")),
    "Brain": (30, 40),
    "Liver": (45, 50),
    "Lung": (-950, -650),
    "Muscle": (45, 50),
    "Adipose": (-150, -50),
    "True Water": (0, 0),
    "Solid Water": (0, 0),
    "Breast": (41, 134),
    "Air": (-1000, -1)
}

# Elemental Properties
ELEMENT_PROPERTIES = {
    'H': {
        "number": 1,
        "mass": 1.00794,
        "ionization": 19.2
    },
    'B': {
        "number": 5,
        "mass": 10.811,
        "ionization": 76.
    },
    'C': {
        "number": 6,
        "mass": 12.0107,
        "ionization": 78.
    },
    'N': {
        "number": 7,
        "mass": 14.0067,
        "ionization": 82.
    },
    'O': {
        "number": 8,
        "mass": 15.9994,
        "ionization": 95.
    },
    'F': {
        "number": 9,
        "mass": 18.998,
        "ionization": 115.
    },
    'Na': {
        "number": 11,
        "mass": 22.98977,
        "ionization": 149.
    },
    'Mg': {
        "number": 12,
        "mass": 24.305,
        "ionization": 156.
    },
    'Al': {
        "number": 13,
        "mass": 26.9815,
        "ionization": 166.
    },
    'Si': {
        "number": 14,
        "mass": 28.0855,
        "ionization": 173.
    },
    'P': {
        "number": 15,
        "mass": 30.9738,
        "ionization": 173.
    },
    'Cl': {
        "number": 17,
        "mass": 35.453,
        "ionization": 174.
    },
    'Ca': {
        "number": 20,
        "mass": 40.078,
        "ionization": 191.
    },
    'Ti': {
        "number": 22,
        "mass": 47.867,
        "ionization": 233.
    },
}

# Material Properties Truth
MATERIAL_PROPERTIES = {
    "Lung": {
        "rho": 0.46,
        "rho_e_w": 0.44,
        "Z_eff": 7.46,
        "composition": {'H': 0.0743, 'O': 0.2071, 'C': 0.5786, 'N': 0.0196, 'Cl': 0.0008, 'Si': 0.0077, 'Mg': 0.1119},
        "density": 0.29
    },
    "Adipose": {
        "rho": 0.94,
        "rho_e_w": 0.93,
        "Z_eff": 6.17,
        "composition": {'H': 0.0973, 'O': 0.1435, 'C': 0.7141, 'N': 0.0271, 'Cl': 0.0012, 'Ca': 0.0034, 'Si': 0.0111, 'B': 0.0005, 'Na': 0.0018},
        "density": 0.96
    },
    "Breast": {
        "rho": 0.99,
        "rho_e_w": 0.97,
        "Z_eff": 6.81,
        "composition": {'H': 0.0948, 'O': 0.1513, 'C': 0.7023, 'N': 0.0247, 'Cl': 0.0012, 'Ca': 0.0074, 'Si': 0.0101, 'B': 0.0004, 'Na': 0.0016, 'Mg': 0.0061},
        "density": 0.98
    },
    "True Water": {
        "rho": 1,
        "rho_e_w": 1,
        "Z_eff": 7.45,
        "composition": {'H': 0.1119, 'O': 0.8881},
        "density": 1.00
    },
    "Solid Water": {
        "rho": 1.02,
        "rho_e_w": 0.99,
        "Z_eff": 7.5,
        "composition": {'H': 0.800, 'C': 0.6730, 'N': 0.0239, 'O': 0.1987, 'Cl': 0.0014, 'Ti': 0.0231},
        "density": 1.018
    },
    "Muscle": {
        "rho": 1.05,
        "rho_e_w": 1.02,
        "Z_eff": 7.55,
        "composition": {'H': 0.0810, 'C': 0.6717, 'N': 0.0242, 'O': 0.1985, 'Cl': 0.0014, 'Ti': 0.0232},
        "density": 1.049
    },
    "Brain": {
        "rho": 1.05,
        "rho_e_w": 1.05,
        "Z_eff": 6.05,
        "composition": {'H': 0.0823, 'O': 0.1970, 'C': 0.6575, 'N': 0.0205, 'Cl': 0.0013, 'Ca': 0.0179, 'Si': 0.0094, 'B': 0.0004, 'Na': 0.0015, 'Mg': 0.0123},
        "density": 1.05 
    },
    "Liver": {
        "rho": 1.09,
        "rho_e_w": 1.06,
        "Z_eff": 7.55,
        "composition": {'H': 0.0825, 'O': 0.1902, 'C': 0.6687, 'N': 0.0225, 'Cl': 0.0014, 'Ca': 0.0194, 'Si': 0.0065, 'B': 0.0003, 'Na': 0.0010, 'Mg': 0.0075},
        "density": 1.08
    },
    "Inner Bone": {
        "rho": 1.15,
        "rho_e_w": 1.10,
        "Z_eff": 10.14,
        "composition": {'H': 0.0638, 'O': 0.2564, 'C': 0.5379, 'N': 0.0173, 'Cl': 0.0010, 'Ca': 0.0982, 'Si': 0.0072, 'B': 0.0003, 'Na': 0.0011, 'Mg': 0.0168},
        "density": 1.21
    },
    "B200": {
        "rho": 1.15,
        "rho_e_w": 1.11,
        "Z_eff": 10.15,
        "composition": {'H': 0.0665, 'C': 0.5552, 'N': 0.0198, 'O': 0.2364, 'P': 0.0324, 'Ca': 0.0887},
        "density": 1.153
    },
    "CB30": {
        "rho": 1.33,
        "rho_e_w": 1.28,
        "Z_eff": 10.61,
        "composition": {'H': 0.0668, 'C': 0.5348, 'N': 0.0212, 'O': 0.2561, 'Ca': 0.1201},
        "density": 1.333
    },
    "CB50": {
        "rho": 1.56,
        "rho_e_w": 1.47,
        "Z_eff": 12.26,
        "composition": {'H': 0.0477, 'C': 0.4163, 'N': 0.0152, 'O': 0.3200, 'Ca': 0.2002},
        "density": 1.560
    },
    "Cortical Bone": {
        "rho": 1.82,
        "rho_e_w": 1.70,
        "Z_eff": 13.38,
        "composition": {'H': 0.0341, 'C': 0.3141, 'N': 0.0184, 'O': 0.3650, 'Ca': 0.2681},
        "density": 1.823
    },
    "Air": {
        "rho": 0,
        "rho_e_w": 0,
        "Z_eff": 7.5,
        "composition": {'N': 0.78, 'O': 0.2095},
        "density": 0
    }
}

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
def alpha_saito(mew_m, rho_e, mew_w=0.268):
    return 1 / ((mew_m / mew_w) - rho_e * (mew_m - 1))

def rho_e_saito(HU):
    return (HU/1000) + 1

def z_eff_saito(alpha, Z):
    sum_term = sum(alpha * (Z_i ** 3.5) for Z_i in Z)
    return sum_term ** (1/3.5)

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
    return JSONResponse(supported_models)

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
