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
Path(IMAGES_DIR).mkdir(exist_ok=True)
supported_models = {
    "Saito": {
        "description": "",

    },
    "Hunemohr": {
        "description": "",
    }
}

# Define HU material categories (Greenway K, Campos A, Knipe H, et al. Hounsfield unit. Reference article, Radiopaedia.org (Accessed on 12 Nov 2024) https://doi.org/10.53347/rID-38181)
HU_CATEGORIES = {
    "air": (-1000, -1000),
    "bone (cortical)": (1000, float("inf")),
    "bone (trabecular)": (300, 800),
    "brain (grey matter)": (40, 40),
    "brain (white matter)": (30, 30),
    "subcutaneous fat": (-115, -100),
    "liver": (45, 50),
    "lungs": (-950, -650),
    "metal": (3000, float("inf")),
    "muscle": (45, 50),
    "renal cortex": (25, 30),
    "spleen": (40, 45),
    "water": (0, 0)
}

# Load circle locations from circle.json
with open("circles.json") as f:
    circle_data = json.load(f)

# Load DICOM image
def load_dicom_image(dicom_path):
    dicom_data = pydicom.dcmread(dicom_path)
    image = dicom_data.pixel_array
    return image, dicom_data

# Convert pixel data to HU  values
def apply_modality_lut(image, dicom_data):
    return apply_modality_lut(image, dicom_data)

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

# Categorize inserts and receive HU
@app.post("/analyze-inserts")
async def analyze_inserts(request: Request):
    data = await request.json()
    images = data.get("images", [])
    circles = data.get("circles", [])
    model_type = data.get("model_type")

    if not images or not circles or not model_type:
        raise HTTPException(status_code=400, detail="Missing required data.")

    analysis_results = []

    for image_path in images:
        # Load DICOM image
        dicom_path = os.path.join(IMAGES_DIR, image_path)
        if not os.path.exists(dicom_path):
            raise HTTPException(
                status_code=404, detail=f"Image {image_path} not found")

        image, dicom_data = load_dicom_image(dicom_path)
        hu_image = apply_modality_lut(image, dicom_data)

        # Analyze each circle
        circle_results = []
        for circle in circles:
            x, y, radius_percentage = circle["x"], circle["y"], circle["radius"]
            scaled_radius = int(radius_percentage *
                                circle["default_radius"] / 100)
            mean_hu_value = calculate_mean_pixel_value(
                hu_image, (x, y, scaled_radius))
            material = categorize_hu_value(mean_hu_value)
            circle_results.append({
                "center": (x, y),
                "scaled_radius": scaled_radius,
                "mean_hu_value": mean_hu_value,
                "material": material
            })

        analysis_results.append({
            "image": image_path,
            "model_type": model_type,
            "circles": circle_results
        })

    return JSONResponse({"results": analysis_results})
