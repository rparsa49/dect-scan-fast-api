import math, os, json, cv2, logging, pydicom, numpy as np
from fastapi import FastAPI, Form, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from methods.saito import saito
from methods.hunemohr import hunemohr
from methods.tanaka import tanaka
from dect_processing.dect import (save_dicom_as_png, process_and_save_circles)
from dect_processing.organize import convert_numpy
import shutil 
import tensorflow as tf

logging.basicConfig(level=logging.INFO)
app = FastAPI()

app.mount("/processed_i`mages",
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

UPLOADED_DIR = ""

# Load JSON data files
DATA_DIR = Path("data")

def load_json(file_name):
    with open(DATA_DIR / file_name, "r") as file:
        return json.load(file)

HU_CATEGORIES = load_json("hu_categories.json")
ELEMENT_PROPERTIES = load_json("element_properties.json")
MATERIAL_PROPERTIES = load_json("material_properties.json")
CIRCLE_DATA = load_json("circles.json")
WATER_ATTENUATION = load_json("water_att.json")
ELEMENT_ATOMIC_NUMBERS = load_json("atomic_numbers.json")

'''
API CALLS
'''
@app.get("/get-supported-models")
async def get_supported_models():
    models = {
        "tanaka": {"name": "Tanaka"},
        "saito": {"name": "Saito"},
        "hunemohr": {"name": "Hunemohr"}
    }
    return JSONResponse(models)


@app.post("/upload-scan")
async def upload_scan(files: List[UploadFile] = File(...)):
    dicom_files = []
    for file in files:
        file_path = os.path.join(IMAGES_DIR, file.filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(await file.read())
        dicom_files.append(file_path)

    high_kvp_files = []
    low_kvp_files = []
    slice_thickness = None
    ref_dcm = pydicom.dcmread(dicom_files[0])
    ref_kvp = ref_dcm.get("KVP")

    logging.info(f"Reference KVP is {ref_kvp}")

    for dicom_file in dicom_files:
        dicom_data = pydicom.dcmread(dicom_file)
        kvp = dicom_data.get("KVP")

        if slice_thickness is None:
            slice_thickness = dicom_data.get("SliceThickness")

        if kvp >= ref_kvp:
            high_kvp_files.append(dicom_file)
        else:
            low_kvp_files.append(dicom_file)

    high_kvp_image_paths = []
    low_kvp_image_paths = []

    for high_file in high_kvp_files:
        original_name = os.path.basename(high_file).replace(".dcm", ".png")
        high_img_path = os.path.join(IMAGES_DIR, original_name)
        save_dicom_as_png(high_file, high_img_path)
        high_kvp_image_paths.append(f"/get-image/{original_name}")

    for low_file in low_kvp_files:
        original_name = os.path.basename(low_file).replace(".dcm", ".png")
        low_img_path = os.path.join(IMAGES_DIR, original_name)
        save_dicom_as_png(low_file, low_img_path)
        low_kvp_image_paths.append(f"/get-image/{original_name}")

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

    if phantom_type not in CIRCLE_DATA:
        raise HTTPException(status_code=400, detail="Invalid phantom type.")
    circles_data = CIRCLE_DATA[phantom_type]

    updated_high_images = []
    updated_low_images = []

    for image_path in high_kvp_images:
        image_name = image_path.split("/get-image/")[-1]
        input_path = os.path.join(IMAGES_DIR, image_name)
        process_and_save_circles(input_path, percentage, circles_data, input_path)
        updated_high_images.append(f"/get-image/{image_name}")

    for image_path in low_kvp_images:
        image_name = image_path.split("/get-image/")[-1]
        input_path = os.path.join(IMAGES_DIR, image_name)
        process_and_save_circles(input_path, percentage, circles_data, input_path)
        updated_low_images.append(f"/get-image/{image_name}")

    return JSONResponse({
        "updated_high_kvp_images": updated_high_images,
        "updated_low_kvp_images": updated_low_images
    })

@app.post("/clean-noise")
async def clean_noise(request: Request):
    data = await request.json()
    # load the images
    high_path = data.get("high_kvp_image")
    low_path = data.get("low_kvp_image")
    # load model
    model = tf.keras.models.load_model("my_model.h5")
    
    # high image
    dicom_data = pydicom.dcmread(high_path)
    image = dicom_data.pixel_array.astype(np.float32)
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis = 0)
    denoised_image_high = model.predict(image)
    denoised_image_high = np.squeeze(denoised_image_high)
    
    # saving images and returning
    original_name = os.path.basename(high_path).replace(".dcm", ".png")
    high_img_path = os.path.join(IMAGES_DIR, original_name, "clean")
    save_dicom_as_png(denoised_image_low, high_img_path)
    new_high = (f"/get-image/{original_name}")


    # low image
    dicom_data = pydicom.dcmread(low_path)
    image = dicom_data.pixel_array.astype(np.float32)
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0)
    denoised_image_low = model.predict(image)
    denoised_image_low = np.squeeze(denoised_image_low)
    
    # saving images and returning
    original_name = os.path.basename(low_path).replace(".dcm", ".png")
    low_img_path = os.path.join(IMAGES_DIR, original_name, "clean")
    save_dicom_as_png(denoised_image_low, low_img_path)
    new_low = (f"/get-image/{original_name}")
    
    return {
        "high": new_high,
        "low": new_low
    }

    
@app.post("/analyze-inserts")
async def analyze_inserts(request: Request):
    data = await request.json()
    radii_ratios = data.get("radius", [1.0])
    radii_ratios = int(radii_ratios) / 100 
    phantom_type = data.get("phantom") 
    method_type = data.get("model")
    high_path = data.get("high_kvp_image")
    low_path = data.get("low_kvp_image")
    
    logging.info(f"High path is: {high_path}")
    logging.info(f"Low path is: {low_path}")
    
    high_name = high_path.split("/get-image/")[-1]
    high_name = os.path.basename(high_name).replace(".png", ".dcm")
    high_name = os.path.join(IMAGES_DIR, "test-data", "high", high_name)
    
    low_name = low_path.split("/get-image/")[-1]
    low_name = os.path.basename(low_name).replace(".png", ".dcm")
    low_name = os.path.join(IMAGES_DIR, "test-data", "low", low_name)
    
    if method_type == "Saito":
        results = saito(high_name, low_name, phantom_type, radii_ratios)
        results = json.loads(results)
    elif method_type == "Hunemohr":
        results = hunemohr(high_name, low_name, phantom_type, radii_ratios)
        results = json.loads(results)
    elif method_type == "Tanaka":
        results = tanaka(high_name, low_name, phantom_type, radii_ratios)
        results = json.loads(results)
    else:
        return JSONResponse({"error": "Invalid method type"}, status_code=400)

    return JSONResponse({"results": {k: convert_numpy(v) for k, v in results.items()}})


@app.post("/go-back")
async def go_back(request: Request):
    # Check if directory exists
    if os.path.exists(IMAGES_DIR) and os.path.isdir(IMAGES_DIR):
        # Remove all files and subdirectories inside the directory
        for filename in os.listdir(IMAGES_DIR):
            file_path = os.path.join(IMAGES_DIR, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Remove files and symbolic links
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Remove subdirectories
            except Exception as e:
                return {"error": f"Failed to delete {file_path}: {str(e)}"}

    return {"message": "Processed images directory cleaned successfully"}
