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
import matplotlib.pyplot as plt

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
BASE_DICOM_HIGH = ""
BASE_DICOM_LOW = ""

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

# Given a folder with two subfolders containing DICOM files, determine which one is the high KVP folder
def identify_high_low_dirs(main_folder):
    subdirs = [os.path.join(main_folder, d) for d in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, d))]
    
    if len(subdirs) != 2:
        raise ValueError("Upload must contain exactly two subfolders with DICOMs.")
    
    kvps = []
    st = []
    for subdir in subdirs:
        dcm_files = [f for f in os.listdir(subdir) if f.lower().endswith(".dcm")]
        if not dcm_files:
            raise ValueError(f"No DICOM files found in {subdir}")
        dcm = pydicom.dcmread(os.path.join(subdir, dcm_files[0]))
        kvp = dcm.get("KVP")
        st.append(dcm.get("SliceThickness"))
        if kvp is None:
            raise ValueError(f"No KVP in file {dcm_files[0]}")
        kvps.append((kvp, subdir))
    
    kvps.sort(reverse=True)
    return kvps[0][1], kvps[1][1], st[0]

@app.post("/upload-scan")
async def upload_scan(files: List[UploadFile] = File(...)):
    dicom_files = []
    for file in files:
        file_path = os.path.join(IMAGES_DIR, file.filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(await file.read())
        dicom_files.append(file_path)

    # Identify session folder
    root_paths = set(Path(f).parents[1] for f in dicom_files)
    if len(root_paths) != 1:
        raise HTTPException(status_code=400, detail="All files must be inside a single root folder.")
    session_folder = str(next(iter(root_paths)))
    
    try:
        high_path, low_path, st = identify_high_low_dirs(session_folder)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    global BASE_DICOM_HIGH, BASE_DICOM_LOW, SLICE_THICKNESS, ROOT_PATH
    BASE_DICOM_HIGH = high_path
    BASE_DICOM_LOW = low_path
    SLICE_THICKNESS = st
    ROOT_PATH = session_folder
    
    high_kvp_files = [str(p) for p in Path(high_path).glob("*dcm")]
    low_kvp_files = [str(p) for p in Path(low_path).glob("*dcm")]

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
        "slice_thickness": SLICE_THICKNESS
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

def convert_to_dicom_path(image_url, is_high=True):
    filename = os.path.basename(image_url).replace(".png", ".dcm")
    base_subfolder = BASE_DICOM_HIGH if is_high else BASE_DICOM_LOW

    if not base_subfolder or not ROOT_PATH:
        raise RuntimeError("BASE_DICOM paths or ROOT_PATH not set")

    # Get only the final subfolder name, e.g., "SubfolderA"
    subfolder_name = os.path.basename(base_subfolder)

    # Now construct: processed_images/<main folder>/<subfolder>/<filename>
    full_path = os.path.join(ROOT_PATH, subfolder_name, filename)

    if not os.path.exists(full_path):
        raise FileNotFoundError(
            f"[convert_to_dicom_path] Missing: {full_path}")

    return full_path

def preprocess_image(dicom_path):
    dicom_data = pydicom.dcmread(dicom_path)
    image = dicom_data.pixel_array.astype(np.float32)
    image = (image - np.min(image)) / \
        (np.max(image) - np.min(image))  # normalize 0-1
    image = cv2.resize(image, (128, 128))  # resize to model input
    image = image.flatten()
    image = np.expand_dims(image, axis=0)
    return image

def save_clean_as_png(image, save_path):
    plt.imsave(save_path, image, cmap="gray")


def denoise_and_save(image_url, is_high=True):
    dicom_path = convert_to_dicom_path(image_url, is_high)
    dcm = pydicom.dcmread(dicom_path)
    original_image = dcm.pixel_array.astype(np.float32)

    normalized = (original_image - np.min(original_image)) / \
        (np.max(original_image) - np.min(original_image))
    resized = cv2.resize(normalized, (128, 128), interpolation=cv2.INTER_AREA)

    noise_factor = 0.2
    noisy_input = resized + noise_factor * \
        np.random.normal(loc=0.0, scale=1.0, size=resized.shape)
    noisy_input = np.clip(noisy_input, 0., 1.)

    input_tensor = noisy_input.reshape(1, -1).astype(np.float32)

    model = tf.keras.models.load_model("dense.h5")
    denoised = model.predict(input_tensor)
    denoised = np.squeeze(denoised).reshape(128, 128)

    # === Resize to match original shape ===
    original_shape = original_image.shape
    denoised_resized = cv2.resize(
        denoised, original_shape[::-1], interpolation=cv2.INTER_CUBIC)


    # === Replace original DICOM pixel data with denoised version ===
    slope = float(dcm.get("RescaleSlope", 1))
    intercept = float(dcm.get("RescaleIntercept", 0))

    denoised_scaled = (denoised_resized * (np.max(original_image) -
                   np.min(original_image))) + np.min(original_image)
    denoised_scaled = denoised_scaled * slope + intercept

    #Overwrite the original .dcm file (or change to save to new path if preferred)
    dcm.save_as(dicom_path) 
    
    name = os.path.basename(dicom_path).replace(".dcm", ".png")
    save_path = os.path.join(IMAGES_DIR, name)
    save_clean_as_png(denoised_resized, save_path)

    return f"/get-image/{name}"

@app.post("/clean-noise")
async def clean_noise(request: Request):
    data = await request.json()

    high_image_url = data.get("high_kvp_image")
    low_image_url = data.get("low_kvp_image")

    if not high_image_url or not low_image_url:
        return {"error": "Missing image paths"}

    new_high = denoise_and_save(high_image_url, is_high=True)
    new_low = denoise_and_save(low_image_url, is_high=False)
    
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

    # Strip cache-busting parameters like ?t=123456
    # high_name = os.path.basename(
    #     high_path.split("?")[0]).replace(".png", ".dcm")
    # low_name = os.path.basename(low_path.split("?")[0]).replace(".png", ".dcm")

    high_name = convert_to_dicom_path(high_path, is_high=True)
    low_name = convert_to_dicom_path(low_path, is_high=False)

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

@app.post("/reset-processed")
async def reset_processed_folder():
    try:
        for filename in os.listdir(IMAGES_DIR):
            file_path = os.path.join(IMAGES_DIR, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        return {"message": "✅ Processed images folder cleared successfully."}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error clearing folder: {str(e)}")
