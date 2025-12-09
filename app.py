import os
import json
import cv2
import logging
import pydicom
import numpy as np
from fastapi import FastAPI, Form, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from methods.saito import saito, saito_test
from methods.hunemohr import hunemohr, hunemohr_test
from methods.tanaka import tanaka, tanaka_test
from methods.schneider import schneider, test_schneider
from dect_processing.dect import (save_dicom_as_png, process_and_save_circles)
from dect_processing.organize import convert_numpy
import shutil
import tensorflow as tf
import matplotlib.pyplot as plt
import re
from typing import Callable, Any, Dict, List, Tuple
from pydicom.dataset import FileDataset
from pydicom.uid import generate_uid, ExplicitVRLittleEndian
from datetime import datetime
from matplotlib.colors import Normalize

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
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
BASE_DICOM_HIGH = ""
BASE_DICOM_LOW = ""
IS_SECT = False
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

# Noise Benchmarking Constants
DATA_LOCO = Path("test_images")
KVP_PAIRS = [(70, 100), (70, 120), (70, 140), (80, 100), (80, 120), (80, 140)]
SERIES_RE = re.compile(
    r'^(?:degraded-)?(.+)-(\d+(?:\.\d+)?)-(\d+)$', re.IGNORECASE)
NOISY_SERIES_RE = re.compile(
    r'^(?:degraded-)?(.+)-(\d+(?:\.\d+)?)-(\d+)-(\d+(?:\.\d+)?)$', re.IGNORECASE)
VAR = [0.01, 0.05, 0.1]
MAX_SLICES = 10
TEMP_ROOT = "temp_noise_series"

'''
API CALLS
'''


@app.get("/get-supported-models")
async def get_supported_models():
    models = {
        "tanaka": {"name": "Tanaka"},
        "saito": {"name": "Saito"},
        "hunemohr": {"name": "Hunemohr"},
        "schneider": {"name": "Schneider"}
    }
    return JSONResponse(models)

# Given a folder with two subfolders containing DICOM files, determine which one is the high KVP folder
def identify_high_low_dirs(main_folder):
    # Get all subdirectories
    subdirs = [os.path.join(main_folder, d) for d in os.listdir(
        main_folder) if os.path.isdir(os.path.join(main_folder, d))]

    # CASE 1: Dual Energy (DECT) - Two subfolders
    if len(subdirs) == 2:
        kvps = []
        st = []
        for subdir in subdirs:
            dcm_files = [f for f in os.listdir(
                subdir) if f.lower().endswith(".dcm")]
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

    # CASE 2: Single Energy (SECT) - One subfolder
    elif len(subdirs) == 1:
        subdir = subdirs[0]
        dcm_files = [f for f in os.listdir(
            subdir) if f.lower().endswith(".dcm")]
        if not dcm_files:
            # If the subdir is empty, check if the files are actually in the root
            pass
        else:
            dcm = pydicom.dcmread(os.path.join(subdir, dcm_files[0]))
            st = dcm.get("SliceThickness", 1.0)
            return subdir, [], st

    # CASE 3: Single Energy (SECT) - Files in root folder (No subfolders)
    # Check if there are DICOMs directly in the main_folder
    root_dcm_files = [f for f in os.listdir(
        main_folder) if f.lower().endswith(".dcm")]

    if len(root_dcm_files) > 0:
        dcm = pydicom.dcmread(os.path.join(main_folder, root_dcm_files[0]))
        st = dcm.get("SliceThickness", 1.0)
        # Return main_folder as the high path, empty list for low, and thickness
        return main_folder, [], st

    # CASE 4: Invalid Structure
    raise ValueError(
        "Upload must contain either DICOM files in the root (SECT), one subfolder (SECT), or two subfolders (DECT).")


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
        raise HTTPException(
            status_code=400, detail="All files must be inside a single root folder.")
    session_folder = str(next(iter(root_paths)))

    try:
        high_path, low_path, st = identify_high_low_dirs(session_folder)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    global BASE_DICOM_HIGH, BASE_DICOM_LOW, SLICE_THICKNESS, ROOT_PATH, IS_SECT
    BASE_DICOM_HIGH = high_path
    BASE_DICOM_LOW = low_path
    SLICE_THICKNESS = st
    ROOT_PATH = session_folder
    IS_SECT = (low_path is None)  # Set the flag based on identification

    high_kvp_image_paths = []
    low_kvp_image_paths = []

    # Process High KVP (Always present)
    high_kvp_files = [str(p) for p in Path(high_path).glob("*dcm")]
    for high_file in high_kvp_files:
        original_name = os.path.basename(high_file).replace(".dcm", ".png")
        high_img_path = os.path.join(IMAGES_DIR, original_name)
        save_dicom_as_png(high_file, high_img_path)
        high_kvp_image_paths.append(f"/get-image/{original_name}")

    # Process Low KVP (Only if DECT)
    if not IS_SECT and low_path:
        low_kvp_files = [str(p) for p in Path(low_path).glob("*dcm")]
        for low_file in low_kvp_files:
            original_name = os.path.basename(low_file).replace(".dcm", ".png")
            low_img_path = os.path.join(IMAGES_DIR, original_name)
            save_dicom_as_png(low_file, low_img_path)
            low_kvp_image_paths.append(f"/get-image/{original_name}")

    return {
        "high_kvp_images": high_kvp_image_paths,
        "low_kvp_images": low_kvp_image_paths,
        "slice_thickness": SLICE_THICKNESS,
        "is_sect": IS_SECT
    }


@app.post("/upload-calibration")
async def upload_calibration(calibration_file: UploadFile = File(...)):
    if not calibration_file.filename.endswith(".json"):
        raise HTTPException(
            status_code=400, detail="Invalid file type. Please upload a JSON file.")

    try:
        content = await calibration_file.read()
        calibration_data = json.loads(content)
        return JSONResponse(calibration_data)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format.")
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing calibration file: {str(e)}")

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
        process_and_save_circles(
            input_path, percentage, circles_data, input_path)
        updated_high_images.append(f"/get-image/{image_name}")

    for image_path in low_kvp_images:
        image_name = image_path.split("/get-image/")[-1]
        input_path = os.path.join(IMAGES_DIR, image_name)
        process_and_save_circles(
            input_path, percentage, circles_data, input_path)
        updated_low_images.append(f"/get-image/{image_name}")

    return JSONResponse({
        "updated_high_kvp_images": updated_high_images,
        "updated_low_kvp_images": updated_low_images
    })


def convert_to_dicom_path(image_url, is_high=True):
    filename = os.path.basename(image_url).replace(".png", ".dcm")

    # Graceful handling for SECT
    if not is_high and IS_SECT:
        return None

    base_subfolder = BASE_DICOM_HIGH if is_high else BASE_DICOM_LOW

    if not base_subfolder:
        # If trying to access low folder but it doesn't exist
        if not is_high:
            return None
        raise RuntimeError("BASE_DICOM paths not set properly.")

    if not ROOT_PATH:
        raise RuntimeError("ROOT_PATH not set")

    subfolder_name = os.path.basename(base_subfolder)
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

    # Overwrite the original .dcm file (or change to save to new path if preferred)
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


@app.post("/test-calibration")
async def test_calibration(calibration_file: UploadFile = File(...), files: List[UploadFile] = File(...)):
    if not calibration_file.filename.endswith(".json"):
        raise HTTPException(
            status_code=400, detail="Invalid calibration file type. Please upload a JSON file.")

    # 1. Process the calibration file
    try:
        content = await calibration_file.read()
        calibration_data = json.loads(content)
        model_name = calibration_data.get("model")
        if not model_name:
            raise HTTPException(
                status_code=400, detail="Model name not found in calibration file.")
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format.")

    # 2. Process the DICOM files
    dicom_files_list = []
    for file in files:
        file_path = os.path.join(IMAGES_DIR, file.filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(await file.read())
        dicom_files_list.append(file_path)

    root_paths = set(Path(f).parents[1] for f in dicom_files_list)
    if len(root_paths) != 1:
        raise HTTPException(
            status_code=400, detail="All files must be inside a single root folder.")
    session_folder = str(next(iter(root_paths)))

    try:
        high_path, low_path, _ = identify_high_low_dirs(session_folder)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    global BASE_DICOM_HIGH, BASE_DICOM_LOW, ROOT_PATH, IS_SECT
    BASE_DICOM_HIGH = high_path
    BASE_DICOM_LOW = low_path
    ROOT_PATH = session_folder
    IS_SECT = (low_path is None)

    high_kvp_image_paths = []
    low_kvp_image_paths = []

    high_kvp_files = [str(p) for p in Path(high_path).glob("*dcm")]
    for high_file in high_kvp_files:
        original_name = os.path.basename(high_file).replace(".dcm", ".png")
        high_img_path = os.path.join(IMAGES_DIR, original_name)
        save_dicom_as_png(high_file, high_img_path)
        high_kvp_image_paths.append(f"/get-image/{original_name}")

    if not IS_SECT and low_path:
        low_kvp_files = [str(p) for p in Path(low_path).glob("*dcm")]
        for low_file in low_kvp_files:
            original_name = os.path.basename(low_file).replace(".dcm", ".png")
            low_img_path = os.path.join(IMAGES_DIR, original_name)
            save_dicom_as_png(low_file, low_img_path)
            low_kvp_image_paths.append(f"/get-image/{original_name}")

    # 3. Analyze inserts using the provided calibration data
    method_type = calibration_data.get("model")

    # Pick first image for analysis
    high_dicom_path = convert_to_dicom_path(
        high_kvp_image_paths[0], is_high=True)
    low_dicom_path = convert_to_dicom_path(
        low_kvp_image_paths[0], is_high=False) if not IS_SECT and low_path else None

    try:
        # --- SCHNEIDER TEST LOGIC ---
        if method_type == "Schneider":
            ed_params = calibration_data.get("ed_params")
            spr_params = calibration_data.get("spr_params")
            phantom_used = calibration_data.get("phantom", "body")

            # Default radius 100% if not specified
            results_dict = test_schneider(
                high_dicom_path, phantom_used, 1.0, ed_params, spr_params)

            return JSONResponse({
                "high_kvp_images": high_kvp_image_paths,
                "low_kvp_images": low_kvp_image_paths,
                "analysis_results": results_dict,
                "model": model_name
            })

        # --- DECT TEST LOGIC ---

        # Safely extract params
        params = {
            "alpha": float(calibration_data.get("alpha", 0.0)),
            "a": float(calibration_data.get("a", 0.0)),
            "b": float(calibration_data.get("b", 0.0)),
            "r": float(calibration_data.get("r", 0.0)),
            "gamma": float(calibration_data.get("gamma", 0.0)),
            "c": float(calibration_data.get("c", 0.0)),
            "c0": float(calibration_data.get("c0", 0.0)),
            "c1": float(calibration_data.get("c1", 0.0)),
        }

        if method_type == "Saito":
            analysis_result_str = saito_test(
                high_dicom_path, low_dicom_path, "head", 1,
                params["alpha"], params["a"], params["b"], params["r"], params["gamma"]
            )
        elif method_type == "Hunemohr":
            analysis_result_str = hunemohr_test(
                high_dicom_path, low_dicom_path, "head", 1,
                params["a"], params["b"], params["c"]
            )
        elif method_type == "Tanaka":
            analysis_result_str = tanaka_test(
                high_dicom_path, low_dicom_path, "head", 1,
                params["alpha"], params["a"], params["b"], params["gamma"], params["c0"], params["c1"]
            )
        else:
            raise HTTPException(status_code=400, detail="Invalid method type.")

        analysis_result = json.loads(analysis_result_str)

        return JSONResponse({
            "high_kvp_images": high_kvp_image_paths,
            "low_kvp_images": low_kvp_image_paths,
            "analysis_results": analysis_result,
            "model": model_name
        })

    except Exception as e:
        logger.error(f"Error running test: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error running test: {str(e)}")


@app.post("/analyze-inserts")
async def analyze_inserts(request: Request):
    data = await request.json()
    radii_ratios = data.get("radius", [1.0])
    radii_ratios = float(radii_ratios)

    phantom_type = data.get("phantom")
    method_type = data.get("model")
    high_path = data.get("high_kvp_image")
    low_path = data.get("low_kvp_image")

    logging.info(f"High path is: {high_path}")
    logging.info(f"Low path is: {low_path}")

    high_name = convert_to_dicom_path(high_path, is_high=True)
    low_name = convert_to_dicom_path(
        low_path, is_high=False) if low_path else None

    # --- SCHNEIDER CALIBRATION LOGIC ---
    if method_type == "Schneider":
        if not high_name:
            raise HTTPException(
                status_code=400, detail="High KVP image required for Schneider")

        try:
            # Run the training function to get params
            ed_params, spr_params = schneider(
                high_name, phantom_type, radii_ratios)

            # Construct Calibration File Structure
            calibration_data = {
                "model": "Schneider",
                "phantom": phantom_type,
                "created_at": datetime.now().isoformat(),
                "ed_params": ed_params,
                "spr_params": spr_params
            }
            # Return this JSON so the frontend can prompt user to save it
            return JSONResponse(calibration_data)
        except Exception as e:
            logger.error(f"Schneider calibration failed: {e}")
            raise HTTPException(
                status_code=500, detail=f"Calibration failed: {e}")

    # --- DECT METHODS LOGIC ---
    if not high_name or not low_name:
        return JSONResponse({"error": "Dual Energy models require both High and Low KVP images."}, status_code=400)

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

# api for benchmarking tests


@app.post("/benchmark-noise")
async def benchmark_noise(request: Request):
    data = await request.json()
    model_name = data.get("model")
    phantom_type = data.get("phantom_type")
    radii_ratio = data.get("radii_ratio", [1.0])

    if not model_name or not phantom_type:
        raise HTTPException(
            status_code=400, detail="Model and phantom type are required")

    methods_map = {
        "saito": saito,
        "hunemohr": hunemohr,
        "tanaka": tanaka
    }

    if model_name not in methods_map:
        raise HTTPException(
            status_code=400, detail=f"Unsupported model: {model_name}")

    method_fn = methods_map[model_name]

    series_clean = ROOT_PATH

    try:
        results = run_noise_benchmark(
            series_clean, model_name, method_fn, phantom_type, radii_ratio)
        return JSONResponse({"results": results})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def run_noise_benchmark(series_clean_path: str, model_name: str, method_fn: Callable, phantom_type: str, radii_ratio: float) -> List[Dict[str, Any]]:
    """
    Orchestrates the entire noise benchmarking process.
    1. Creates noisy series from clean series.
    2. Indexes both clean and noisy series.
    3. Runs the specified decomposition model on both sets.
    4. Cleans up temporary noisy files.
    5. Returns the aggregated results.
    """
    series_clean = Path(series_clean_path)
    series_noisy_root = Path(TEMP_ROOT)
    results = []

    if not series_clean.exists():
        logger.error(f"Clean series path not found: {series_clean_path}")
        return []

    try:
        # Step 1: Create noisy series
        logger.info(f"Generating noisy series in {series_noisy_root}...")
        process_upload(series_clean_path, str(series_noisy_root))

        # Step 2: Index series
        logger.info("Indexing clean and noisy series...")
        clean_idx = index_series_by_kvp(series_clean)
        noisy_idx = index_series_by_kvp(series_noisy_root, noisy=True)

        if not noisy_idx:
            logger.warning("❌ No noisy series found after generation.")
            return []

        # Step 3: Run the tests
        for clean_key in clean_idx:
            prefix, thickness = clean_key
            logger.info(
                f"\n=== Starting TEST: phantom={prefix}, thickness={thickness}, radii={radii_ratio} ===")

            for noisy_key in sorted(noisy_idx.keys(), key=lambda x: x[2]):
                noisy_prefix, noisy_thickness, noise_level = noisy_key

                if (noisy_prefix != prefix or noisy_thickness != thickness):
                    continue

                logger.info(
                    f"\n🔊 Noise level: {noise_level} | Clean: {clean_key} | Noisy: {noisy_key}")

                for kvp_low, kvp_high in KVP_PAIRS:
                    clean_low_dir = clean_idx[clean_key].get(kvp_low)
                    clean_high_dir = clean_idx[clean_key].get(kvp_high)
                    noisy_low_dir = noisy_idx[noisy_key].get(kvp_low)
                    noisy_high_dir = noisy_idx[noisy_key].get(kvp_high)

                    if not all([clean_low_dir, clean_high_dir, noisy_low_dir, noisy_high_dir]):
                        logger.info(
                            f"Skipping {kvp_low}/{kvp_high} due to missing directories")
                        continue

                    clean_low_files = get_sorted_dicoms(clean_low_dir)
                    clean_high_files = get_sorted_dicoms(clean_high_dir)
                    noisy_low_files = get_sorted_dicoms(noisy_low_dir)
                    noisy_high_files = get_sorted_dicoms(noisy_high_dir)

                    pair_count = min(len(clean_low_files), len(clean_high_files),
                                     len(noisy_low_files), len(
                                         noisy_high_files),
                                     MAX_SLICES)

                    if pair_count == 0:
                        logger.info(
                            f"  No matching file count in {kvp_low}/{kvp_high}")
                        continue

                    for i in range(pair_count):
                        method_results = run_methods(
                            i,
                            clean_high_files[i],
                            clean_low_files[i],
                            noisy_high_files[i],
                            noisy_low_files[i],
                            model_name,
                            method_fn,
                            prefix,
                            thickness,
                            kvp_low,
                            kvp_high,
                            noise_level,
                            radii_ratio,
                            phantom_type
                        )
                        if method_results:
                            results.extend(method_results)

    finally:
        # Step 4: Clean up temporary files
        if series_noisy_root.exists():
            logger.info(
                f"Cleaning up temporary noise directory: {series_noisy_root}")
            shutil.rmtree(series_noisy_root)
        else:
            logger.warning(
                f"Temporary directory not found for cleanup: {series_noisy_root}")

    return results

# HELPERS FOR NOISE BENCHMARKING
def degrade_image(file: str | Path, out_dir: str | Path, var: float, original_dicom_data: FileDataset):
    file = Path(file)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dicom_data = original_dicom_data
    image = dicom_data.pixel_array.astype(np.float32)

    # Normalization and resizing
    image_normalized = (image - np.min(image)) / \
        (np.max(image) - np.min(image))
    image_resized = cv2.resize(
        image_normalized, (512, 512), interpolation=cv2.INTER_AREA)

    # Reshape for noise application
    x, y = image_resized.shape
    image_flat = image_resized.flatten()
    image_input = np.expand_dims(image_flat, axis=0)

    # Generate and apply noise
    mean = 0
    sigma = np.sqrt(var)
    n = np.random.normal(loc=mean, scale=sigma, size=image_input.shape)
    degraded_image_flat = image_input + n
    degraded_image_flat = np.clip(
        degraded_image_flat, 0.0, 1.0)  # Clip to 0-1 range

    # Reshape back to 2D for saving
    degraded_2d = degraded_image_flat.reshape(x, y)
    # Convert back to 16-bit
    degraded_u16 = (degraded_2d * 65535.0).astype(np.uint16)

    # --- Create new DICOM file ---

    rows, cols = degraded_u16.shape
    pixel_array = degraded_u16

    # Create FileDataset
    file_meta = pydicom.Dataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    out_path_dcm = out_dir / f"{file.stem}.dcm"

    ds = FileDataset(str(out_path_dcm), {},
                     file_meta=file_meta, preamble=b"\0" * 128)

    # Copy relevant DICOM tags from original
    ds.PatientName = dicom_data.get("PatientName", "Test^Patient")
    ds.PatientID = dicom_data.get("PatientID", "123456")
    ds.Modality = "CT"
    ds.StudyInstanceUID = dicom_data.get("StudyInstanceUID", generate_uid())
    ds.SeriesInstanceUID = generate_uid()
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.RescaleSlope = dicom_data.get("RescaleSlope", 1.0)
    ds.RescaleIntercept = dicom_data.get("RescaleIntercept", 0.0)
    ds.KVP = dicom_data.get("KVP", None)  # Important KVP tag

    # Set image-specific data
    ds.Rows = rows
    ds.Columns = cols
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0  # Unsigned
    ds.PixelData = pixel_array.tobytes()

    # Set timestamp
    dt = datetime.now()
    ds.StudyDate = dt.strftime('%Y%m%d')
    ds.StudyTime = dt.strftime('%H%M%S')

    # Save DICOM file
    ds.save_as(out_path_dcm)
    logger.info(f"DICOM saved to {out_path_dcm}")


def process_upload(series_path: str, out_root: str):
    series_path = Path(series_path)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Store KVP values for naming
    kvp_map: Dict[str, pydicom.Dataset] = {}

    for root, _, files in os.walk(series_path):
        root = Path(root)

        dicom_files = [f for f in files if f.lower().endswith(".dcm")]
        if not dicom_files:
            continue

        # Read one DICOM to get KVP and SeriesName
        try:
            sample_dcm_path = root / dicom_files[0]
            dcm = pydicom.dcmread(str(sample_dcm_path))
            kvp = dcm.get("KVP")

            series_name = root.name
            kvp_map[series_name] = dcm
        except Exception as e:
            logger.warning(
                f"Skipping folder {root.name}: Could not read KVP/DICOM metadata: {e}")
            continue

        # Process all DICOMs in the subfolder
        for filename in dicom_files:
            src_path = root / filename
            original_dcm_data = pydicom.dcmread(str(src_path))
            series_name = root.name

            for var in VAR:
                subfolder_name = f"degraded-{series_name}-{var}"
                out_dir = out_root / subfolder_name
                out_dir.mkdir(parents=True, exist_ok=True)

                degrade_image(src_path, out_dir, var, original_dcm_data)


def index_series_by_kvp(root: str | Path, noisy: bool = False) -> Dict[Tuple[Any, ...], Dict[int, Path]]:
    index = {}
    root = Path(root)

    for paths, _, _ in os.walk(root):
        base = Path(paths).name
        m = NOISY_SERIES_RE.match(base) if noisy else SERIES_RE.match(base)

        if not m:
            continue

        if noisy:
            prefix, thickness, kvp_str, noise_str = m.groups()
            kvp = int(kvp_str)
            noise = float(noise_str)

            key: Tuple[Any, ...] = (prefix, thickness, noise)
        else:
            prefix, thickness, kvp_str = m.groups()
            kvp = int(kvp_str)

            key = (prefix, thickness)

        index.setdefault(key, {})
        index[key][kvp] = Path(paths)

    return index


def get_sorted_dicoms(directory: Path) -> List[Path]:
    return sorted([p for p in directory.glob("*.dcm") if p.is_file()])


def run_methods(i: int, clean_high_file: Path, clean_low_file: Path, noisy_high_file: Path, noisy_low_file: Path, method_name: str, method_fn: Callable, prefix: str, thickness: str, kvp_low: int, kvp_high: int, noise_level: float, radii: float, phantom_type: str) -> List[Dict[str, Any]]:
    result = []
    try:
        logger.info(
            f"  → Running {kvp_low}/{kvp_high} pair index {i} with {method_name}")

        clean_result_str = method_fn(
            str(clean_high_file), str(clean_low_file), phantom_type, radii)
        noisy_result_str = method_fn(
            str(noisy_high_file), str(noisy_low_file), phantom_type, radii)

        clean_result = json.loads(clean_result_str)
        noisy_result = json.loads(noisy_result_str)

        result.append({
            "phantom": prefix,
            "thickness": thickness,
            "kvp_pair": (kvp_low, kvp_high),
            "method": method_name,
            "pair_index": i,
            "clean": clean_result,
            "noisy": noisy_result,
            "noise_level": noise_level,
            "radii_ratio": radii
        })
    except Exception as e:
        logger.error(
            f"    [!] Error running {method_name} on pair index {i} ({kvp_low}/{kvp_high}): {e}")
    return result


def _load_dicom_hu(path: str):
    ds = pydicom.dcmread(str(path))
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    inter = float(getattr(ds, "RescaleIntercept", 0.0))
    return arr * slope + inter, ds


def _get_display_window(hu: np.ndarray, ds: Optional[pydicom.dataset.FileDataset]) -> Tuple[float, float]:
    wc, ww = getattr(ds, "WindowCenter", None), getattr(
        ds, "WindowWidth", None)

    def _f(x):
        if hasattr(x, "__len__"):
            return float(x[0])
        try:
            return float(x)
        except Exception:
            return None

    c, w = _f(wc), _f(ww)
    if c and w and w > 1e-6:
        return c - w/2, c + w/2
    q1, q99 = np.nanpercentile(hu, [1, 99])
    if q99 <= q1:
        q1, q99 = np.nanmin(hu), np.nanmax(hu)
    return float(q1), float(q99)


def _apply_window_to_01(hu: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    return np.clip((hu - vmin) / max(vmax - vmin, 1e-6), 0, 1)


def _colorize_inside_masks_single(base_gray01: np.ndarray,
                                  circles: List[Dict],
                                  spr_table: Dict[str, float],
                                  spr_range: Optional[Tuple[float, float]],
                                  cmap_name: str = "viridis",
                                  saturation: float = 0.95,
                                  draw_outline: bool = False,
                                  phantom_material: str = "Background",
                                  phantom_thresh: float = 0.02) -> Tuple[np.ndarray, Tuple[float, float]]:

    H, W = base_gray01.shape
    base_rgb = np.stack([base_gray01]*3, axis=-1)

    used = [float(spr_table.get(c["material"], 1.00)) for c in circles]
    used.append(float(spr_table.get(phantom_material, 1.00)))
    if spr_range is None:
        spr_min, spr_max = float(np.min(used)), float(np.max(used))
        if spr_max <= spr_min:
            spr_max = spr_min + 1e-3
    else:
        spr_min, spr_max = spr_range

    norm = Normalize(vmin=spr_min, vmax=spr_max)
    cmap = plt.get_cmap(cmap_name)

    hsv = cv2.cvtColor((base_rgb*255).astype(np.uint8),
                       cv2.COLOR_RGB2HSV).astype(np.float32)
    yy, xx = np.ogrid[:H, :W]
    circle_union = np.zeros((H, W), dtype=bool)

    for c in circles:
        spr = float(spr_table.get(c["material"], 1.00))
        rgb = np.array(cmap(norm(spr))[:3], dtype=np.float32).reshape(1, 1, 3)
        hsv_color = cv2.cvtColor(
            (rgb*255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        h = hsv_color[0, 0, 0]
        mask = (xx - c["x"])**2 + (yy - c["y"])**2 <= c["r"]**2
        circle_union |= mask
        hsv[mask, 0] = h
        hsv[mask, 1] = saturation*255
        if draw_outline:
            cv2.circle(hsv, (c["x"], c["y"]), c["r"],
                       (h, 0, hsv[mask, 2].mean()), 1)

    phantom_mask = (base_gray01 > phantom_thresh) & (~circle_union)
    phantom_spr = float(spr_table.get(phantom_material, 1.00))
    phantom_rgb = np.array(cmap(norm(phantom_spr))[
                           :3], dtype=np.float32).reshape(1, 1, 3)
    phantom_hsv = cv2.cvtColor(
        (phantom_rgb*255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    phantom_h = phantom_hsv[0, 0, 0]
    hsv[phantom_mask, 0] = phantom_h
    hsv[phantom_mask, 1] = saturation*255

    colored = cv2.cvtColor(hsv.astype(np.uint8),
                           cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
    return colored, (spr_min, spr_max)


def _save_rgb_png(rgb01: np.ndarray, save_path: str):
    plt.imsave(save_path, np.clip(rgb01, 0, 1))

@app.post("/make-spr-map")
async def make_spr_map(request: Request):
    data = await request.json()

    phantom = data.get("phantom")
    which = data.get("which", "high").lower()
    image_url = data.get("image_url")
    spr_values = data.get("spr_values", {})
    spr_range = data.get("spr_range")
    cmap_name = data.get("cmap", "viridis")
    saturation = float(data.get("saturation", 0.95))
    draw_outline = bool(data.get("draw_outline", False))
    phantom_material = data.get("phantom_material", "Background")

    if phantom not in CIRCLE_DATA:
        raise HTTPException(
            status_code=400, detail=f"Invalid phantom type. Choose from: {list(CIRCLE_DATA.keys())}")
    if not image_url:
        raise HTTPException(status_code=400, detail="Missing image_url.")
    if which not in ("high", "low"):
        raise HTTPException(
            status_code=400, detail='Parameter "which" must be "high" or "low".')

    spr_table = {}
    spr_table.update({})
    for k, v in spr_values.items():
        try:
            spr_table[str(k)] = float(v)
        except Exception:
            spr_table[str(k)] = 1.00

    if phantom_material not in spr_table:
        spr_table[phantom_material] = float(
            spr_values.get(phantom_material, 1.00))

    dicom_path = convert_to_dicom_path(image_url, is_high=(which == "high"))

    try:
        hu, ds = _load_dicom_hu(dicom_path)
        vmin, vmax = _get_display_window(hu, ds)
        base01 = _apply_window_to_01(hu, vmin, vmax)

        circles = CIRCLE_DATA[phantom]
        circles_norm = [{"x": int(c["x"]), "y": int(c["y"]), "r": int(
            c["radius"]), "material": str(c["material"])} for c in circles]

        if spr_range is not None:
            try:
                spr_range = (float(spr_range[0]), float(spr_range[1]))
            except Exception:
                spr_range = None

        overlay_rgb, used_range = _colorize_inside_masks_single(
            base01, circles_norm, spr_table, spr_range,
            cmap_name=cmap_name, saturation=saturation,
            draw_outline=draw_outline, phantom_material=phantom_material
        )

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_name = f"sprmap_{phantom}_{which}_{stamp}.png"
        out_path = os.path.join(IMAGES_DIR, out_name)
        _save_rgb_png(overlay_rgb, out_path)

        return JSONResponse({
            "spr_map": f"/processed_images/{out_name}",
            "spr_minmax": {"min": used_range[0], "max": used_range[1]},
            "cmap": cmap_name
        })

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("SPR map generation failed")
        raise HTTPException(
            status_code=500, detail=f"SPR map generation failed: {e}")


@app.post("/make-spr-map-dicom")
async def make_spr_map_dicom(request: Request):
    data = await request.json()

    phantom = data.get("phantom")
    which = data.get("which", "high").lower()
    image_url = data.get("image_url")
    spr_values = data.get("spr_values", {})
    spr_range = data.get("spr_range")
    cmap_name = data.get("cmap", "viridis")
    saturation = float(data.get("saturation", 0.95))
    draw_outline = bool(data.get("draw_outline", False))
    phantom_material = data.get("phantom_material", "Background")

    if phantom not in CIRCLE_DATA:
        raise HTTPException(
            status_code=400, detail=f"Invalid phantom type. Choose from: {list(CIRCLE_DATA.keys())}")
    if not image_url:
        raise HTTPException(status_code=400, detail="Missing image_url.")
    if which not in ("high", "low"):
        raise HTTPException(
            status_code=400, detail='Parameter "which" must be "high" or "low".')

    spr_table = {}
    for k, v in spr_values.items():
        try:
            spr_table[str(k)] = float(v)
        except Exception:
            spr_table[str(k)] = 1.00

    if phantom_material not in spr_table:
        spr_table[phantom_material] = float(
            spr_values.get(phantom_material, 1.00))

    dicom_path = convert_to_dicom_path(image_url, is_high=(which == "high"))

    try:
        hu, ds = _load_dicom_hu(dicom_path)
        vmin, vmax = _get_display_window(hu, ds)
        base01 = _apply_window_to_01(hu, vmin, vmax)

        circles = CIRCLE_DATA[phantom]
        circles_norm = [{"x": int(c["x"]), "y": int(c["y"]), "r": int(
            c["radius"]), "material": str(c["material"])} for c in circles]

        if spr_range is not None:
            try:
                spr_range = (float(spr_range[0]), float(spr_range[1]))
            except Exception:
                spr_range = None

        overlay_rgb, used_range = _colorize_inside_masks_single(
            base01, circles_norm, spr_table, spr_range,
            cmap_name=cmap_name, saturation=saturation,
            draw_outline=draw_outline, phantom_material=phantom_material
        )

        overlay_rgb_uint8 = (overlay_rgb * 255).astype(np.uint8)

        gray_image = np.dot(overlay_rgb_uint8[..., :3], [
                            0.299, 0.587, 0.114]).astype(np.uint8)

        new_ds = ds.clone() if hasattr(ds, "clone") else ds.copy()
        new_ds.file_meta = ds.file_meta.copy()
        new_ds.SOPInstanceUID = generate_uid()
        new_ds.SeriesInstanceUID = generate_uid()
        new_ds.SOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
        new_ds.Modality = "OT"
        new_ds.SeriesDescription = "SPR Map"
        new_ds.Rows, new_ds.Columns = gray_image.shape
        new_ds.PhotometricInterpretation = "MONOCHROME2"
        new_ds.SamplesPerPixel = 1
        new_ds.BitsAllocated = 8
        new_ds.BitsStored = 8
        new_ds.HighBit = 7
        new_ds.PixelRepresentation = 0
        new_ds.PixelData = gray_image.tobytes()
        new_ds.is_little_endian = True
        new_ds.is_implicit_VR = False

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_name = f"sprmap_{phantom}_{which}_{stamp}.dcm"
        out_path = os.path.join(IMAGES_DIR, out_name)
        new_ds.save_as(out_path)

        return JSONResponse({
            "spr_map": f"/processed_images/{out_name}",
            "spr_minmax": {"min": used_range[0], "max": used_range[1]},
            "cmap": cmap_name
        })

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("SPR map generation failed")
        raise HTTPException(
            status_code=500, detail=f"SPR map generation failed: {e}")
