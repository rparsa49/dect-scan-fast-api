import json
import pydicom
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import logging

DATA_DIR = Path("data")
def load_json(file_name):
    with open(DATA_DIR / file_name, "r") as file:
        return json.load(file)

SUPPORTED_MODELS = load_json("supported_models.json")
HU_CATEGORIES = load_json("hu_categories.json")
ELEMENT_PROPERTIES = load_json("element_properties.json")
MATERIAL_PROPERTIES = load_json("material_properties.json")
IMAGES_DIR = "processed_images"

# Load circle locations from circle.json
with open("data/circles.json") as f:
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
        if float(min_hu) <= float(hu_value) <= float(max_hu):
            return material
    # return "unknown"

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
        logging.info(
            f"Circle data \n X: {circle['x']} \n Y: {circle['y']}\n Radius: {circle['radius']}")
        new_radius = int(radius * radii_ratios)
        logging.info(f"New radius: {new_radius}")
        mean_hu_value = calculate_mean_pixel_value(
            hu_image, (x, y, new_radius))
        mean_hu_values.append(mean_hu_value)

    return mean_hu_values

# Find DICOM files from uploaded directry
# def find_first_dicom_file():
#     processed_images_dir = Path(IMAGES_DIR)
#     for user_folder in processed_images_dir.iterdir():
#         if user_folder.is_dir():
#             for subfolder in user_folder.iterdir():
#                 if subfolder.is_dir():
#                     dicom_files = sorted(subfolder.glob("*.dcm"))
#                     if dicom_files:
#                         # Return the first DICOM file found
#                         return str(dicom_files[0])
#     return None

# Find high and low DICOM files from uploaded directory
def find_dicom_files():
    processed_images_dir = Path(IMAGES_DIR)
    dicom_files = []

    # Traverse all DICOM files
    for user_folder in processed_images_dir.iterdir():
        if user_folder.is_dir():
            for subfolder in user_folder.iterdir():
                if subfolder.is_dir():
                    dicom_files.extend(sorted(subfolder.glob("*.dcm")))

    # Load first image metadata to determine two energy levels
    first_dicom = pydicom.dcmread(str(dicom_files[0]))
    base_kVp = first_dicom.KVP  # Reference kVp value

    high_kvp_file, low_kvp_file = None, None

    for dicom_path in dicom_files:
        dicom_data = pydicom.dcmread(str(dicom_path))
        if dicom_data.KVP > base_kVp:
            high_kvp_file = dicom_path
        elif dicom_data.KVP < base_kVp:
            low_kvp_file = dicom_path
        else:
            # Assign first file as one of the energy levels if others are missing
            if not high_kvp_file:
                high_kvp_file = dicom_path
            elif not low_kvp_file:
                low_kvp_file = dicom_path

    return str(high_kvp_file), str(low_kvp_file)

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
