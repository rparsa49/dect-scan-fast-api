import os
from typing import Tuple, List
from PIL import Image
from pydicom import dcmread
import pydicom
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from pydicom.pixel_data_handlers.util import apply_modality_lut

# Function to save circles data to a file
# Arguments:
# - circles: Detected circles data
# - file_path: Path to the file where circles data will be saved
def save_circles_data(circles, file_path):
    circles_list = circles[0].tolist()
    with open(file_path, 'w') as f:
        json.dump(circles_list, f)

# Function to load circles data from a file
# Arguments:
# - file_path: Path to the file where circles data is stored
# Returns:
# - Numpy array of circles data
def load_circles_data(file_path):
    with open(file_path, 'r') as f:
        circles_list = json.load(f)
    return np.array([circles_list], dtype=np.uint16)

# Function to filter out unwanted circles based on coordinates
# Arguments:
# - circles: Numpy array of detected circles
# Returns:
# - Numpy array of filtered circles
def filter_circles(circles):
    filtered_circles = []
    for circle in circles[0]:
        x, y, radius = circle
        if y < 359:
            filtered_circles.append(circle)
    return np.array([filtered_circles], dtype=np.uint16)

# Function to calculate the mean pixel value for a circle
# Arguments:
# - image: Image data
# - circle: Circle parameters (x, y, radius)
# Returns:
# - Mean pixel value within the circle
def calculate_mean_pixel_value(image, circle):
    x, y, radius = circle
    mask = np.zeros(image.shape, dtype=np.uint8)
    cv2.circle(mask, (x, y), radius, 1, thickness=-1)
    pixel_values = image[mask == 1]
    mean_value = np.mean(pixel_values)
    return mean_value

# Function to load a DICOM image
# Arguments:
# - dicom_path: Path to the DICOM file
# Returns:
# - Image data and DICOM metadata
def load_dicom_image(dicom_path):
    dicom_data = pydicom.dcmread(dicom_path)
    image = dicom_data.pixel_array
    return image, dicom_data

# Function to process the image and save detected circles data
# Arguments:
# - image: Image data
# - circles_data_path: Path to save the circles data
def process_and_save_circles(image, circles_data_path):
    image_8bit = cv2.normalize(
        image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    blurred_image = cv2.GaussianBlur(image_8bit, (9, 9), 0)
    circles = cv2.HoughCircles(blurred_image, cv2.HOUGH_GRADIENT, 1,
                               minDist=20, param1=50, param2=30, minRadius=0, maxRadius=50)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        circles = filter_circles(circles)
        save_circles_data(circles, circles_data_path)

# Function to draw circles on an image and calculate HU values using modality LUT
# Arguments:
# - image: Image data
# - saved_circles: Numpy array of saved circles data
# - radii_ratios: List of radii ratios for smaller circles
# - dicom_data: DICOM dataset to apply the modality LUT
# Returns:
# - Image with drawn circles and list of mean HU values
def draw_and_calculate_circles(image, saved_circles, radii_ratios, dicom_data):
    image_8bit = cv2.normalize(
        image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    contour_image = cv2.cvtColor(image_8bit, cv2.COLOR_GRAY2BGR)
    mean_hu_values = []

    # Apply modality LUT to convert pixel values to Hounsfield Units (HU)
    hu_image = apply_modality_lut(image, dicom_data)

    for circle in saved_circles[0, :]:
        x, y, radius = circle
        # Draw the main circle's outline only (no fill)
        cv2.circle(contour_image, (x, y), radius,
                   (255, 0, 0), 2)  # Red outline

        for ratio in radii_ratios:
            new_radius = int(radius * ratio)
            mean_hu_value = calculate_mean_pixel_value(
                hu_image, (x, y, new_radius))
            mean_hu_values.append(mean_hu_value)
            # Draw the inner circles' outline only (no fill)
            cv2.circle(contour_image, (x, y), new_radius,
                       (0, 255, 255), 2)  # Yellow outline

    return contour_image, mean_hu_values

# Function to display an image using matplotlib with a grayscale colormap
# Arguments:
# - image: Image data to be displayed
# - title: Title for the displayed image
def display_image(image, title='Image'):
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.colorbar(label="Hounsfield Units (HU)")
    plt.axis('off')
    plt.show()

# Function to categorize material based on HU value
# Arguments: 
# - hu_value: List of Houndsfield Units
# Source: F. Fartin 2020 (https://prod-images-static.radiopaedia.org/images/52608436/ffb5a7e3ebb12255dec689e924ddbd_big_gallery.jpeg)
def categorize_material(hu_value):
    if hu_value <= -1000:
        return "Air"
    elif -50 < hu_value <= 20:
        return "Simple fluid"
    elif 20 < hu_value <= 45:
        return "Soft tissue"
    elif 45 < hu_value <= 90:
        return "Acute blood"
    elif 100 <= hu_value <= 500:
        return "Iodinated contrast"
    elif 300 <= hu_value <= 800:
        return "Trabecular bone"
    elif hu_value > 1000:
        return "Cortical bone"
    else:
        return "Unknown"


# Process each circle's and sub-circle's HU values to determine the material
# Arguments:
# - hu_values: List of Houndsfield Units to be categorized
def determine_materials(hu_values):
    materials = []
    for hu in hu_values:
        material = categorize_material(hu)
        materials.append(material)
    return materials


# Define known rho_e and z_eff values for each material with additional parameters
material_properties = {
    "Air": {
        "rho_e": 0.001,
        "z_eff": 7.6,
        "w_m": [1.0],  # Mostly nitrogen
        "Z": [7],  # Nitrogen
        "A": [14.007],  # Atomic weight of nitrogen
        "I": [82.0]  # Mean excitation potential for nitrogen
    },
    "Fat": {
        "rho_e": 0.91,
        "z_eff": 5.92,
        "w_m": [0.6, 0.4],  # Carbon and Hydrogen in fat
        "Z": [6, 1],  # Carbon and Hydrogen
        "A": [12.01, 1.0079],  # Atomic weights of Carbon and Hydrogen
        "I": [78.0, 19.2]  # Mean excitation potential for Carbon and Hydrogen
    },
    "Simple fluid": {
        "rho_e": 1.0,
        "z_eff": 7.42,
        "w_m": [0.111894, 0.888106],  # Hydrogen and Oxygen in water
        "Z": [1, 8],  # Hydrogen and Oxygen
        "A": [1.0079, 15.999],  # Atomic weights of Hydrogen and Oxygen
        "I": [19.2, 95.0]  # Mean excitation potential for Hydrogen and Oxygen
    },
    "Soft tissue": {
        "rho_e": 1.0,
        "z_eff": 7.42,
        "w_m": [0.1, 0.2, 0.7],  # Hydrogen, Carbon, Oxygen
        "Z": [1, 6, 8],  # Hydrogen, Carbon, Oxygen
        "A": [1.0079, 12.01, 15.999],  # Atomic weights
        "I": [19.2, 78.0, 95.0]  # Mean excitation potentials
    },
    "Acute blood": {
        "rho_e": 1.05,
        "z_eff": 7.51,
        "w_m": [0.1, 0.8, 0.1],  # Hydrogen, Oxygen, Carbon
        "Z": [1, 8, 6],  # Hydrogen, Oxygen, Carbon
        "A": [1.0079, 15.999, 12.01],  # Atomic weights
        "I": [19.2, 95.0, 78.0]  # Mean excitation potentials
    },
    "Iodinated contrast": {
        "rho_e": 1.9,
        "z_eff": 53.0,
        "w_m": [0.2, 0.8],  # Iodine and water
        "Z": [53, 8],  # Iodine and Oxygen
        "A": [126.904, 15.999],  # Atomic weights
        "I": [484.0, 95.0]  # Mean excitation potentials
    },
    "Trabecular bone": {
        "rho_e": 1.1,
        "z_eff": 12.31,
        "w_m": [0.4, 0.6],  # Calcium and Phosphorus
        "Z": [20, 15],  # Calcium and Phosphorus
        "A": [40.08, 30.974],  # Atomic weights
        "I": [322.0, 214.0]  # Mean excitation potentials
    },
    "Cortical bone": {
        "rho_e": 1.85,
        "z_eff": 13.8,
        "w_m": [0.4, 0.6],  # Calcium and Phosphorus
        "Z": [20, 15],  # Calcium and Phosphorus
        "A": [40.08, 30.974],  # Atomic weights
        "I": [322.0, 214.0]  # Mean excitation potentials
    },
    "Unknown": {
        "rho_e": None,
        "z_eff": None,
        "w_m": None,
        "Z": None,
        "A": None,
        "I": None
    }
}

# Function to get true rho_e and z_eff for a material
def get_true_values(material):
    return material_properties.get(material, {"rho_e": None, "z_eff": None})

# Function to calculate RMSE manually using numpy


def calculate_rmse(true_values, predicted_values):
    return np.sqrt(np.mean((np.array(true_values) - np.array(predicted_values))**2))


def process_dicom_files(dicom_file_paths: List[str]) -> Tuple[Image.Image, Image.Image, float]:
    """
    Processes DICOM files to classify them as high or low kVp and extract images.
    
    Parameters:
        dicom_file_paths (List[str]): Paths to the DICOM files.
        
    Returns:
        Tuple[Image.Image, Image.Image, float]: High kVp image, Low kVp image, slice thickness.
    """
    high_kvp_files = []
    low_kvp_files = []
    slice_thickness = None

    # Classify files into high or low kVp based on the first file in each folder
    for file_path in dicom_file_paths:
        # Read the DICOM file's metadata
        dicom_file = dcmread(file_path)

        # Get kVp value and slice thickness
        kvp = dicom_file.get("KVP")
        # Get slice thickness once (same for all slices in the study)
        if slice_thickness is None:
            slice_thickness = dicom_file.get("SliceThickness")

        # Classify based on kVp value (assuming high is > 80 kVp)
        if kvp > 80:
            high_kvp_files.append(file_path)
        else:
            low_kvp_files.append(file_path)

    # Convert the first DICOM file in each category to an image
    high_kvp_image = dicom_to_image(
        high_kvp_files[0]) if high_kvp_files else None
    low_kvp_image = dicom_to_image(low_kvp_files[0]) if low_kvp_files else None

    return high_kvp_image, low_kvp_image, slice_thickness


def dicom_to_image(dicom_path: str) -> Image.Image:
    """
    Converts a DICOM file to a PIL Image.

    Parameters:
        dicom_path (str): Path to the DICOM file.

    Returns:
        Image.Image: The converted image.
    """
    dicom_file = dcmread(dicom_path)
    pixel_array = dicom_file.pixel_array
    image = Image.fromarray(pixel_array).convert("L")
    return image
