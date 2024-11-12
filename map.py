import pydicom
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pydicom.pixel_data_handlers.util import apply_modality_lut
from dect_processing import categorize_material, load_circles_data, load_dicom_image, process_and_save_circles, draw_and_calculate_circles

# Define color mapping for each material
material_colors = {
    "Air": (0, 0, 255),         # Blue
    "Fat": (0, 255, 255),       # Yellow
    "Simple fluid": (255, 0, 0),  # Red
    "Soft tissue": (0, 255, 0),  # Green
    "Acute blood": (255, 165, 0),  # Orange
    "Iodinated contrast": (128, 0, 128),  # Purple
    "Trabecular bone": (139, 69, 19),   # Brown
    "Cortical bone": (255, 255, 255),   # White
    "Unknown": (128, 128, 128)  # Grey
}

# Function to draw color-coded circles for each material
def generate_material_map(image, circles, mean_hu_values):
    # Create a blank RGB image for overlay
    color_map = np.zeros((*image.shape, 3), dtype=np.uint8)

    for i, circle in enumerate(circles[0]):
        x, y, radius = circle
        material = categorize_material(mean_hu_values[i])
        # Default to grey for unknown
        color = material_colors.get(material, (128, 128, 128))

        # Draw filled circle with material color
        cv2.circle(color_map, (x, y), radius, color, -1)

    return color_map

# Function to process a DICOM file and generate a material map with legend
def process_dicom_file(dicom_path, circles_data_path, radii_ratios=[1]):
    # Load DICOM image and metadata
    image, dicom_data = load_dicom_image(dicom_path)

    # Process and detect circles
    process_and_save_circles(image, circles_data_path)
    circles = load_circles_data(circles_data_path)

    # Draw and calculate HU values for circles
    contour_image, mean_hu_values = draw_and_calculate_circles(
        image, circles, radii_ratios, dicom_data)

    # Generate color-coded material map
    material_map = generate_material_map(image, circles, mean_hu_values)

    # Overlay the material map on the grayscale image
    overlayed_image = cv2.addWeighted(cv2.cvtColor(
        contour_image, cv2.COLOR_BGR2RGB), 0.5, material_map, 0.5, 0)

    # Create legend patches
    legend_patches = [Patch(color=np.array(color)/255.0, label=material)
                      for material, color in material_colors.items()]

    # Display the result with a legend
    plt.imshow(overlayed_image)
    plt.title('Material Map')
    plt.axis('off')
    plt.legend(handles=legend_patches, bbox_to_anchor=(
        1.05, 1), loc='upper left', borderaxespad=0.)
    plt.show()

    return overlayed_image


# Example usage:
# DICOM file path and circles data file path
dicom_file_path = 'img.dcm'
circles_data_path = 'circles_data.json'

# Generate and display material map with legend
overlayed_material_map = process_dicom_file(dicom_file_path, circles_data_path)
