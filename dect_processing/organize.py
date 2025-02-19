import json
import os
import pydicom
import numpy as np
from collections import defaultdict
from pathlib import Path
from dect_processing.dect import (
    categorize_hu_value,
    load_dicom_image,
    draw_and_calculate_circles,
)

# Load circle data
DATA_DIR = Path("data")
CIRCLE_DATA = json.load(open(DATA_DIR / "circles.json"))


def categorize_series(data_dir):
    """Organizes DICOM series by protocol, kernel, kVp, and slice thickness."""
    organized_series = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list)))

    for series_folder in os.listdir(data_dir):
        series_path = os.path.join(data_dir, series_folder)
        if os.path.isdir(series_path):
            metadata = {}

            for dicom_file in os.listdir(series_path):
                dicom_path = os.path.join(series_path, dicom_file)
                try:
                    dicom_data = pydicom.dcmread(dicom_path)
                    metadata = {
                        "kVp": dicom_data.KVP,
                        "SliceThickness": dicom_data.SliceThickness,
                        "ProtocolName": dicom_data.ProtocolName,
                        "ConvolutionKernel": dicom_data.ConvolutionKernel,
                    }
                    break  # Only read the first file for metadata
                except Exception as e:
                    print(f"Error reading {dicom_path}: {e}")
                    continue

            if not metadata:
                continue  # Skip if no metadata was found

            key = (
                metadata["SliceThickness"],
                metadata["kVp"],
                metadata["ProtocolName"],
                metadata["ConvolutionKernel"],
            )

            # Organize by Protocol Name, Kernel, kVp, and Slice Thickness
            organized_series[metadata["ProtocolName"]
                             ][metadata["ConvolutionKernel"]][str(key)].append(series_folder)

    return organized_series


def get_series_materials(organized_series, data_dir):
    """Processes the 10th DICOM file per series folder to extract unique materials."""
    for protocol, kernels in organized_series.items():
        for kernel, series_dict in kernels.items():
            for key, series_folders in series_dict.items():
                for series_folder in series_folders:
                    series_path = os.path.join(data_dir, series_folder)
                    dicom_files = sorted(
                        [f for f in os.listdir(series_path) if f.endswith('.dcm')])

                    # Pick the 10th image, or the closest one if fewer than 10 exist
                    dicom_index = min(30, len(dicom_files) - 1)
                    if dicom_index < 0:
                        print(f"No DICOM files found in {series_folder}")
                        continue

                    dicom_path = os.path.join(
                        series_path, dicom_files[dicom_index])

                    try:
                        # Load DICOM image
                        image, dicom_data = load_dicom_image(dicom_path)

                        # Get circle data based on phantom type
                        phantom_type = "head"  # Default, change as needed
                        if phantom_type not in CIRCLE_DATA:
                            print(f"Unknown phantom type: {phantom_type}")
                            continue
                        saved_circles = CIRCLE_DATA[phantom_type]

                        # Extract HU values from predefined circular regions
                        hu_values = draw_and_calculate_circles(
                            image, saved_circles, 1.0, dicom_data)

                        # Identify all unique materials in the scan
                        unique_materials = set(
                            categorize_hu_value(float(hu)) for hu in hu_values if isinstance(hu, (int, float))
                        )

                        print(
                            f"Materials identified in {series_folder} (using image {dicom_index + 1}): {unique_materials}")

                        # **Fix: Convert tuple key to string before storing**
                        organized_series[protocol][kernel][str(key)] = {
                            "SeriesFolders": series_folders,
                            # Store multiple materials
                            "Materials": list(unique_materials)
                        }

                    except Exception as e:
                        print(f"Error processing {dicom_path}: {e}")
                        continue

    return organized_series


# Run categorization and material extraction
# data_dir = "/Users/royaparsa/Downloads/Data/20240513"
# organized = categorize_series(data_dir)
# print("Now categorizing materials...")
# organized_with_materials = get_series_materials(organized, data_dir)

# Print results (now JSON-safe)
# print(json.dumps(organized_with_materials, indent=4, default=str))

def convert_numpy(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert NumPy array to Python list
    elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
        return float(obj)  # Convert NumPy float to Python float
    return obj  # Return as is if not a NumPy type
