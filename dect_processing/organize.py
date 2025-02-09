import os
import pydicom
from collections import defaultdict
import numpy as np

def categorize_series(data_dir):
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
                    break
                except Exception as e:
                    print(f"Error reading {dicom_path}: {e}")
                    continue

            key = (
                metadata["SliceThickness"],
                metadata["kVp"],
                metadata["ProtocolName"],
                metadata["ConvolutionKernel"],
            )

            # Organize by Protocol Name, Kernel, Pixel Spacing, kVp, and Slice Thickness
            organized_series[metadata["ProtocolName"]
                             ][metadata["ConvolutionKernel"]][key].append(series_folder)

    return organized_series


organized = categorize_series('/Users/royaparsa/Downloads/Data/20240513')
# print(organized)

def convert_numpy(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert NumPy array to Python list
    elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
        return float(obj)  # Convert NumPy float to Python float
    return obj  # Return as is if not a NumPy type
