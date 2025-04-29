import pydicom, json
from pathlib import Path
import numpy as np

from collections import defaultdict
# Iterates over all dicom files and collects their KVP and slice thickness
# For each labeled insert, records the HU value at that location
# Make a range of the averages from high to low

# Load circle data
DATA_DIR = Path("data")
# dicom location
DICOMS = Path("/Users/royaparsa/Downloads/Data/DICOM-DATA/")

def load_json(file_name):
    with open(DATA_DIR / file_name, "r") as file:
        return json.load(file)

CIRCLE_DATA = load_json("circles.json")
CIRCLES = CIRCLE_DATA["body"]

# Get avg pixel value inside the inserts
def get_circle_mean(dcm, x, y, radius):
    pixels = dcm.pixel_array
    circle_values = []
    
    for i in range(pixels.shape[0]):
        for j in range(pixels.shape[1]):
            if (i - y) ** 2 + (j - x) ** 2 <= radius ** 2:
                circle_values.append(pixels[i, j])
                
    if circle_values:
        return np.mean(circle_values)
    else:
        return None
    
# Data
data = defaultdict(lambda: defaultdict(list))

# Iterate over each dicom series
for scan_dir in DICOMS.iterdir():
    if not scan_dir.is_dir():
        continue
    
    kvp_values = set()
    slice_thickness = None
    
    # Track avg. pixel values for each material across slices in this series
    material_pixel_values = defaultdict(list)
    
    for dicom_file in scan_dir.glob("*.dcm"):
        dcm = pydicom.dcmread(dicom_file)
        
        kvp = float(dcm.KVP)
        slice_thickness = float(dcm.SliceThickness)
        
        kvp_values.add(kvp)
        
        # For each insert, calculate mean pixel value and store it
        for circle in CIRCLES:
            material = circle["material"]
            avg_pixel = get_circle_mean(dcm, circle["x"], circle["y"], circle["radius"])
            if avg_pixel is not None:
                material_pixel_values[material].append(avg_pixel)
    
    # Store avg. pixel values for each material across slices in this series
    for material, values in material_pixel_values.items():
        avg_pixel_value = np.mean(values)
        
        for kvp in kvp_values:
            data[slice_thickness][material].append((kvp, avg_pixel_value))

# Process high/low pairs and calc. ranges per material and slice thickness
ranges = defaultdict(lambda: defaultdict(tuple))

for slice_thickness, materials in data.items():
    for material, kvp_values in materials.items():
        kvp_groups = defaultdict(list)
        
        for kvp, avg_pixel_value in kvp_values:
            if kvp < 100:
                kvp_groups["low"].append(avg_pixel_value)
            else:
                kvp_groups["high"].append(avg_pixel_value)
        if kvp_groups["low"] and kvp_groups["high"]:
            avg_low = np.mean(kvp_groups["low"])
            avg_high = np.mean(kvp_groups["high"])
            
            ranges[slice_thickness][material] = (avg_low, avg_high)

# Output results
print("\nAverage HU Ranges (Low/High) per Material and Slice Thickness:\n")
for slice_thickness, materials in ranges.items():
    print(f"Slice Thickness: {slice_thickness:.2f} mm")
    for material, (low, high) in materials.items():
        print(f"  {material}: Low={low:.2f}, High={high:.2f}")
    print()