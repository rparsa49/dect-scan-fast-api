import cv2
import numpy as np
import pydicom
from methods.saito import saito
from methods.hunemohr import hunemohr
from methods.tanaka import tanaka
from methods.schneider import schneider
from pathlib import Path
import os
import re

# Constants
DATA_LOCO = Path("test_images")
KVP_PAIRS = [(70, 100), (70, 120), (70, 140), (80, 100), (80, 120), (80, 140)]
SERIES_RE = re.compile(r'^(?:degraded-)?(.+)-(\d+(?:\.\d+)?)-(\d+)$', re.IGNORECASE)

# Process uploaded folder of series
def process_upload(series_path, out_root = "test_images"):

    series_path = Path(series_path)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for root, subdirs, files in os.walk(series_path):
        root = Path(root)

        # Skip writing anything for the top if it directly contains files you don't intend to process.
        for filename in files:
            if filename.startswith('.') or filename == '.DS_Store':
                continue

            src_path = root / filename

            # Build output directory under the fixed local folder "test_images"
            subfolder_name = root.name 
            out_dir = out_root / f"degraded-{subfolder_name}"
            out_dir.mkdir(parents=True, exist_ok=True)

            degrade_image(src_path, out_dir)

def degrade_image(file: str | Path, out_dir: str | Path):
    """
    Read a DICOM, add Gaussian noise, and write PNG to:
      out_dir / (stem + ".png")
    """
    file = Path(file)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dicom_data = pydicom.dcmread(str(file))
    image = dicom_data.pixel_array.astype(np.float32)
    
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    image = cv2.resize(image, (512, 512))
    image = image.flatten()
    image = np.expand_dims(image, axis=0)
    
    x, y = image.shape
    mean = 0
    var = 0.01
    sigma = np.sqrt(var)
    n = np.random.normal(loc = mean, scale = sigma, size = (x, y))
    degraded_image = image + n

    out_path = out_dir / f"{file.stem}.png"
    degraded_2d = degraded_image.reshape(512, 512)
    degraded_u8 = (np.clip(degraded_2d, 0.0, 1.0) * 255.0).astype(np.uint8)

    ok = cv2.imwrite(str(out_path), degraded_u8)
    if not ok:
        raise IOError(f"Failed to write image to {out_path}")

    return degraded_image

def index_series_by_kvp(root):
    '''
    Walks root directory and returns {(prefix, thickness) : {kvp: pathToSeries}}
    '''
    index = {}
    
    for paths, subdirs, files in os.walk(root):
        base = Path(paths).name
        m = SERIES_RE.match(base)
        if not m:
            continue
        prefix, thickness, kvp = m.groups()
        kvp = int(kvp)
        
        key = (prefix, thickness)
        index.setdefault(key, {})
        index[key][kvp] = Path(paths)
    return index

# Run test on selected method and series
def test(series_clean, series_noisy, phantom_type, radii):
    # Index series
    clean_idx = index_series_by_kvp(series_clean)
    noisy_idx = index_series_by_kvp(series_noisy)

    # Try each kVp pair
    common_keys = sorted(set(clean_idx.keys()) | set(noisy_idx.keys()))
    

if __name__ == "__main__":

    # process_upload("/Users/royaparsa/Desktop/Body-0.6/")
    
    series_clean = "/Users/royaparsa/Desktop/Body-0.6/"
    series_noisy = "/Users/royaparsa/NYPC-DCT-BE/test_images"

    phantom_type = "Body"
    radii = 100

    print("Indexing series...")
    clean_idx = index_series_by_kvp(series_clean)
    noisy_idx = index_series_by_kvp(series_noisy)

    print("CLEAN keys found:", list(clean_idx.keys()))
    print("NOISY keys found:", list(noisy_idx.keys()))
    print()

    # Now run the test harness
    test(series_clean, series_noisy, phantom_type, radii)
