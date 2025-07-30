import cv2
import numpy as np
import pydicom
import matplotlib.pyplot as plt
from methods.saito import saito
from methods.hunemohr import hunemohr
from methods.tanaka import tanaka
from methods.schneider import schneider
from pathlib import Path

# Constants
DATA_LOCO = Path("test_images")

# Take in clean image and apply gaussian noise
def degrade_image(file):
    dicom_data = pydicom.dcmread(file)
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
    
    out_filename = f"{Path(file).stem}.png"
    out_path = DATA_LOCO / out_filename
    
    degraded_2d = degraded_image.reshape(512, 512)
    degraded_u8 = (np.clip(degraded_2d, 0.0, 1.0) * 255.0).astype(np.uint8)
    cv2.imwrite(str(out_path), degraded_u8)
    
    return degraded_image

# Run test on selected method and series
def test(high, low, method, phantom_type, radii):
    # Create noisy images
    noisy_high, noisy_low = [], []
    for h, l in zip(high, low):
        nh = degrade_image(h)
        nh = nh.save()
        noisy_low.append(degrade_image(l))
    
    # Run tests on selected method
    if method == "Saito":
        clean_res = saito(high, low, phantom_type, radii)
        noisy_res = saito(noisy_high, noisy_low, phantom_type, radii)
        return clean_res, noisy_res
    if method == "Tanaka":
        clean_res = tanaka(high, low, phantom_type, radii)
        noisy_res = tanaka(noisy_high, noisy_low, phantom_type, radii)
        return clean_res, noisy_res
    if method == "Hunemohr":
        clean_res = hunemohr(high, low, phantom_type, radii)
        noisy_res = hunemohr(noisy_high, noisy_low, phantom_type, radii)
        return clean_res, noisy_res
    if method == "Schneider":
        clean_res = schneider(high, phantom_type, radii)
        noisy_res = schneider(high, phantom_type, radii)
        return clean_res, noisy_res
    

degrade_image("/Users/royaparsa/Desktop/test-data/high/CT1.3.12.2.1107.5.1.4.83775.30000024051312040257200019274.dcm")
