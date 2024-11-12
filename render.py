import os
import numpy as np
import pydicom
from scipy.ndimage import zoom
import matplotlib.pyplot as plt


def load_scan(path):
    slices = [pydicom.dcmread(os.path.join(path, f))
              for f in os.listdir(path) if f.endswith('.dcm')]
    slices.sort(key=lambda x: int(x.InstanceNumber))  # Sort by InstanceNumber
    return slices


def get_pixels_hu(scans):
    image = np.stack([s.pixel_array for s in scans])
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    for i in range(len(scans)):
        intercept = scans[i].RescaleIntercept
        slope = scans[i].RescaleSlope
        if slope != 1:
            image[i] = slope * image[i].astype(np.float64)
            image[i] = image[i].astype(np.int16)
        image[i] += np.int16(intercept)

    return np.array(image, dtype=np.int16)


def reslice_image(volume, spacing, new_spacing=[1, 1, 1]):
    resize_factor = np.array(spacing) / np.array(new_spacing)
    new_shape = np.round(volume.shape * resize_factor)
    new_shape = new_shape.astype(int)

    resliced_volume = zoom(volume, resize_factor, mode='nearest')
    return resliced_volume


# Specify the folder containing the DICOM series
folder_path = "/Users/royaparsa/Downloads/Data/20240513/1.3.12.2.1107.5.1.4.83775.30000024051312040257200010628/"

# Load the DICOM files and convert to Hounsfield units
scans = load_scan(folder_path)
volume = get_pixels_hu(scans)

# Get the pixel spacing from the DICOM metadata
spacing = (float(scans[0].SliceThickness), float(
    scans[0].PixelSpacing[0]), float(scans[0].PixelSpacing[1]))

# Reslice the image to have isotropic spacing
resliced_volume = reslice_image(volume, spacing)

# Visualize the axial, coronal, and sagittal slices


def show_slices(volume):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(volume[volume.shape[0] // 2], cmap="gray")  # Axial view
    axes[0].set_title("Axial Slice")
    axes[1].imshow(volume[:, volume.shape[1] // 2, :],
                   cmap="gray")  # Coronal view
    axes[1].set_title("Coronal Slice")
    axes[2].imshow(volume[:, :, volume.shape[2] // 2],
                   cmap="gray")  # Sagittal view
    axes[2].set_title("Sagittal Slice")
    plt.show()


# Display the 3D slices
show_slices(resliced_volume)
