# import pydicom
# import cv2
# import json
# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path

# # Load circle data
# DATA_DIR = Path("data")


# def load_json(file_name):
#     with open(DATA_DIR / file_name, "r") as file:
#         return json.load(file)


# CIRCLE_DATA = load_json("circles.json")

# # Load DICOM images
# high_path = '/Users/royaparsa/Downloads/high-low/1.3.12.2.1107.5.1.4.83775.30000024051312040257200015808/test.dcm'
# low_path = '/Users/royaparsa/Downloads/high-low/1.3.12.2.1107.5.1.4.83775.30000024051312040257200019235/test.dcm'

# dicom_datah = pydicom.dcmread(high_path)
# dicom_datal = pydicom.dcmread(low_path)
# high_image = dicom_datah.pixel_array
# low_image = dicom_datal.pixel_array

# saved_circles = CIRCLE_DATA["head"]

# # Create subplots
# fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# # Display first DICOM image with circles
# axes[0].imshow(high_image, cmap="gray")
# axes[0].set_title("DICOM Image High")
# axes[0].axis("off")

# # Display second DICOM image with circles
# axes[1].imshow(low_image, cmap="gray")
# axes[1].set_title("DICOM Image Low")
# axes[1].axis("off")

# # Draw circles on both images
# for circle in saved_circles:
#     x, y, radius = circle["x"], circle["y"], circle["radius"]

#     # Draw on first image
#     circ1 = plt.Circle((x, y), radius, color='red', fill=False)
#     axes[0].add_patch(circ1)

#     # Draw on second image
#     circ2 = plt.Circle((x, y), radius, color='red', fill=False)
#     axes[1].add_patch(circ2)

# # Show images
# plt.show()
