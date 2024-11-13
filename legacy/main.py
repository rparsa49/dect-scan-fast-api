import matplotlib.pyplot as plt
import numpy as np
import os
from legacy.dect_processing import *
from models import *
import scipy.optimize as sp

# Define the folder containing the DICOM files
dicom_folder_path = "/Users/royaparsa/Downloads/Data/20240513/1.3.12.2.1107.5.1.4.83775.30000024051312040257200015607/"
circles_data_path = 'circles_data.json'

# Define radii ratios
radii_ratios = [0.3, 0.6, 0.9]

# Parameters for the Saito models (placeholders, these should be calibrated)
alpha = 0.05
a = 0.9
b = 1.0
n = 3.1
c = 1.2
d = 0.5

# Parameters for the Hunemohr models (placeholders, these need to be calibrated for energy levels)
ce = 0.5 
de = 0.6
z_eff_w = 7.4

# Parameters for proton energy and other necessary constants
proton_energy_mev = 140  # Example proton energy in MeV
beta2 = beta_proton(proton_energy_mev)  # Calculate beta squared

# Known atomic and mass properties for water
Z_water = [1, 8]  # Atomic numbers for H and O in water
A_water = [1.0079, 15.999]  # Atomic weights for H and O in water
I_water = [19.2, 95.0]  # Ionization energies for H and O in water
w_water = [2 * 1.0079 / 18.01528, 16 / 18.01528]  # Weight fractions for H2O

# Compute ln_i for water once as it's constant
ln_i_w = ln_i_truth(w_water, Z_water, A_water, I_water)

# Accumulate results across all files
all_rho_e_true = []
all_rho_e_saito = []
all_rho_e_hunemohr = []
all_z_eff_true = []
all_z_eff_saito = []
all_z_eff_hunemohr = []
all_spr_saito = []
all_spr_true = []
all_spr_hunemohr = []

# Iterate over all DICOM files in the folder
for filename in os.listdir(dicom_folder_path):
    if filename.endswith(".dcm"):
        dicom_file_path = os.path.join(dicom_folder_path, filename)

        print(f"Processing file: {filename}")

        # Load the DICOM image
        new_image, new_dicom_data = load_dicom_image(dicom_file_path)

        # Load saved circles data (assuming it's the same for all files)
        saved_circles = load_circles_data(circles_data_path)

        # Draw circles and calculate HU values using modality LUT
        contour_image, mean_hu_values = draw_and_calculate_circles(
            new_image, saved_circles, radii_ratios, new_dicom_data)
        
        # Extract hu_low, hu_mid, hu_high for each insert
        hu_low = mean_hu_values[::3]
        hu_mid = mean_hu_values[1::3]
        hu_high = mean_hu_values[2::3]

        # Determine materials
        materials_high = determine_materials(hu_high)
        materials_mid = determine_materials(hu_mid)
        materials_low = determine_materials(hu_low)

        print(materials_low)
        break
        
        # Calculate beta based on the retrieved kVp
        kvp = new_dicom_data.KVP
        beta = beta_proton(kvp)

        # Calculate rho_e and z_eff using the Saito models
        rho_e_calculated = [rho_e_saito(
            hu_low[i], hu_high[i], alpha, a, b) for i in range(len(hu_low))]
        z_eff_calculated = [z_eff_saito(
            hu_low[i], hu_high[i], rho_e_calculated[i], n, beta, c, d) for i in range(len(hu_low))]

        # Calculate rho_e and z_eff using Hunemohr models
        rho_e_calculated_hunemohr = [rho_e_hunemohr(hu_low[i], hu_high[i], ce) for i in range(len(hu_low))]
        z_eff_calculated_hunemohr = [z_eff_hunemohr(hu_low[i], hu_high[i], rho_e_calculated_hunemohr[i], de, n) for i in range(len(hu_low))]

        # Calculate mean excitation potential (ln(I)) for each material
        ln_i_calculated = [ln_i_fit(z_eff_calculated[i])
                           for i in range(len(z_eff_calculated))]
        ln_i_calculated_hunemohr = [ln_i_fit(z_eff_calculated_hunemohr[i]) for i in range(len(z_eff_calculated_hunemohr))]

        # Calculate the stopping power ratio (SPR) for each material relative to water
        spr_calculated = [spr_w_truth(rho_e_calculated[i], ln_i_calculated[i], ln_i_w, beta2)
                          for i in range(len(rho_e_calculated))]
        spr_calculated_hunemohr = [spr_w_truth(rho_e_calculated_hunemohr[i], ln_i_calculated_hunemohr[i], ln_i_w, beta2) for i in range(len(rho_e_calculated_hunemohr))]

        # Get true values for rho_e, z_eff, and calculate the true stopping power
        for i, material in enumerate(materials_high):
            true_values = get_true_values(material)

            # Skip if any required value is None
            if any(val is None for val in [true_values["w_m"], true_values["Z"], true_values["A"], true_values["I"]]):
                print(f"Skipping material '{material}' due to missing data.")
                continue

            # Append true values
            all_rho_e_true.append(true_values["rho_e"])
            all_z_eff_true.append(true_values["z_eff"])
            
            # Append calculated values
            all_rho_e_saito.append(rho_e_calculated[i])
            all_z_eff_saito.append(z_eff_calculated[i])
            all_rho_e_hunemohr.append(rho_e_calculated_hunemohr[i])
            all_z_eff_hunemohr.append(z_eff_calculated_hunemohr[i])
            
            # Calculate the true ln(I) for the material
            ln_i_m = ln_i_truth(
                true_values["w_m"], true_values["Z"], true_values["A"], true_values["I"])

            # Calculate the true stopping power using the Bethe formula
            sp_true = sp_truth(
                true_values["Z"][0] / true_values["A"][0], ln_i_m, beta2)
            all_spr_true.append(sp_true)

            # Append the calculated stopping power ratio
            all_spr_saito.append(spr_calculated[i])
            all_spr_hunemohr.append(spr_calculated_hunemohr[i])

# Define a threshold for filtering out outliers in z_eff
# z_eff_threshold = 100

# # Filter the data based on the threshold for the zoomed-in plot
# filtered_z_eff_true = []
# filtered_z_eff_saito = []
# filtered_z_eff_hunemohr = []

# for true, saito, hunemohr in zip(all_z_eff_true, all_z_eff_saito, all_z_eff_hunemohr):
#     if saito < z_eff_threshold and hunemohr < z_eff_threshold:
#         filtered_z_eff_true.append(true)
#         filtered_z_eff_saito.append(saito)
#         filtered_z_eff_hunemohr.append(hunemohr)

# # Plotting the zoomed-out graph (including outliers)
# fig, axs = plt.subplots(2, 3, figsize=(21, 14))

# # Zoomed-out: Plot for rho_e
# axs[0, 0].scatter(all_rho_e_true, all_rho_e_saito,
#                   color='blue', label='Saito Model')
# axs[0, 0].scatter(all_rho_e_true, all_rho_e_hunemohr,
#                   color='orange', label='Hunemohr Model')
# axs[0, 0].plot(all_rho_e_true, all_rho_e_true,
#                color='red', linestyle='--', label='True')
# axs[0, 0].set_xlabel('True rho_e')
# axs[0, 0].set_ylabel('Calculated rho_e')
# axs[0, 0].set_title('rho_e: True vs Calculated (All Files - Zoomed Out)')
# axs[0, 0].legend()
# axs[0, 0].grid(True)

# # Zoomed-out: Plot for z_eff
# axs[0, 1].scatter(all_z_eff_true, all_z_eff_saito,
#                   color='blue', label='Saito Model')
# axs[0, 1].scatter(all_z_eff_true, all_z_eff_hunemohr,
#                   color='orange', label='Hunemohr Model')
# axs[0, 1].plot(all_z_eff_true, all_z_eff_true,
#                color='red', linestyle='--', label='True')
# axs[0, 1].set_xlabel('True z_eff')
# axs[0, 1].set_ylabel('Calculated z_eff')
# axs[0, 1].set_title('z_eff: True vs Calculated (All Files - Zoomed Out)')
# axs[0, 1].legend()
# axs[0, 1].grid(True)

# # Zoomed-out: Plot for stopping power ratio (SPR)
# axs[0, 2].scatter(all_spr_true, all_spr_saito,
#                   color='purple', label='Saito Model')
# axs[0, 2].scatter(all_spr_true, all_spr_hunemohr,
#                   color='orange', label='Hunemohr Model')
# axs[0, 2].plot(all_spr_true, all_spr_true, color='red',
#                linestyle='--', label='True')
# axs[0, 2].set_xlabel('True Stopping Power Ratio')
# axs[0, 2].set_ylabel('Calculated Stopping Power Ratio')
# axs[0, 2].set_title(
#     'Stopping Power Ratio: True vs Calculated (All Files - Zoomed Out)')
# axs[0, 2].legend()
# axs[0, 2].grid(True)

# # Zoomed-in: Plot for z_eff
# axs[1, 0].scatter(filtered_z_eff_true, filtered_z_eff_saito,
#                   color='blue', label='Saito Model')
# axs[1, 0].scatter(filtered_z_eff_true, filtered_z_eff_hunemohr,
#                   color='orange', label='Hunemohr Model')
# axs[1, 0].plot(filtered_z_eff_true, filtered_z_eff_true,
#                color='red', linestyle='--', label='True')
# axs[1, 0].set_xlabel('True z_eff')
# axs[1, 0].set_ylabel('Calculated z_eff')
# axs[1, 0].set_title('z_eff: True vs Calculated (Zoomed In)')
# axs[1, 0].legend()
# axs[1, 0].grid(True)

# plt.tight_layout()
# plt.show()
