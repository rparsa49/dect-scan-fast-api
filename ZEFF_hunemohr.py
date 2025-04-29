import numpy as np
import pandas as pd

element_properties = {
    "H": {"number": 1, "mass": 1.008, "ionization": 19.2},
    "B": {"number": 5, "mass": 10.81, "ionization": 76},
    "C": {"number": 6, "mass": 12.011, "ionization": 78},
    "N": {"number": 7, "mass": 14.007, "ionization": 82},
    "O": {"number": 8, "mass": 15.999, "ionization": 95},
    "F": {"number": 9, "mass": 18.998401316, "ionization": 115},
    "Na": {"number": 11, "mass": 22.9897693, "ionization": 149},
    "Mg": {"number": 12, "mass": 24.305, "ionization": 156},
    "Al": {"number": 13, "mass": 26.981538, "ionization": 166},
    "Si": {"number": 14, "mass": 28.085, "ionization": 173},
    "P": {"number": 15, "mass": 30.97376200, "ionization": 173},
    "Cl": {"number": 17, "mass": 35.45, "ionization": 174},
    "Ca": {"number": 20, "mass": 40.08, "ionization": 191},
    "Ti": {"number": 22, "mass": 47.867, "ionization": 233}
}

materials = {
    "Lung": {
        "composition": {"H": 0.0847, "C": 0.5957, "N": 0.0197, "O": 0.1811, "Mg": 0.1121, "Si": 0.0058, "Cl": 0.0010},
        "density": 0.460
    },
    "Adipose": {
        "composition": {"H": 0.0906, "C": 0.7230, "N": 0.0225, "O": 0.1627, "Ca": 0.0013},
        "density": 0.942
    },
    "Breast": {
        "composition": {"H": 0.0859, "C": 0.7011, "N": 0.0233, "O": 0.1790, "Ca": 0.0013, "Ti": 0.0095},
        "density": 0.988
    },
    "Brain": {
        "composition": {"H": 0.1083, "C": 0.7254, "N": 0.0169, "O": 0.1486, "Cl": 0.0008},
        "density": 1.052
    },
    "Liver": {
        "composition": {"H": 0.0806, "C": 0.6701, "N": 0.0247, "O": 0.2001, "Cl": 0.0014, "Ti": 0.0231},
        "density": 1.089
    },
    "Inner Bone": {
        "composition": {"H": 0.0667, "C": 0.5564, "N": 0.0196, "O": 0.2352, "P": 0.0323, "Ca": 0.0886},
        "density": 1.147
    },
    "Cortical Bone": {
        "composition": {"H": 0.0341, "C": 0.3141, "N": 0.0184, "O": 0.3650, "Ca": 0.2681},
        "density": 1.823
    }
}

# Calculate Z_eff for each material
results = []

def z_eff_truth(weight_fractions, atomic_numbers,
                atomic_weights, n = 3.1):
    """
    The effective atomic number (Phys. Med. Biol. 59 (2014) 83).
    Arguments:
        weight_fractions: weight fraction of elements for material
        atomic_numbers: atomic number of elements
        atomic_weights: atomic weight of elements
        n: the fitting parameter, determined to be 3.1
    """
    numerator = (weight_fractions * atomic_numbers **
                 (n + 1) / atomic_weights).sum()
    denominator = (weight_fractions * atomic_numbers / atomic_weights).sum()
    return (numerator / denominator)**(1. / n)


for material_name, material_data in materials.items():
    composition = material_data["composition"]
    n_i_unnorm = []
    Z_i = []
    As = []

    for element, mass_frac in composition.items():
        props = element_properties[element]
        Z = props["number"]
        A = props["mass"]
        As.append(A)
        n_i_unnorm.append(mass_frac * Z / A)
        Z_i.append(Z)

    n_i_unnorm = np.array(n_i_unnorm)
    Z_i = np.array(Z_i)
    n_i = n_i_unnorm / np.sum(n_i_unnorm)

    z_eff = z_eff_truth(n_i, Z_i, As)
    results.append({"Material": material_name,
                   "z_eff_truth": round(z_eff, 4)})

# Display as DataFrame
df_results = pd.DataFrame(results)
print(df_results)
