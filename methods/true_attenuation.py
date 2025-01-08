import json
from pathlib import Path

DATA_DIR = Path("data")
def load_json(file_name):
    with open(DATA_DIR / file_name, "r") as file:
        return json.load(file)

ELEMENT_PROPERTIES = load_json("element_properties.json")

# Method for linear attenuation of a material
def linear_attenuation(material_info):
    rho = material_info["density"]
    composition = material_info["composition"]

    mu_total = 0.0
    for element, fraction in composition.items():
        # get elemental properties
        atomic_mass = ELEMENT_PROPERTIES[element]["mass"]
        atomic_number = ELEMENT_PROPERTIES[element]["number"]

        # number density of the element in the material
        N = (rho * fraction) / atomic_mass

        mu_a = atomic_number

        mu_total += mu_a * N
    return mu_total

