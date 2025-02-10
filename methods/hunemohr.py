import numpy as np
from pathlib import Path
import json

DATA_DIR = Path("data")


def load_json(file_name):
    with open(DATA_DIR / file_name, "r") as file:
        return json.load(file)


WATER_SPR = load_json("water_sp.json")

def rho_e_hunemohr(HU_h, HU_l, c):
    '''
    Hunemohr 2014
    HU_h: CT Number at High Energy
    HU_l: CT Number at Low Energy
    c: calibration parameter
    '''
    return c * (HU_h/1000 + 1) + (1 - c) * (HU_l/1000) + 1

def spr_hunemohr(rho, Z, kvp):
    '''
    Hunemohr 2014
    rho: Electron density ratio of the material to water
    Z: Effective atomic number
    '''
    a = 0.125 if Z <= 8.5 else 0.098
    b = 3.378 if Z <= 8.5 else 3.376
    return (rho * ((12.77 - (a * Z + b)) / 8.45)) / WATER_SPR.get(str(kvp))


def z_eff_hunemohr(n_i, Z_i, n=3.1):
    '''
    Hunemohr 2014
    n_i: List of number densityies for each atom type
    Z_i: List of atomic numbers for each atom type
    n: Exponent for effective atomic number
    '''
    num = np.sum(n_i * (Z_i ** (n+1)))
    den = np.sum(n_i * Z_i)
    
    Z_eff = (num/den) ** (1/n)
    return Z_eff
