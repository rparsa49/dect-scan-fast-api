import math, json
from pathlib import Path

DATA_DIR = Path("data")
def load_json(file_name):
    with open(DATA_DIR / file_name, "r") as file:
        return json.load(file)
WATER_SPR = load_json("water_sp.json")

def sp_truth(z, a, ln_i_m, beta2, kvp):
    spr = []
    for zeff, mep in zip(z, ln_i_m):
        num = 0.307075*(zeff/a)/beta2*(math.log(2 * 511000.0 * beta2 / (1 - beta2)) - beta2 - mep)
        spr.append(num/WATER_SPR.get(str(kvp)))
    return spr