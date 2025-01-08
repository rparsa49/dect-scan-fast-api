import math
def sp_truth(z, a, ln_i_m, beta2):
    return 0.307075*(z/a)/beta2*(math.log(2 * 511000.0 * beta2 / (1 - beta2)) - beta2 - ln_i_m)
