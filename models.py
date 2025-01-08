import math
import numpy as np

'''
These method formulas are references from the following:
https://github.com/NewYorkProtonCenter/dual-energy-ct/blob/main/dect_formula.py
'''

def beta_proton(ke):
    """
    Calculate the beta of proton with kinetic energy
    Argument:
        ke: kinetic energy of proton (MeV)
    """
    m0 = 938.272  # MeV/c^2
    c = 299792458  # m/s
    ene = m0 + ke
    gamma = ene / m0
    return 1 - (1 / (gamma * gamma))  # math.sqrt(1 - (1 / (gamma * gamma)))

def rho_e_saito(hu1, hu2, alpha, a):
    """
    The electron density ratio (Phys. Med. Biol. 62 (2017) 7056).
    Arguments:
        hu1 (float): 1st CT-value in HU
        hu2 (float): 2nd CT-value in HU
        alpha (float): calibration constant
        a (float): calibration constant
    """
    delta_hu = (1.0 + alpha) * hu2 - alpha * hu1
    return a * delta_hu / 1000.0 + 1

def z_eff_saito(hu1, hu2, rho_e, n, beta, c, d):
    """
    The effective atomic number (Phys. Med. Biol. 62 (2017) 7056).
    Arguments:
        hu1 (float): 1st CT-value in HU
        hu2 (float): 2nd CT-value in HU
        rho_e (float): electron density ratio
        n (float): power law
        beta (float): calibration constant
        c (float): calibration constant
        d (float): calibration constant
    """
    delta_hu = (1.0 + beta) * hu2 - beta * hu1
    inpow = (c * delta_hu / 1000.0 + d) / rho_e
    if inpow.any() < 0:
        print('in pow:', (c * delta_hu / 1000.0 + d) / rho_e)
    return pow((c * delta_hu / 1000.0 + d) / rho_e, 1.0/n)

def z_eff_hunemohr(hu_1, hu_2, r_e, d_e, n):
    """
    The effective atomic number (Phys. Med. Biol. 59 (2014) 83).
    Arguments:
        hu_1 (float): 1st CT-value in HU
        hu_2 (float): 2nd CT-value in HU
        r_e (float): the electron density ratio
        d_e (float): calibration parameter
        n: power
    """
    z_eff_w = 7.48  # 7.45  # (Phys. Med. Biol. 59 (2014) 83)
    z_eff_w_n = pow(z_eff_w, n)
    return pow((d_e * (hu_1 / 1000.0 + 1.0) + (z_eff_w_n - d_e) * (hu_2 / 1000.0 + 1.0)) / r_e, 1.0/n)

def z_eff_truth(w_m, Z, A, n):
    """
    The effective atomic number (Phys. Med. Biol. 59 (2014) 83).
    Arguments:
        w_m: weight fraction of elements for material
        Z: atomic number of elements
        A: atomic weight of elements
        n: the fitting parameter
    """
    numerator = 0
    denominator = 0
    for ele_i in range(0, len(w_m)):
        numerator = numerator + w_m[ele_i] * pow(Z[ele_i], (n+1)) / A[ele_i]
        denominator = denominator + w_m[ele_i] * Z[ele_i] / A[ele_i]
    return pow(numerator / denominator, 1.0/n)

def rho_e_truth(rho_m, w_m, rho_w, w_w, Z, A):
    """
    The ln of mean excitation potential (truth).
    Arguments:
        rho_m: density of material
        w_m: weight fraction of elements for material
        rho_w: density of water
        w_w: weight fraction of elements for water
        Z: atomic number of elements
        A: atomic weight of elements
    """

    n_m = 0
    for ele_i in range(0, len(w_m)):
        n_m = n_m + w_m[ele_i] * Z[ele_i] / A[ele_i]
    n_w = 0
    for ele_i in range(0, len(w_w)):
        n_w = n_w + w_w[ele_i] * Z[ele_i] / A[ele_i]
    return rho_m * n_m / (rho_w * n_w)

def rho_e_hunemohr(hu_1, hu_2, c_e):
    """
    The electron density ratio (Phys. Med. Biol. 59 (2014) 83).
    Arguments:
        c_e (float): calibration parameter
        hu_1 (float): 1st CT-value in HU
        hu_2 (float): 2nd CT-value in HU
    """
    return c_e * (hu_1 / 1000.0 + 1.0) + (1.0 - c_e) * (hu_2 / 1000.0 + 1.0)


def ln_i_fit(z):
    """
    The ln of mean excitation potential (Phys. Med. Biol. 59 (2014) 83).
    Arguments:
        z (numpy.ndarray): effective atomic number
    """
    if z > 8.5:
        a = 0.098
        b = 3.376
    else:
        a = 0.125
        b = 3.378

    return a * z + b

def ln_i_truth(w_m, Z, A, I):
    """
    The ln of mean excitation potential (truth).
    Arguments:
        w_m: weight fraction of elements for material
        Z: atomic number of elements
        A: atomic weight of elements
        I: ionization energy of elements
    """

    numerator = 0
    denominator = 0
    for ele_i in range(0, len(w_m)):
        numerator = numerator + w_m[ele_i] * \
            Z[ele_i] / A[ele_i] * math.log(I[ele_i])
        denominator = denominator + w_m[ele_i] * Z[ele_i] / A[ele_i]
    return numerator / denominator


def spr_w_truth(rho, ln_i_m, ln_i_w, beta2):
    """
    The stopping power ratio to water based on Bethe formula.
    Arguments:
        rho: electron density ratio
        beta2: (v/c)^2 of the incident particle
        ln_i_m: mean excitation potential
        ln_i_w: mean excitation potential of water
    """
    return rho * (math.log(2 * 511000.0 * beta2 / (1 - beta2)) - beta2 - ln_i_m) / (math.log(2 * 511000.0 * beta2 / (1 - beta2)) - beta2 - ln_i_w)

def sp_truth(za, ln_i_m, beta2):
    """
    The stopping power based on Bethe formula.
    Arguments:
        za: Z/A
        beta2: (v/c)^2 of the incident particle
        ln_i_m: mean excitation potential
    """
    return 0.307075*za/beta2*(math.log(2 * 511000.0 * beta2 / (1 - beta2)) - beta2 - ln_i_m)

def sigma_christian0(v, z, a, m):
    """
    The relative cross-section to water for material with (w, z).
    Arguments:
        v: electron density fractions
        z: atomic numbers
        a: fitting par
        b: fitting par
        m: fitting par
    """
    sum_i = 0
    for i in range(0, len(z)):
        sum_i = sum_i + v[:, i] * pow(z[i], m)
    b = (1 - a)/(0.2 + 0.8 * pow(8, m))
    return a + b * sum_i

def sigma_christian(v, z, a, m):
    """
    The relative cross-section to water for material with (w, z).
    Arguments:
        v: electron density fractions
        z: atomic numbers
        a: fitting par
        m: fitting par
    """
    b = (1 - a)/(0.2 + 0.8 * pow(8, m))
    sigma = np.zeros(len(v))
    for j in range(0, len(v)):
        sum_i = 0
        for i in range(0, len(z)):
            sum_i = sum_i + v[j][i] * pow(z[i], m)
        sigma[j] = a + b * sum_i
    return sigma

def sigma_mono_ene(hu_l, hu_h, rho_e, alpha):
    """
    The relative cross-section to water for material with (w, z).
    Arguments:
        hu_l: HU low
        hu_h: HU high
        rho_e: electron density ratio
        alpha: fitting parameter
    """
    sigma = np.zeros(len(rho_e))
    for i in range(0, len(rho_e)):
        mu_w_l_i = hu_l[i] / 1000.0 + 1.0
        mu_w_h_i = hu_h[i] / 1000.0 + 1.0
        sigma_w_l_i = mu_w_l_i / rho_e[i]
        sigma_w_h_i = mu_w_h_i / rho_e[i]
        sigma[i] = alpha * sigma_w_l_i + (1.0 - alpha) * sigma_w_h_i
    return sigma

def rho_e_trujillo(hu_j, omega_j, Z_N, E_ji, E_jiw, Z, Zw, F_func, G_func):
    """
    Calculate electron density (rho_e) using the Trujillo-Bastidas method.
    Arguments:
        hu_j (float): Hounsfield Unit for X-ray spectrum j
        omega_j (numpy.ndarray): Weights for energy spectra j
        Z_N (float): Normalization constant for atomic number
        E_ji (numpy.ndarray): Energy values for material (80, 120, 180 kVP)
        E_jiw (numpy.ndarray): Energy values for water (80, 120, 180 kVP)
        Z (float): Atomic number for material
        Zw (float): Atomic number for water
        F_func (callable): Function to calculate F(E, Z)
        G_func (callable): Function to calculate G(E, Z)
    """
    numerator = np.sum(omega_j * (Z_N**-1 * F_func(E_ji, Z) + G_func(E_ji, Z)))
    denomintator = np.sum(omega_j * (Z_N**-1 * F_func(E_jiw, Zw) + G_func(E_jiw, Zw)))
    return (hu_j / 1000 + 1) * (denomintator / numerator)

def z_eff_trujillo(U1w, U2w, omega1, omega2, Z_N, Z, E1i, E2i, Zw, F_func, G_func):
    """
    Calculate effective atomic number (Z_Eff) using the Trujillo-Bastidas method.
    Arguments:
        U1w (float): Linear attenuation coefficient ratio for X-ray spectrum 1
        U2w (float): Linear attenuation coefficient ratio for X-ray spectrum 2
        omega1 (numpy.ndarray): Weights for energy spectra 1
        omega2 (numpy.ndarray): Weights for energy spectra 2
        Z_N (float): Normalization constant for atomic number
        Z (float): Atomic number for material
        E1i (numpy.ndarray): Energy values for material at spectrum 1 (80, 120, 180 kVP)
        E2i (numpy.ndarray): Energy values for material at specturm 2 (80, 120, 180 kVP)
        Zw (float): Atomic number for water
        F_func (callable): Function to calculate F(E, Z)
        G_func (callable): Function to calculate G(E, Z)
    """
    num1 = np.sum(omega2 * Z_N * F_func(E2i, Zw))
    num2 = np.sum(omega1 * (G_func(E1i, Z) + F_func(E1i, Z)))
    denom1 = np.sum(omega1 * Z_N * F_func(E1i, Zw))
    denom2 = np.sum(omega2 * (G_func(E2i, Z) + F_func(E2i, Z)))

    z_eff_num = U2w * num1 * num2 - U1w * denom1 * denom2
    z_eff_denom = U1w * denom1 * denom2 - U2w * num1 * num2
    return z_eff_num / z_eff_denom

def F(E, Z):
    """
    Will calculate terms based on photon cross-sectional data
    Args:
        E (_type_): _description_
        Z (_type_): _description_
    """
    return

def G(E, Z):
    return