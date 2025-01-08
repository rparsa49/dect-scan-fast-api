def ln_mean_excitation_potential(z_eff):
    if z_eff > 8.5:
        a = 0.098
        b = 3.376
    else:
        a = 0.125
        b = 3.378

    return a * z_eff + b
