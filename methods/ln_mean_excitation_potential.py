def ln_mean_excitation_potential(z_eff):
    mep = []
    for z in z_eff:
        if z > 8.5:
            a = 0.098
            b = 3.376
        else:
            a = 0.125
            b = 3.378
        mep.append(a * z + b)
    return mep
