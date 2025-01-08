def alpha_saito(mew_l, mew_h, mew_lw, mew_hw, rho):
    '''
    Saito 2012
    mew_l: linear attenuation coeff. at low energy
    mew_h: linear attenuation coeff. at high energy
    mew_lw: linear attenuation coeff. of water at low energy
    mew_hw: linear attenuation coeff. of water at high energy
    rho: electron density
    '''
    numerator = 1
    denominator = ((mew_l / mew_lw) - rho) / ((mew_h / mew_hw) - rho) - 1
    return numerator/denominator


def hu_saito(alpha, high, low):
    '''
    Saito 2012
    alpha: weighing factor
    high: CT high
    low: CT low
    '''
    return (1+alpha)*high-(alpha*low)


def rho_e_saito(HU):
    '''
    Saito 2012
    '''
    return (HU/1000) + 1


def mew_saito(HU):
    '''
    Saito 2017
    HU: CT number at either high or low energy
    '''
    return HU/1000 + 1


def z_eff_saito(mew, lam, rho):
    '''
    Saito 2017 
    mew: linear attenuation coefficient (high or low)
    lam: 1/Q(E) is a material independent proportionality constant
    rho: electron density
    '''
    return lam*(mew/rho - 1) + 1
