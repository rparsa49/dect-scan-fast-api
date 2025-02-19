import numpy as np

def bvm_weights(mew_1h, mew_1l, mew_2h, mew_2l, mew_l, mew_h):
    '''
    Chika 2024
    mew_1h: basis material 1 linear attenuation coeff. at high energy
    mew_1l: basis material 1 linear attenuation coeff. at low energy
    mew_2h: basis material 2 linear attenuation coeff. at high energy
    mew_2l: basis material 2 linear attenuation coeff. at low energy
    mew_l: unknown tissue linear attenuation coeff. at low energy
    mew_h: unknown tissue linear attenuation coeff. at high energy
    
    return: [c1, c2]: energy independent weights
    '''
    # Coefficient matrix
    A = np.array([[mew_1l, mew_2l], [mew_1h, mew_2h]])
    
    # Vector of linear attenuation coefficients
    b = np.array([mew_l, mew_h])
    
    # Solve for weights
    weights = np.linalg.inv(A).dot(b)
    
    return weights
    

def bvm_rho_e(weights, rho_1, rho_2):
    '''
    Chika 2024
    weights: energy independent weights c1, c2
    rho_1: electron density of material 1
    rho_2: electron density of material 2
    
    return: electron density of unknown material
    '''
    return weights[0]*rho_1 + weights[1]*rho_2
