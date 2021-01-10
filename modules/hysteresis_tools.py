#%%
# standard library imports
import numpy as np
import os


#%%
def sigmoid(x, k, a):
    """
    Sigmoid function with growth rate k and maximum value a, rescaled such that center point is at origin.
    """
    return a * (1 - np.exp(-k*x)) / (1 + np.exp(-k*x))

def deriv_sigmoid(x,k,a):
    return 2*a*k*np.exp(-k*x)/(1+np.exp(-k*x))**2

def second_deriv_sigmoid(x,k,a):
    return 4*a* k**2 *np.exp(-k*x)/ (1+np.exp(-k*x))**3 - 2*a* k**2 *np.exp(-k*x) / (1+np.exp(-k*x))**2

def abs_sigmoid(x, k, a):
    """
    Return absolute value of sigmoid function with growth rate k and maximum value a, 
    rescaled such that center point is at origin.
    """
    return np.abs(sigmoid(x, k, a))

def coth(x):
    """Return cosh(x)/sinh(x)"""
    return np.cosh(x) / np.sinh(x)

def brillouin_fct(x, J, a):
    """
    Implement the Brillouin function, which is used to describe paramagnet. 
    """
    s = 1/(2*J)
    return a * ((1+s) * coth((1+s)*x) - s * coth(s*x))

def abs_brillouin_fct(x, J, a):
    """
    Implement the absolute value of Brillouin function, which is used to describe paramagnet. 
    """
    return np.abs(brillouin_fct(x, J, a))

def lin_and_const(x, x_kink, a):
    """
    Implement a function that raises linearly as a*x until |x|= x_kink and remains constant afterwards.

    Note that x_kink >= 0 is required.
    """
    try:
        len(x)
    except TypeError:
        return a*x if np.abs(x) <= x_kink else a*x_kink*np.sign(x)
    else:
        return np.array([a*x[i] if np.abs(x[i]) <= x_kink else a*x_kink*np.sign(x[i]) for i in range(len(x))])

def abs_lin_and_const(x, x_kink, a):
    """
    Implement a function that raises linearly as |a*x| until |x|= x_kink and remains constant afterwards.

    Note that x_kink >= 0 is required and that the returned value is >= 0.
    """
    return np.abs(lin_and_const(x, x_kink, a))

def get_remanent_B(H_falling, H_rising, B_falling, B_rising):
    # find zero values of H
    i_fall = np.argmin(np.abs(H_falling))
    i_rise = np.argmin(np.abs(H_rising))
    # return y-axis offsets
    return B_falling[i_fall], B_rising[i_rise]

def linearly_fit_root(a, b):
    return -a[1]*(b[0]-a[0])/(b[1]-a[1]) + a[0]

def get_coercivity(H_falling, H_rising, B_falling, B_rising):
    # falling branch
    if np.min(np.abs(B_falling)) == 0:
        # check for zero B values, although it is rather unlikely that an experiments produces exactly zero field 
        i_fall = np.argmin(np.abs(B_falling))
        H_coer_fall = H_falling[i_fall]
    else:
        # take two entries with lowest absolut values, draw line through them and estimate shift of origin
        i1, i2 = np.argsort(np.abs(B_falling))[:2]
        H_coer_fall = linearly_fit_root([H_falling[i1], B_falling[i1]], [H_falling[i2], B_falling[i2]])

    # rising branch
    if np.min(np.abs(B_rising)) == 0:
        # check for zero B values, although it is rather unlikely that an experiments produces exactly zero field 
        i_rise = np.argmin(np.abs(B_rising))
        H_coer_rise = H_rising[i_rise]
    else:
        # take two entries with lowest absolut values, draw line through them and estimate shift of origin
        i1, i2 = np.argsort(np.abs(B_falling))[:2]
        H_coer_rise = linearly_fit_root([H_rising[i1], B_rising[i1]], [H_rising[i2], B_rising[i2]])

    return H_coer_fall, H_coer_rise