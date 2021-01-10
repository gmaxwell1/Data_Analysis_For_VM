""" 
filename: analysis_tools.py

This file contains functions that are supposed to simplify data analysis, 
e.g. by estimating polar/azimuthal angles, relative angles...

Author: Nicholas Meinhardt (Qzabre)
        nmeinhar@student.ethz.ch
        
Date: 09.10.2020
"""

# standard library imports
import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, MaxNLocator
from datetime import datetime
from itertools import product
from scipy.optimize import curve_fit
from scipy.spatial.transform import Rotation as R

# local imports
# from modules.interpolation_tools import find_start_of_saturation


# %%
def estimate_std_theta(mean_values, std_values):
    """
    Estimate the standard deviation of the estimated angle theta (wrt. z-axis), 
    assuming that x,y,z components of magnetic field are independent variables.

    Args: mean_data_specific_sensor, std_data_specific_sensor are ndarrays of shape (#measurements, 3).

    Returns: ndarray of length #measurements containing standard deviations of theta
    """
    # estimate magnitudes
    mag = np.linalg.norm(mean_values, axis=1)

    # estimate partial derivatives
    deriv_arccos = 1 / np.sqrt(1 - mean_values[:, 2]**2 / mag**2)

    pderiv_x = deriv_arccos * mean_values[:, 2]*mean_values[:, 0] / mag**3
    pderiv_y = deriv_arccos * mean_values[:, 2]*mean_values[:, 1] / mag**3
    pderiv_z = deriv_arccos * (mean_values[:, 2]**2 / mag**3 - 1/mag)

    var_theta = (pderiv_x*std_values[:, 0])**2 + (pderiv_y *
                                                  std_values[:, 1])**2 + (pderiv_z*std_values[:, 2])**2
    return np.degrees(np.sqrt(var_theta))

def estimate_std_phi(mean_values, std_values):
    """
    Estimate the standard deviation of the estimated in-plane angle phi (wrt. x-axis) in degrees, 
    assuming that x,y,z components of magnetic field are independent variables.

    Args: mean_data_specific_sensor, std_data_specific_sensor are ndarrays of shape (#measurements, 3).

    Returns: ndarray of length #measurements containing standard deviations of phi in degrees
    """
    # estimate partial derivatives
    deriv_arctan = np.cos(np.arctan2(mean_values[:,1], mean_values[:,0]))**2

    pderiv_x = deriv_arctan * (-mean_values[:,1]/ mean_values[:,0]**2)
    pderiv_y = deriv_arctan / mean_values[:,0]

    var_phi = (pderiv_x*std_values[:,0])**2 + (pderiv_y*std_values[:,1])**2 
    return np.degrees(np.sqrt(var_phi))

def estimate_std_magnitude(mean_values, std_values):
    """
    Estimate the standard deviation of the estimated magnitude |B|, 
    assuming that x,y,z components of magnetic field are independent variables.

    Args: mean_values, std_values are ndarrays of shape (#measurements, 3).

    Returns: ndarray of length #measurements containing standard deviations of |B|
    """
    # estimate magnitudes
    mag = np.linalg.norm(mean_values, axis=1)

    return np.sqrt(np.einsum('ij,ij->i', mean_values**2, std_values**2)) / mag


def estimate_std_inplane(mean_values, std_values):
    """
    Estimate the standard deviation of the estimated in-plane magnitude |B_xy|, 
    assuming that x,y,z components of magnetic field are independent variables.

    Args: mean_values, std_values are ndarrays of shape (#measurements, 3).

    Returns: ndarray of length #measurements containing standard deviations of |B_xy|
    """
    # estimate magnitudes
    inplane_mag = np.linalg.norm(mean_values[:, 0:2], axis=1)

    return np.sqrt(np.einsum('ij,ij->i', mean_values[:, 0:2]**2, std_values[:, 0:2]**2)) / inplane_mag

def get_phi(values, cut_phi_at_0=False):
    """
    Return the in-plane angle phi with respect to x-axis in degrees.
    """
    angles = np.degrees(np.arctan2(values[...,1], values[...,0]))

    # if cut should be at 0 degrees, add 360 degrees to all negative values
    if cut_phi_at_0:
        mask = angles < 0
        angles[mask] = 360 + angles[mask]

    return angles

def get_theta(values):
    """
    Return the angle theta between z axis and the provided values in degrees.
    """
    mag = np.linalg.norm(values, axis=-1)
    angles = np.degrees(np.arccos(values[...,2]/mag))

    return angles

def angle_wrt_z(vec):
    """
    Return angle (radian) of vector with respect to z axis.
    """
    mag = np.linalg.norm(vec)
    return np.arccos(vec[2]/mag)

def inplane_angle_wrt_x(vec):
    """
    Return angle (radian) of vector with respect to z axis.
    """
    return np.arctan2(vec[1], vec[0])

def estimate_power(ratios, value, coil_number=1, R = 0.47 ):
    """
    Return the total power [W] of all three coils when the current in the passed coil (1, 2 or 3) is value [A].

    Args:
    - ratios (nonzero 1d-ndarray of length 3): ratios of the currents in the three coils. This can be the first element of 
    the current array of shape (number measurements, 3). Note that the entry corresponding to the provided coil_number
    must not be 0, because this coil would have zero current for the whole measurement series. 
    - value (float): current value [A] in coil with number coil_number
    - coil_number (int): number of the coil for which value is provided, can be 1, 2 or 3.
    - R (float): resistance [Ohm] of a coil 

    Return: power (float)
    """
    # check reasonability of inputs
    if np.all(ratios == 0):
        raise ValueError('ratios must be non-zero!')
    if coil_number not in [1,2,3]:
        raise ValueError('coil_number must be in \{1,2,3\}, not {}!'.format(coil_number))
    if ratios[coil_number-1] == 0:
        raise ValueError('The provided ratios are invalid, since coil {} has zero current.'.format(coil_number))
    
    # normalize ratios, such that the desired coil has a factor 1, and multiply by value
    currents = value * ratios/ratios[coil_number-1]

    return R * np.sum(currents**2)  # sum over all three coils


def fit_rotation_matrix(mean_values, expected_values, convention = 'xzx'):
    """
    Fits a rotation that yields the minimum sum of square distances between mean_values and rotated
    expected_values. This method uses the curve_fit method of scipy.optimize, which makes use of the 
    least_squares method.

    Args: 
    - mean_values, expected_values (ndarrays of shape (#measurements, 3)): contain the measured and expected data
    - convention (str): the convention used to describe a general 3d-rotation by Euler angles. The resulting 
    angles depend on the convention, while the calculated rotation remains the same.

    Returns:
    - p, pcov (list of floats): the estimated Euler angles that achieve the best fitting rotation.
    - rotated_expections (ndarray of shape (#measurements, 3)): the estimated values when applying the
    final rotation to the expected_values 
    """
    p, pcov = curve_fit(lambda x, alpha, beta, gamma: apply_rotation(x, alpha, beta, gamma, convention=convention).flatten(), 
                        expected_values, mean_values.flatten(), p0=[5,90,0])
    return p, pcov, apply_rotation(expected_values, *p)

def apply_rotation(x, alpha, beta, gamma, convention = 'xzx'):
    """ 
    Applies a rotation with Euler angles alpha, beta and gamma to an input vector

    Args: 
    - x (ndarray of length 3 or shape (N,3)): Input vector(s) which should be rotated
    - alpha, beta, gamma (flaot): Euler angles of the rotation
    - convention (str): the convention used to describe a general 3d-rotation by Euler angles.

    Returns the rotated input vectors
    """
    r = R.from_euler(convention, [alpha, beta, gamma], degrees=True)
    return r.apply(x)

def rotation_on_basis_vectors(alpha, beta, gamma, convention = 'xzx', verbose=True):
    """ 
    Estimate the effect of a rotation with Euler angles alpha, beta, gamma on the coordinate axes.
    Note that considering the coordinate axes actually involves the inverse of the rotation.

    Args:
    - alpha, beta, gamma (flaot): Euler angles of the rotation
    - convention (str): the convention used to describe a general 3d-rotation by Euler angles.
    - verbose (bool): Flag to switch on/off additional printing of effects

    Returns:
    - rotated (ndarray of shape (3,3)): Contains coordinate axes of the rotated system in the original coordinates.
    The first axis covers the three axes, the second the respective coordinates.
    - delta_phi (ndarray of length 3): differences between the polar angles of the original and 
    transformed axes (expressed in original coordinates)
    - delta_phi (ndarray of length 3): differences between the azimuthal angles of the original and 
    transformed axes (expressed in original coordinates)
    """
    r = R.from_euler(convention, [alpha, beta, gamma], degrees=True).inv()
    basis_vectors = np.array([[1,0,0], [0,1,0], [0,0,1]])
    rotated = np.array([r.apply(basis_vectors[i]) for i in range(3)])

    delta_phi = get_phi(basis_vectors) - get_phi(rotated)
    delta_theta = get_theta(basis_vectors) - get_theta(rotated)

    if verbose:
        print('effect on axes (note that this involves inverse of rotation):')
        axes = ['x', 'y', 'z']
        for i in range(3):
            print('{}-axis -> {} (Dphi = {:.2f}°, Dtheta = {:.2f}°)'.format(axes[i], np.round(rotated[i],2),
                                                                        delta_phi[i], delta_theta[i]))
    
    return rotated, delta_phi, delta_theta


def evaluate_performance(measured, fitted):
    """
    Evaluate the performance of the fit by estimating different measured of 
    deviation between fitted and measured parameters. 
    Estimated parameters are RMS error, and the angular accuracy in terms of 
    RMS angular error as well as mean, std, min, max and median angular error.

    Args:
    - measured, fitted (ndarrays of shape (N,3)): Estimated and fitted values. 
    """
    # estimate RMS errors 
    RMSE = estimate_RMS_error(measured.flatten(), fitted.flatten())

    # evaluate angular accuracy
    dot = np.array([np.dot(measured[i], fitted[i]) for i in range(len(measured))])
    norms_measured = np.linalg.norm(measured, axis=1)
    norms_fits = np.linalg.norm(fitted, axis=1)
    # omit vectors with zero norm due to reasonability
    mask = (norms_measured!=0) * (norms_fits!=0)
    alphas = np.degrees(np.arccos(dot[mask] / (norms_measured[mask] * norms_fits[mask])))


    # print all measures
    print(f'RMS error fit: {RMSE:.2f} mT')
    print('RMS angular error: {:.2f}°'.format(estimate_RMS_error(alphas, np.zeros_like(alphas))))
    print('mean angular error: {:.2f}°, std: {:.2f}°'.format(np.mean(alphas), np.std(alphas)))
    print('min / max angular error: {:.2f}° / {:.2f}°'.format(np.min(alphas), np.max(alphas)))
    print('median angular error: {:.2f}°'.format(np.median(alphas)))

def estimate_RMS_error(x, y):
    """
    Estimate and return root mean square error between x and y.

    Args: x,y (1d ndarrays): contain measured and predicted data
    """
    # print(x-y)
    # print(np.mean(x-y))
    return  np.sqrt(np.mean((x-y)**2))