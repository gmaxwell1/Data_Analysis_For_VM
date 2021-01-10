""" 
filename: general_functions.py

The following functions are of general use for file management and simple calculations. 
They also contain transformations between the different coordinate systems in use,
namely stage coordinates, (Metrolab) sensor coordinates and magnet coordinates. 

Author: Nicholas Meinhardt (QZabre)
        nmeinhar@student.ethz.ch
        
Date: 20.10.2020
"""

#%%
# standard library imports
import numpy as np
from scipy.optimize import curve_fit



#%%
def sensor_to_magnet_coordinates(data):
    """
    Transform from Metrolab sensor coordinates to magnet coordinates, using the following transformation:

    x -> -y
    y -> z
    z -> -x

    Arg: data (ndarray) can be 1d or multidimensional array, where the last dimension must be of length 3 and contain 
    x,y,z data. 
    """
    data_magnet_coords = np.zeros_like(data)
    
    # treat 1d arrays and multi-dimensional arrays differently
    if len(data.shape)==1:
        data_magnet_coords[0] = -data[2]
        data_magnet_coords[1] = -data[0]
        data_magnet_coords[2] = data[1]
    else:
        data_magnet_coords[...,0] = -data[...,2]
        data_magnet_coords[...,1] = -data[...,0]
        data_magnet_coords[...,2] = data[...,1]

    return data_magnet_coords

def magnet_to_sensor_coordinates(data):
    """
    Transform from magnet coordinates to Metrolab sensor coordinates, using the following transformation:

    x -> -z
    y -> -x
    z -> y

    Arg: data (ndarray) can be 1d or multidimensional array, where the last dimension must be of length 3 and contain 
    x,y,z data. 
    """
    data_magnet_coords = np.zeros_like(data)
    
    # treat 1d arrays and multi-dimensional arrays differently
    if len(data.shape)==1:
        data_magnet_coords[0] = -data[1]
        data_magnet_coords[1] = data[2]
        data_magnet_coords[2] = -data[0]
    else:
        data_magnet_coords[...,0] = -data[...,1]
        data_magnet_coords[...,1] = data[...,2]
        data_magnet_coords[...,2] = -data[...,0]

    return data_magnet_coords

def transform_between_sensor_stage_coordinates(data):
    """
    Transform from magnet coordinates to Metrolab sensor coordinates, using the following transformation:

    x -> -y, 
    y -> -x

    Arg: data (ndarray) can be 1d or multidimensional array, where the last dimension must be of length 3 and contain 
    x,y,z data. 
    """
    data_magnet_coords = np.zeros_like(data)
    
    # treat 1d arrays and multi-dimensional arrays differently
    if len(data.shape)==1:
        if len(data) == 2:
            data_magnet_coords[0] = -data[1]
            data_magnet_coords[1] = -data[0]
        else:
            data_magnet_coords[0] = -data[1]
            data_magnet_coords[1] = -data[0]
            data_magnet_coords[2] = data[2]
    else:
        if data.shape[-1] == 2:
            data_magnet_coords[...,0] = -data[...,1]
            data_magnet_coords[...,1] = -data[...,0]
        else:
            data_magnet_coords[...,0] = -data[...,1]
            data_magnet_coords[...,1] = -data[...,0]
            data_magnet_coords[...,2] = data[...,2]

    return data_magnet_coords

def get_direction_vector(vecs):
    """
    Estimate the overall normalized direction vector within the xy-plane for a set of vectors, 
    pointing from (about) along the curve defined by the vectors and projected onto the xy-plane.

    Args:
    - vecs (ndarray of shape (N,3) or (N,2)): Input data vectors (for example of measured 
    magnetic field vectors). The third dimension is ignored entirely. 

    Returns: direction (1d-ndarray of length 2): normalized direction vector
    """
    # first check that x-data are not too close to 0, take 5 mT as boundary
    if not np.all(np.abs(vecs[:,0]) < 5 ):
        # fit data projected onto xy-plane with linear function plus an offset
        lin_fct = lambda x, a, b: a*x +b
        p, _ = curve_fit(lin_fct, vecs[:,0], vecs[:,1], p0 = [1,0])
        
        # take the x-value with largest magnitude to correctly set positive direction along provided data
        x = vecs[np.argmax(np.abs(vecs[:,0])), 0]
        direction = np.array([x, lin_fct(x,*p)-lin_fct(0,*p)])

    # else do the same thing as before, but exchange x and y when fitting:
    else:
        lin_fct = lambda x, a, b: a*x +b
        p, _ = curve_fit(lin_fct, vecs[:,1], vecs[:,0], p0 = [1,0])
        
        # take the y-value with largest magnitude to correctly set positive direction along provided data
        y = vecs[np.argmax(np.abs(vecs[:,1])), 1]
        direction = np.array([lin_fct(y,*p)-lin_fct(0,*p), y])

    # normalize direction before returning
    return direction / np.linalg.norm(direction)

def get_relative_inplane_angles(fields, verbose=False):
    """
    Estimate the two angles between the directions of first + second entry along first dimension 
    and between directions of first and third entry along first dimension, each projected onto the xy-plane. 
    This function is supposed to be used with data that originate from ramping the current in 
    the three coils individually, while the remaining coils are switched off.  

    Args:
    - fields (ndarray of shape (3, N, 3)): estimated fields with the three field directions on last axis
    and the data originating from ramping the coils individually along first dimension.
    - verbose (bool): If True, print the resulting three directions

    Return angles (1d-ndarray of length 2), containing the angles between virgin hysteresis curves of
    first and second coil, first and third coil, and second and third coil. 
    """
    # get normalized direction vectors first
    directions = np.zeros((3,2))
    for i in range(3):
        directions[i] = get_direction_vector(fields[i])
    
    if verbose:
        [print('direction for coil {}: {}'.format(i+1, directions[i])) for i in range(3)]

    # estimate angles using dot product
    angles = np.arccos(np.array([np.dot(directions[0], directions[1]),
                                np.dot(directions[0], directions[2]),
                                np.dot(directions[1], directions[2]) ]))

    # convert to degrees before returning
    return np.degrees(angles)
