""" 
filename: find_center_in_2dScan.py

This file contains functions that can be used to estimate the center position for the 
sensor from 2d scan data. 
(Most of them are currently not in use.)
        
Date: 08.01.2021
"""

# standard library imports
import numpy as np

# local imports
# from modules.analysis_tools import *


#%%
def find_closest_measured_field(pos, field, a, b):
    """
    Return the measured magnetic field at the closest measurement position next to [a,b]
    """
    distances = np.sqrt((pos[:,0]-a)**2 + (pos[:,1]-b)**2)
    index_closest = np.argmin(distances)
    
    return field[index_closest]

def distance_to_center(pos, field, a, b):
    """
    Estimate |B_i - B_ab|_2 for magnetic field vectors B_i at position i and B_ab at position 
    [a,b] for all i in range(len(i)).
    """
    # take the measured field at the position that is closest to [a,b]
    closest_field = find_closest_measured_field(pos, field, a, b)
    return  np.sqrt((field[:,0] - closest_field[0])**2 + (field[:,1] - closest_field[1])**2 + (field[:,2]- closest_field[2])**2)

def weighted_distances(pos, field, a, b):
    """
    Estimate |B_i - B_ab|_2 for magnetic field vectors B_i at position i and B_ab at position 
    [a,b] for all i in range(len(i)).
    """
    # take the measured field at the position that is closest to [a,b]
    distances = np.sqrt((pos[:,0]-a)**2 + (pos[:,1]-b)**2)
    inplane_fields = np.sqrt((field[:,0])**2 + (field[:,1])**2)
    return distances*inplane_fields**2

def find_center_of_mass(pos, field):
    """
    Estimate the resulting weights for each position in pos when assuming that this position is the center, 
    using the distance_to_center function to estimate the weights. 
    Return the position with smalles weight. 
    """
    objectives = np.zeros(len(pos))
    for i in range(len(pos)):
        objectives[i] = np.sum(weighted_distances(pos, field, pos[i,0], pos[i,1]))

    index_min = np.argmin(objectives)
    return pos[index_min, :2]

def find_center(pos, field):
    """
    Estimate the resulting weights for each position in pos when assuming that this position is the center, 
    using the distance_to_center function to estimate the weights. 
    Return the position with smalles weight. 
    """
    objectives = np.zeros(len(pos))
    for i in range(len(pos)):
        objectives[i] = np.sum(distance_to_center(pos, field, pos[i,0], pos[i,1]))

    index_min = np.argmin(objectives)
    return pos[index_min, :2]

def weights_of_rays(pos, field, phi, a, b, length=20, num=20):
    """
    For each point in pos, generate three rays with this point at center, where the rays are at 120 degrees 
    angle to each other and the first one has an angle phi wrt to x-axis.
    For each point on these rays, find the closest position in pos at which a measurement was performed 
    and save the norm of the in-plane field component of the magnetic field measured at this point. 
    Return an ndarray of shape (3, num), where each entry contains the in-plane component of 
    the closest measured point.
    """
    # generate rays at 120 degrees between each other, starting from a, b
    rays, radius_rays = generate_lines(phi, a, b, length=length, num=num)
    
    field_distances = np.ones((3,num))
    for i in range(3):
        for j in range(num):
            # take the measured field at the position that is closest to a point on one of the rays
            closest_field = find_closest_measured_field(pos, field, rays[i, j, 0], rays[i, j, 1])
            field_distances[i,j] = np.linalg.norm(closest_field[:2])
    return field_distances

def find_center_of_rays(pos, field, phi=30, length=20, num=20):
    """
    For each point in pos, estimate the weight using weights_of_rays of this point as center. 
    The weights are returned as ndarray of shape (3, num), average over the squares of the second axis 
    as a weighted sum, where the latter weights are set originate from a Gaussian function.
    Return the point with the smalles weight. 

    Note that the result is heavily dependent on the gaussian function used (i.e. the prefactor and 
    the factor inside the exponential).
    """
    objectives = np.zeros(len(pos))
    for i in range(len(pos)):
        # increase weights of points that are closer to center
        gaussian = lambda x: np.exp(-x/(num/10))
        weights_along_ray = gaussian(np.linspace(0,10,num=20))
        objectives_per_ray = np.average(weights_of_rays(pos, field, phi, pos[i,0], pos[i,1])**2, 
                                            axis=1, weights=weights_along_ray)
        objectives[i] = np.sum(objectives_per_ray)

    index_min = np.argmin(objectives)

    return pos[index_min, :2]

def generate_lines(phi, center_x, center_y, length=20, num=20):
    """
    Generate 3 ndarrays of shape (length, 2) that correspond to three lines in xy-plane starting 
    center position [center_x, center_y] and moving outwards, all at an angle of 120 degrees.
    One of the lines is at angle phi wrt to x-axis.

    Args: 
    - phi (float) is angle wrt to x-axis in degrees
    """
    radius = np.linspace(0, length, num=num)

    angles = np.array([0, 120, -120]) + phi
    angles = np.radians(angles)

    rays = np.zeros((3, num, 2))
    for i in range(3):
        rays[i,:,0] = np.cos(angles[i])*radius[:] + center_x
        rays[i,:,1] = np.sin(angles[i])*radius[:] + center_y

    return rays, radius

def distance_to_lines(pos, field, phi, a, b, length=20, num=20):
    # generate rays at 120 degrees between each other, starting from a, b
    rays, radius_rays = generate_lines(phi, a, b, length=length, num=num)
    
    field_distances = np.ones(len(pos))
    for i in range(len(pos)):
        # find distance from pos[i] to (a,b)
        radius = np.linalg.norm(pos[i,:2]- np.array([a, b]))

        # find three points on ray at the same radius from center 
        i_same_radius = np.argmin(np.abs(radius-radius_rays))
        
        # find the closest point of the three
        i_closest_point = np.argmin(np.linalg.norm(rays[:,i_same_radius] - pos[i,:2], axis=1))

        # take the measured field at the position that is closest to the closest point on one of the rays
        closest_field = find_closest_measured_field(pos, field, 
                            rays[i_closest_point, i_same_radius, 0], rays[i_closest_point, i_same_radius, 1])
        
        field_distances[i] = np.sqrt((field[i,0] - closest_field[0])**2 + (field[i,1] - closest_field[1])**2 )

    return field_distances

def find_center_of_rays(pos, field, phi=30, length=20, num=20):
    """
    For each point in pos, estimate the weight using weights_of_rays of this point as center. 
    The weights are returned as ndarray of shape (3, num), average over the squares of the second axis 
    as a weighted sum, where the latter weights are set originate from a Gaussian function.
    Return the point with the smalles weight. 

    Note that the result is heavily dependent on the gaussian function used (i.e. the prefactor and 
    the factor inside the exponential).
    """
    objectives = np.zeros(len(pos))
    for i in range(len(pos)):
        # increase weights of points that are closer to center
        gaussian = lambda x: np.exp(-x/(num/10))
        weights_along_ray = gaussian(np.linspace(0,10,num=20))
        objectives_per_ray = np.average(weights_of_rays(pos, field, phi, pos[i,0], pos[i,1])**2, 
                                            axis=1, weights=weights_along_ray)
        objectives[i] = np.sum(objectives_per_ray)

    index_min = np.argmin(objectives)

    return pos[index_min, :2]

def find_center_of_rays_all(pos, field, phi=30, length=20, num=20):
    """
    For each point in pos, estimate the weight using weights_of_rays of this point as center. 
    The weights are returned as ndarray of shape (3, num), average over the squares of the second axis 
    as a weighted sum, where the latter weights are set originate from a Gaussian function.
    Return the point with the smalles weight. 

    Note that the result is heavily dependent on the gaussian function used (i.e. the prefactor and 
    the factor inside the exponential).
    """
    objectives = np.zeros(len(pos))
    for i in range(len(pos)):
        # increase weights of points that are closer to center
    
        objectives[i] = np.sum(distance_to_lines(pos, field, phi, pos[i,0], pos[i,1])**2)

    index_min = np.argmin(objectives)

    return pos[index_min, :2]