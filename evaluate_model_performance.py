"""
This script can be used to analyze the performance of a model on test sets. 
It is structured in 3 parts:
(1): Evaluation on a general test set, use this for random test sets
(2): Evaluation of performance on sweeps along main axes, 
    including appropriate plots
(3): Evaluation of performance on rotations about main axes, 
    including appropriate plots 
"""


#%%
# standard library imports
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model 
from sklearn.svm import SVR
import pickle

# local imports
from modules.analysis_tools import *
from modules.data_management import extract_raw_data_from_file
from modules.analysis_tools import *
from modules.interpolation_tools import *
from modules.fitting_tools import *


"""
# (1) --------------------------------------------------------------------
# Performance on general test sets
"""
#%%
# read in measurement results
# data_directory = './test_data/B_field_on_grid'
data_directory = './final_tests/new_characterization_meas_21_01_22'
data_filename = '21_01_22_17-32-42_field_meas.csv'
data_filepath = os.path.join(data_directory, data_filename)

I, B_measured, B_measured_std, expected = extract_raw_data_from_file(data_filepath)

# read in test set
# filename_testset = 'vectors_rng2222_0-50mT_size2000'
# filepath_testset = f'./predictions_on_test_set/{filename_testset}.csv'
filename_testset = 'vectors_rng987_0-60mT_size25'
filepath_testset = f'./test_sets/{filename_testset}.csv'
# test_vectors = read_test_set(filepath_testset)
test_vectors = expected


# evaluate the performance
print('performance of cubic fit')
evaluate_performance(B_measured, test_vectors)

print('\nperformance of currently implemented model')
evaluate_performance(B_measured, expected)


# %%
# plot angular errors as histogram for all data and for the data >= 30 mT
dot = np.array([np.dot(B_measured[i], test_vectors[i]) for i in range(len(B_measured))])
norms_measured = np.linalg.norm(B_measured, axis=1)
norms_test = np.linalg.norm(test_vectors, axis=1)
mask = (norms_measured!=0) * (norms_test!=0) # omit vectors with zero norm due to reasonability
alphas = np.degrees(np.arccos(dot[mask] / (norms_measured[mask] * norms_test[mask])))

# generate histogram for all data
fig, ax = plt.subplots()
n, bins, patches = ax.hist(alphas, 50)
ax.set_xlabel('Angular Error, $\\alpha$ [°]')
ax.set_ylabel('Counts')
ax.text(0.7, 0.55, f'$\\mu$ = {np.mean(alphas):.2f}°,\n'+ 
                f'$\\sigma$ = {np.std(alphas):.2f}°,\n'+ 
                f'median = {np.median(alphas):.2f}°\n'+
                f'RMS $\\alpha$ = {estimate_RMS_error(alphas, np.zeros_like(alphas)):.2f}°\n'+
                f'min = {np.min(alphas):.2f}°,\n'+ 
                f'max = {np.max(alphas):.2f}°,\n', transform=ax.transAxes)

image_file_name = f'{os.path.splitext(data_filename)[0]}_performance_histogram.png'
image_path = os.path.join(data_directory, image_file_name)
fig.savefig(image_path, dpi=300)
plt.show()


# only consider data >= 30 mT
mask_mag = norms_measured[mask] >= 30

fig, ax = plt.subplots()
n, bins, patches = ax.hist(alphas[mask_mag], 50)
ax.set_xlabel('Angular Error for |B| $\\geq$ 30 mT, $\\alpha$ [°]')
ax.set_ylabel('Counts')
ax.text(0.7, 0.55, f'$\\mu$ = {np.mean(alphas[mask_mag]):.2f}°,\n'+ 
                f'$\\sigma$ = {np.std(alphas[mask_mag]):.2f}°,\n'+ 
                f'median = {np.median(alphas[mask_mag]):.2f}°\n'+
                f'RMS $\\alpha$ = {estimate_RMS_error(alphas[mask_mag], np.zeros_like(alphas[mask_mag])):.2f}°\n'+
                f'min = {np.min(alphas[mask_mag]):.2f}°,\n'+ 
                f'max = {np.max(alphas[mask_mag]):.2f}°,\n', transform=ax.transAxes)

image_file_name = f'{os.path.splitext(data_filename)[0]}_performance_histogram_30-50mT.png'
image_path = os.path.join(data_directory, image_file_name)
# fig.savefig(image_path, dpi=300)
plt.show()

#%%
# plot angular error vs polar angle, azimuthal angle and field magnitude
fig, axs = plt.subplots(3, figsize = (6,8))

axs[0].plot(get_theta(B_measured)[mask], alphas, marker='.', linestyle='')
axs[0].set_xlabel('polar angle, $\\theta$ [°]')
axs[0].set_ylabel('Angular Error, $\\alpha$ [°]')

axs[1].plot(get_phi(B_measured)[mask], alphas, marker='.', linestyle='')
axs[1].set_xlabel('azimuthal angle, $\\phi$ [°]')
axs[1].set_ylabel('Angular Error, $\\alpha$ [°]')

axs[2].plot(np.linalg.norm(B_measured, axis=1)[mask], alphas, marker='.', linestyle='')
axs[2].set_xlabel('measured B field, $B_{meas}$ [mT]')
axs[2].set_ylabel('Angular Error, $\\alpha$ [°]')

[axs[i].set_ylim(bottom=0) for i in range(len(axs))]

plt.tight_layout()

image_file_name = f'{os.path.splitext(data_filename)[0]}_performance.png'
image_path = os.path.join(data_directory, image_file_name)
fig.savefig(image_path, dpi=300)
plt.show()




"""
# (2) --------------------------------------------------------------------
# Performance on sweeps along main axes 
"""
#%%
# read in measurement data 
# data_directory = './test_data/xyzaxis_sweeps_21_01_08'
data_directory = './final_measurements/sweep_along_axes_21_01_24'
data_filename = '21_01_24_20-28-04_along_z.csv'
data_filepath = os.path.join(data_directory, data_filename)
_, B_measured, B_measured_std, B_expected = extract_raw_data_from_file(data_filepath)

# evaluate performance along main axes
print('\nperformance of currently implemented model')
evaluate_performance(B_measured, B_expected)


# estimate the axis along which sweep is performed
sweep_axis = int(np.nonzero(B_expected[0])[0])
print('sweep along {} axis'.format(sweep_axis))

# define which components should be plotted - only plot combinations containing sweep_axis!
components = np.array([[sweep_axis, (sweep_axis+1) % 3], [sweep_axis, (sweep_axis+2) % 3]])

# generate a plot of sweeps
fig, axs = plt.subplots(ncols=2)
fig.set_size_inches(8, 3)

for i in range(2):
    axs[i].errorbar(B_measured[:, components[i, 0]], B_measured[:, components[i, 1]], 
                            xerr = B_measured_std[:, components[i, 0]], 
                            yerr = B_measured_std[:, components[i, 1]],
                            linestyle='', marker='.', capsize = 2, 
                            label = 'measured')
    axs[i].plot(B_expected[:, components[i, 0]], B_expected[:, components[i, 1]], 
                            linestyle='--', label = 'desired')

    # set axis labels
    labels_components = ['$B_x$ [mT]', '$B_y$ [mT]', '$B_z$ [mT]']
    axs[i].set_xlabel(labels_components[components[i, 0]])
    axs[i].set_ylabel(labels_components[components[i, 1]])

# equalize the ranges
xmin = np.min([axs[i].get_xlim()[0] for i in range(len(axs))])
xmax = np.max([axs[i].get_xlim()[1] for i in range(len(axs))])
ymin = np.min([axs[i].get_ylim()[0] for i in range(len(axs))])
ymax = np.max([axs[i].get_ylim()[1] for i in range(len(axs))])
# for i in range(len(axs)):
#     axs[i].set_xlim(xmin, xmax)
#     axs[i].set_ylim(ymin, ymax)
axs[0].legend()

plt.tight_layout()

sweep_ax_label = ['x', 'y', 'z'][sweep_axis]
image_file_name = f'along_mainAxis_{sweep_ax_label}.png'
image_path = os.path.join(data_directory, image_file_name)
fig.savefig(image_path, dpi=300)
plt.show()


#%%
# estimate angular errors and generate plot of precision
dot = np.array([np.dot(B_measured[i], B_expected[i]) for i in range(len(B_measured))])
norms_measured = np.linalg.norm(B_measured, axis=1)
norms_test = np.linalg.norm(B_expected, axis=1)
alphas = np.degrees(np.arccos(dot / (norms_measured * norms_test)))

# plot angular errors versus desired field strength along main axis 
fig, ax = plt.subplots()
ax.plot(B_expected[:, sweep_axis], alphas, linestyle='', marker='.')

# add text containing statistics
plt.text(0.05, 0.6, '$\\langle \\alpha \\rangle$ = {:.2f}°,\n'.format(np.mean(alphas))+ 
                '$\\sigma (\\alpha)$ = {:.2f}°,\n'.format(np.std(alphas))+ 
                'min ($\\alpha$) = {:.2f}°,\n'.format(np.min(alphas))+ 
                'max ($\\alpha$) = {:.2f}°,\n'.format(np.max(alphas))+ 
                'median ($\\alpha$) = {:.2f}°,\n'.format(np.median(alphas)), transform=ax.transAxes)

# axis settings
labels_components = ['$B_x$ [mT]', '$B_y$ [mT]', '$B_z$ [mT]']
ax.set_xlabel(f'desired field {labels_components[sweep_axis]}')
ax.set_ylabel('Angular Error, $\\alpha$ [°]')
ax.set_ylim(bottom=0, top=17)

plt.tight_layout()

sweep_ax_label = ['x', 'y', 'z'][sweep_axis]
image_file_name = f'along_mainAxis_{sweep_ax_label}_angular_error.png'
image_path = os.path.join(data_directory, image_file_name)
fig.savefig(image_path, dpi=300)

plt.show()

#%%
# generate histogram for all data
fig, ax = plt.subplots()
n, bins, patches = ax.hist(alphas, 50)
ax.set_xlabel('Angular Error, $\\alpha$ [°]')
ax.set_ylabel('Counts')
ax.text(0.7, 0.55, f'$\\mu$ = {np.mean(alphas):.2f}°,\n'+ 
                f'$\\sigma$ = {np.std(alphas):.2f}°,\n'+ 
                f'median = {np.median(alphas):.2f}°\n'+
                f'RMS $\\alpha$ = {estimate_RMS_error(alphas, np.zeros_like(alphas)):.2f}°\n'+
                f'min = {np.min(alphas):.2f}°,\n'+ 
                f'max = {np.max(alphas):.2f}°,\n', transform=ax.transAxes)

image_file_name = f'along_mainAxis_{sweep_ax_label}_axis_histogram.png'
image_path = os.path.join(data_directory, image_file_name)
fig.savefig(image_path, dpi=300)
plt.show()




"""
# (3) --------------------------------------------------------------------
# Performance on rotations about main axes 
"""
#%%
# read in measurement data 
# data_directory = './test_data/rotation_around_axes_21_01_08'
data_directory = './final_measurements/rotations_around_axes_21_01_24'
data_filename = '21_01_24_21-33-07_field_meas.csv'
data_filepath = os.path.join(data_directory, data_filename)
_, B_measured, B_measured_std, B_expected = extract_raw_data_from_file(data_filepath)

# find the rotation axis
if np.all(np.isclose(B_expected[:,0], B_expected[0,0])):
    rot_axis = 0 # 'x'
elif np.all(np.isclose(B_expected[:,1], B_expected[0,1])):
    rot_axis = 1 # 'y'
elif np.all(np.isclose(B_expected[:,2], B_expected[0,2])):
    rot_axis = 2 # 'z'
# rot_axis=0
print('rotation about {} axis'.format(rot_axis))
rot_ax_label = ['x', 'y', 'z'][rot_axis]

# generate a plots of rotations
fig, axs = plt.subplots(ncols=3)
fig.set_size_inches(12, 4)
components = np.array([[0,1], [0,2], [1,2]])

for i in range(3):
    axs[i].errorbar(B_measured[:, components[i, 0]], B_measured[:, components[i, 1]], 
                            xerr = B_measured_std[:, components[i, 0]], 
                            yerr = B_measured_std[:, components[i, 1]],
                            linestyle='', marker='.', capsize = 2, 
                            label = 'measured')
    axs[i].plot(B_expected[:, components[i, 0]], B_expected[:, components[i, 1]], 
                            linestyle='--', label = 'desired')

    # set axis labels
    labels_components = ['$B_x$ [mT]', '$B_y$ [mT]', '$B_z$ [mT]']
    axs[i].set_xlabel(labels_components[components[i, 0]])
    axs[i].set_ylabel(labels_components[components[i, 1]])

    # set aspect ratio to one, such that a circle actually looks round 
    axs[i].set_aspect('equal')
    
# equalize the ranges
xmin = np.min([axs[i].get_xlim()[0] for i in range(3)])
xmax = np.max([axs[i].get_xlim()[1] for i in range(3)])
ymin = np.min([axs[i].get_ylim()[0] for i in range(3)])
ymax = np.max([axs[i].get_ylim()[1] for i in range(3)])
for i in range(3):
    axs[i].set_xlim(xmin, xmax)
    axs[i].set_ylim(ymin, ymax)

axs[0].legend()

plt.tight_layout()

image_file_name = f'rotation_about_{rot_ax_label}_axis_{int(np.linalg.norm(B_expected[0]))}mT.png'
image_path = os.path.join(data_directory, image_file_name)
fig.savefig(image_path, dpi=300)
plt.show()


# evaluate performance of rotations
print('\nperformance of currently implemented model')
evaluate_performance(B_measured, B_expected)

# estimate angular errors and generate plot of precision
dot = np.array([np.dot(B_measured[i], B_expected[i]) for i in range(len(B_measured))])
norms_measured = np.linalg.norm(B_measured, axis=1)
norms_test = np.linalg.norm(B_expected, axis=1)
alphas = np.degrees(np.arccos(dot / (norms_measured * norms_test)))

# plot angular errors versus desired field strength along main axis 
fig, ax = plt.subplots()
rot_angles = np.linspace(0, 360, len(B_expected))
ax.plot(rot_angles, alphas, linestyle='', marker='.')

# add text containing statistics
plt.text(0.05, 0.75, f'$\\langle \\alpha \\rangle \pm$ std  = {np.mean(alphas):.2f}° $\\pm$ {np.std(alphas):.2f}°,\n'+
                f'min / max ($\\alpha$) = {np.min(alphas):.2f}° / {np.max(alphas):.2f}°,\n'+ 
                f'median ($\\alpha$) = {np.median(alphas):.2f}°,\n', transform=ax.transAxes)

# axis settings
ax.set_xlabel('Rotation angle, $\\Phi$ [°]')
ax.set_ylabel('Angular Error, $\\alpha$ [°]')
ax.set_ylim(bottom=0, top =10)

plt.tight_layout()

image_file_name = f'rotation_about_{rot_ax_label}_axis_{int(np.linalg.norm(B_expected[0]))}mT_angular_error.png'
image_path = os.path.join(data_directory, image_file_name)
fig.savefig(image_path, dpi=300)
plt.show()

#%%
# generate histogram for all data
fig, ax = plt.subplots()
n, bins, patches = ax.hist(alphas, 50)
ax.set_xlabel('Angular Error, $\\alpha$ [°]')
ax.set_ylabel('Counts')
ax.text(0.7, 0.55, f'$\\mu$ = {np.mean(alphas):.2f}°,\n'+ 
                f'$\\sigma$ = {np.std(alphas):.2f}°,\n'+ 
                f'median = {np.median(alphas):.2f}°\n'+
                f'RMS $\\alpha$ = {estimate_RMS_error(alphas, np.zeros_like(alphas)):.2f}°\n'+
                f'min = {np.min(alphas):.2f}°,\n'+ 
                f'max = {np.max(alphas):.2f}°,\n', transform=ax.transAxes)

image_file_name = f'rotation_about_{rot_ax_label}_axis_{int(np.linalg.norm(B_expected[0]))}mT_histogram.png'
image_path = os.path.join(data_directory, image_file_name)
fig.savefig(image_path, dpi=300)
plt.show()
# %%
