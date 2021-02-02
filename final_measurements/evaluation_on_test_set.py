#%%
# standard library imports
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# local imports
try:
    from modules.analysis_tools import *
except ModuleNotFoundError:
    import sys
    sys.path.insert(1, os.path.join(sys.path[0], '..'))
finally:
    from modules.analysis_tools import *
    from modules.data_management import extract_raw_data_from_file
    from modules.interpolation_tools import *
    from modules.fitting_tools import *


#%%
# read in measurement results
data_directory = './random_test_21_01_25'
data_filename = '21_01_25_16-16-46_field_meas.csv'
data_filepath = os.path.join(data_directory, data_filename)

currents, B_measured, B_measured_std, B_expected = extract_raw_data_from_file(data_filepath)
test_vectors = B_expected

# evaluate the performance
print('\nperformance of currently implemented model')
evaluate_performance(B_measured, B_expected)

#%%
# load a model from file to compare to set currents
filename_model = './model_poly3_final_B2I.sav'
[loaded_model, loaded_poly] = pickle.load(open(filename_model, 'rb'))


# estimate predicted currents and compare to currents set during experiment
test_vectors = B_expected
test_vectors_ = loaded_poly.fit_transform(test_vectors) 
predicted_currents = loaded_model.predict(test_vectors_)

# set up a mask to only select valid currents
mask_valid = np.all(np.abs(currents - predicted_currents) <= 0.2, axis=1)
print(f'There are {np.sum(mask_valid)} valid and {len(currents) - np.sum(mask_valid)} invalid events')

# plot deviations of predicted and actually set currents 
fig, axs = plt.subplots(3, sharex=True, figsize = (6,8))
coil_labels = np.arange(1,4)
for i in range(3):
    axs[i].plot(currents[mask_valid, i]-predicted_currents[mask_valid, i], marker='.',linestyle='')
    axs[i].set_ylabel(f'deviation $\\Delta I_{coil_labels[i]}$ [A]')
image_file_name = f'{os.path.splitext(data_filename)[0]}_currents_corrected.png'
image_path = os.path.join(data_directory, image_file_name)
# fig.savefig(image_path, dpi=300)
plt.show()



# %%
# plot angular errors as histogram for all data and for the data >= 30 mT
dot = np.array([np.dot(B_measured[i], test_vectors[i]) for i in range(len(B_measured))])
norms_measured = np.linalg.norm(B_measured, axis=1)
norms_test = np.linalg.norm(test_vectors, axis=1)
mask = (norms_measured!=0) * (norms_test!=0) # omit vectors with zero norm due to reasonability
alphas = np.degrees(np.arccos(dot[mask] / (norms_measured[mask] * norms_test[mask])))[mask_valid]

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

image_file_name = f'{os.path.splitext(data_filename)[0]}_performance_histogram_corrected.png'
image_path = os.path.join(data_directory, image_file_name)
fig.savefig(image_path, dpi=300)
plt.show()


#%%
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

axs[0].plot(get_theta(B_measured)[mask][mask_valid], alphas, marker='.', linestyle='')
axs[0].set_xlabel('polar angle, $\\theta$ [°]')
axs[0].set_ylabel('Angular Error, $\\alpha$ [°]')

axs[1].plot(get_phi(B_measured)[mask][mask_valid], alphas, marker='.', linestyle='')
axs[1].set_xlabel('azimuthal angle, $\\phi$ [°]')
axs[1].set_ylabel('Angular Error, $\\alpha$ [°]')

axs[2].plot(np.linalg.norm(B_measured, axis=1)[mask][mask_valid], alphas, marker='.', linestyle='')
axs[2].set_xlabel('measured B field, $B_{meas}$ [mT]')
axs[2].set_ylabel('Angular Error, $\\alpha$ [°]')

[axs[i].set_ylim(bottom=0) for i in range(len(axs))]

plt.tight_layout()

image_file_name = f'{os.path.splitext(data_filename)[0]}_performance_corrected.png'
image_path = os.path.join(data_directory, image_file_name)
fig.savefig(image_path, dpi=300)
plt.show()


