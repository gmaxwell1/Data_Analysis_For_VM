#%%
# standard library imports
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import curve_fit, least_squares
from scipy.special import binom
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model 
from sklearn.svm import SVR
from time import time
import pickle

# local imports
try:
    from modules.analysis_tools import *
except ModuleNotFoundError:
    import sys
    sys.path.insert(1, os.path.join(sys.path[0], '..'))
finally:
    from modules.analysis_tools import *
    from modules.interpolation_tools import *
    from modules.fitting_tools import *
    from modules.data_management import extract_raw_data_from_file



#%%
# read in measurement data from suitable training set
data_directory = './training_set_measurement_21_01_24'
filename = '21_01_24_09-35-21_field_meas.csv'
data_filepath = os.path.join(data_directory, filename)
currents, fields, _, _ = extract_raw_data_from_file(data_filepath)

print(f'number of training samples: {len(currents)}')

#  ---------------------------------------------------------------------------------
# %%
# generate a model of polynomial features, remove constant term of polynomial using include_bias
poly = PolynomialFeatures(degree = 3, include_bias=False)

train_inputs = fields
train_outputs = currents *1000
test_inputs = fields
test_outputs = currents *1000

# transform the x data for proper fitting (for single variable type it returns,[1,x,x**2])
train_inputs_ = poly.fit_transform(train_inputs)
print(len(train_inputs_[0]))

# generate the regression object
model = linear_model.LinearRegression()

# preform the actual regression
model.fit(train_inputs_, train_outputs)

# estimate performance on training data to check consistency
test_vectors_ = poly.fit_transform(test_inputs) 
predictions_new_sweep = model.predict(test_vectors_)
RMSE_new_sweep = estimate_RMS_error(test_outputs.flatten(), predictions_new_sweep.flatten())
print(f'rms error on training set: {RMSE_new_sweep*1000:.3f} mA')

print(model.predict(poly.fit_transform(np.array([[50,0,0]]) )))
# save the model 
model_name = 'model_poly3_final'
filename = f'./{model_name}_B2I.sav'
# pickle.dump([model, poly], open(filename, 'wb'))


# inverse mapping
poly = PolynomialFeatures(degree = 3, include_bias=False)
train_outputs_ = poly.fit_transform(train_outputs)
model = linear_model.LinearRegression()
model.fit(train_outputs_, train_inputs)

# estimate performance on training data to check consistency
test_vectors_ = poly.fit_transform(test_outputs) 
predictions_new_sweep = model.predict(test_vectors_)
RMSE_new_sweep = estimate_RMS_error(train_inputs.flatten(), predictions_new_sweep.flatten())
print(f'rms error on training set: {RMSE_new_sweep:.3f} mT')

# save inverse model
filename = f'./{model_name}_I2B.sav'
# pickle.dump([model, poly], open(filename, 'wb'))


#%%
# generate histogram for magnitudes of vectors in training data
fig, axs = plt.subplots(3, figsize = (6,6))
mask = np.linalg.norm(fields, axis=1) !=0 # omit vectors with zero norm due to reasonability

axs[0].hist(get_theta(fields)[mask], 30)
axs[0].set_xlabel('polar angle, $\\theta$ [°]')
axs[2].set_ylabel('Counts')

axs[1].hist(get_phi(fields)[mask], 30)
axs[1].set_xlabel('azimuthal angle, $\\phi$ [°]')
axs[2].set_ylabel('Counts')

axs[2].hist(np.linalg.norm(fields, axis=1), 30)
axs[2].set_xlabel('measured B field, $B_{meas}$ [mT]')
axs[2].set_ylabel('Counts')

[axs[i].set_ylim(bottom=0) for i in range(len(axs))]

plt.tight_layout()

image_file_name = f'training_set_size{len(fields)}_composition.png'
image_path = os.path.join(data_directory, image_file_name)
# fig.savefig(image_path, dpi=300)
plt.show()
# %%
