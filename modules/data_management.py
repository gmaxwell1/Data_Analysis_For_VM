#%%
# standard library imports
import numpy as np
import pandas as pd
import os
from datetime import datetime

# local imports



#%%
def extract_raw_data_from_file(filepath):
    """
    Extract current, mean and std values of measured magnetic field and expected magnetic field from file.

    Also checks whether the current is provided as single float value or as vector
    """
    # extract data and convert to ndarray
    raw_data = pd.read_csv(filepath).to_numpy()

    # current can be single-valued or a vector, differentiate these cases
    dimension_I = len(raw_data[0]) - 9
    if dimension_I == 1:
        I = raw_data[:, 0]
        mean_data_specific_sensor = raw_data[:, 1:4]
        std_data_specific_sensor = raw_data[:, 4:7]
        expected_fields = raw_data[:, 7:10]
    else:
        I = raw_data[:, 0:3]
        mean_data_specific_sensor = raw_data[:, 3:6]
        std_data_specific_sensor = raw_data[:, 6:9]
        expected_fields = raw_data[:, 9:12]

    return I, mean_data_specific_sensor, std_data_specific_sensor, expected_fields

def extract_raw_data_from_2d_scan(filepath):
    """
    Extract positions and measured magnetic fields from file that was created after 2d-scan.

    Returns: positions, B_field (both are arrays of shape (#points, 3))
    """
    # extract data and convert to ndarray
    raw_data = pd.read_csv(filepath).to_numpy()

    # current can be single-valued or a vector, differentiate these cases
    dimension_I = len(raw_data[0]) - 9
    
    positions = raw_data[:, 0:3]
    B_field = raw_data[:, 3:6]

    return positions, B_field


def collectAndExtract(directory, B_min, remove_saturation = True,
                        verbose=False, fraction_cutoff = 0.02,
                        affine_fct = lambda x, a, b: a*x + b):
    """
    Extract data from a measurement series, where all measurement runs are stored in 
    various files in a directory. All data below the threshold B_min are ignored.
    If desired, data above saturation are ignored, too.

    Args:
    - directory (str): valid path of directory that contains the data
    - B_min (float): threshold value that sets the minimum considered magnetic field magnitude 
    - remove_saturation (bool): if True, data above saturation are removed as well. 
    For estimating the boundaries of the linear regime and where saturation starts the function
    find_start_of_saturation is used, which takes the remaining optional arguments fraction_cutoff
    and affine_fct. This algorithm is not perfect and requires some fine-tuning, but it seems 
    to work qualitatively well, so the estimated boundaries are not far from what one would 
    pick as start of saturation. 
    - fraction_cutoff (float): paramter passed to find_start_of_saturation 
    - affine_fct (function): paramter passed to find_start_of_saturation 

    Return:
    - currents, B_measured, B_expected (ndarrays of shape (N, 3)): All currents, measured and expected
    fields extracted from the directory that are above the threshold magnitude (and below saturation
    if desired). 
    """
    # import function here rather than at beginning of script to omit circular import
    from modules.interpolation_tools import find_start_of_saturation

    # collect all csv-files in this directory
    filenames = []
    [filenames.append(file) for file in os.listdir(directory) if file.endswith(".csv")]
    
    # if in list, remove the linear_fits file from previous fits
    try:
        filenames.remove('linear_fits.csv')
    except ValueError:
        pass
    print(f'files considered: {len(filenames)}')

    # loop through all csv files in a dictionary and fit the data
    for i in range(len(filenames)):
        if verbose:
            print(filenames[i])

        # read in raw measurment data
        data_filepath = os.path.join(directory, filenames[i])
        I, mean_data, std_data, expected_fields = extract_raw_data_from_file(data_filepath)

        # -> this could be used to exclude data above saturation, but could be left out
        # estimate minimum and maximum indices of region within which the linear relation between current and field holds  
        # even though find_start_of_saturation offers the possibility to specify the considered component, 
        # keep the default stting, which detects the field component that has the greatest absolute field values. 
        # This should work fine for situations, where one component is dominating. 
        i_min, i_max = find_start_of_saturation(I, mean_data, std_data, fraction_cutoff=fraction_cutoff,
                                fitting_fct = affine_fct)

        # estimate field magnitudes
        magnitudes = np.linalg.norm(mean_data, axis=1)

        # set up mask to only keep data with magnitudes larger than B_min
        mask_keep = magnitudes >= B_min

        # optionally: also remove potentially saturated part
        if remove_saturation:
            mask_keep[:i_min+1] = False
            mask_keep[i_max:] = False

        # collect all relevant data
        if i == 0:
            B_measured = mean_data[mask_keep]
            B_expected = expected_fields[mask_keep]
            currents = I[mask_keep]
        else:
            B_measured = np.append(B_measured, mean_data[mask_keep], axis=0)
            B_expected = np.append(B_expected, expected_fields[mask_keep], axis=0)
            currents = np.append(currents, I[mask_keep], axis=0)

    print(f'final shape of considered array: {B_measured.shape}')
    return currents, B_measured, B_expected

def save_in_dir(means, directory, label, stds=None, coords=False, now=False):
    """
    Write the provided means and (optional) standard deviations to the file 'means_'+label+'.csv' 
    or 'yyy_mm_dd_hh-mm-ss_means_'+label+'.csv' in provided directory.

    The coords flag distinguishes between magnetic field strength (False) and spatial coordinates (True) as source of the data.

    Args:
    - means (array or list): measured mean values of B-field or coordinates 
    - directory (string): valid path of the directory in which the file should be stored
    - label (string or float): used to label the csv file  
    - stds (array or list): measured standard deviations of B-field or coordinates. 
    Should have at least the same size as means.
    - coords (bool): Flag to switch between B-field (False) and spatial coordinates (True)
    - verbose (bool): switching on/off print-statements for displaying progress
    - now (bool): if True, the current date time is added to the filename, 
    such that it reads 'yyy_mm_dd_hh-mm-ss_means_'+label+'.csv'
    """
    # Under Linux, user rights can be set with the scheme below,
    # where 755 means read+write+execute for owner and read+execute for group and others.
    # However, note that you can only set the file’s read-only flag with it under Windows.
    access_rights = 0o755
    ensure_dir_exists(directory, access_rights=access_rights)
    os.chmod(directory, access_rights)

    if now:
        time_stamp = datetime.now().strftime("%y_%m_%d_%H-%M-%S") 
        output_file_name = "{}_means_{}.csv".format(time_stamp, label)
    else:
        output_file_name = "means_{}.csv".format(label)
    data_filepath = os.path.join(directory, output_file_name)

    if stds is not None and not coords:
        df = pd.DataFrame({ 'mean Bx [mT]':  means[:, 0], 
                            'std Bx [mT] ': stds[:, 0], 
                            'mean By [mT]':  means[:, 1], 
                            'std By [mT] ': stds[:, 1],
                            'mean Bz [mT]':  means[:, 2], 
                            'std Bz [mT] ': stds[:, 2]})
        df.to_csv(data_filepath, index=False, header=True)
    
    elif not coords:
        df = pd.DataFrame({ 'mean Bx [mT]':  means[:, 0], 
                            'mean By [mT]':  means[:, 1], 
                            'mean Bz [mT]':  means[:, 2]})
        df.to_csv(data_filepath, index=False, header=True)

    elif coords and stds is None:
        df = pd.DataFrame({ 'Index':  np.arange(len(means[:, 0])) + 1, 
                            'x [mm]':  means[:, 0], 
                            'y [mm]':  means[:, 1], 
                            'z [mm]':  means[:, 2]})
        df.to_csv(data_filepath, index=False, header=True)
        
    elif not coords:
        df = pd.DataFrame({ 'Index':  np.arange(len(means[:, 0])) + 1, 
                            'x [mm]':  means[:, 0], 
                            'std x [mm]':  stds[:, 0], 
                            'y [mm]':  means[:, 1], 
                            'std y [mm]':  stds[:, 1], 
                            'z [mm]':  means[:, 2],
                            'std z [mm]':  stds[:, 2]})
        df.to_csv(data_filepath, index=False, header=True)


def ensure_dir_exists(directory, access_rights=0o755, purpose_text='', verbose=False):
    """
    Ensure that the directory exists and create the respective folders if it doesn't.

    Prints message that either a folder has been created or that it already exists. 
    Note that only read and write options can be set under Windows, rest ignored.

    Args:
    - directory (string) is a path which should exist after calling this function
    - access_rights: set rights for reading and writing.
    - purpose_text (str): add more information what the dictionary was created for

    Return:
    - 0 if directory needed to be created
    - 1 if directory already exists
    - -1 if there was an exception
    """
    try:
        os.mkdir(directory, access_rights)
        os.chmod(directory, access_rights)
        if verbose:
            print('Created directory {}: {}'.format(
                purpose_text, os.path.split(directory)[1]))
        return 0
    except FileExistsError:
        if verbose:
            print('Folder already exists, no new folder created.')
        return 1
    except Exception as e:
        # if verbose:
        # this is important and should always be printed - gmaxwell, 8.10.2020
        print('Failed to create new directory due to {} error'.format(e))
        return -1

def save_time_resolved_measurement(results_dict, directory, label='time_resolved', now=False):
    """
    Write the provided times and field (optionally also standard deviations) to the file 'time_resolved.csv' 
    or 'yyy_mm_dd_hh-mm-ss_means_'+label+'.csv' in provided directory.

    Args:
    - results_dict (dictionary): Contains 'time', 'Bx', 'By' and 'Bz' as keys, which are 1dimensional lists containing 
    the times of measurements and the measured mean values of B-field
    - directory (string): valid path of the directory in which the file should be stored
    - now (bool): if True, the current date time is added to the filename, 
    such that it reads 'yyy_mm_dd_hh-mm-ss_means_'+label+'.csv'
    """
    if now:
        time_stamp = datetime.now().strftime("%y_%m_%d_%H-%M-%S") 
        output_file_name = "{}_{}.csv".format(time_stamp, label)
    else:
        output_file_name = label + '.csv'     
    ensure_dir_exists(directory)
    data_filepath = os.path.join(directory, output_file_name)
    
  
    df = pd.DataFrame({ 'time [s]': results_dict['time'], 
                        'Bx [mT]':  results_dict['Bx'], 
                        'By [mT]':  results_dict['By'], 
                        'Bz [mT]':  results_dict['Bz'],
                        'Temperature':  results_dict['temp']})
        
    df.to_csv(data_filepath, index=False, header=True)


