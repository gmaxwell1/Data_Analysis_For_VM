import numpy as np
import pandas as pd
import tricubic

from modules.analysis_tools import *



n = 3
Bx = np.zeros((n, n, n), dtype='float')

# currents, B_measured, B_expected = collectAndExtract(r'data_sets\first_dataset_for_tricubic_20_12_17', 0, remove_saturation=False)
data = pd.read_csv(r'data_sets\first_dataset_for_tricubic_20_12_17\20_12_18_14-33-02_field_meas.csv').to_numpy()
currents = data[:, 0:3]
B_measured = data[:, 3:6]
B_expected = data[:, 9:]

print(len(currents))

for n in range(len(currents)):
    print(B_measured[n,:])
        

# for i in range(n):
#     for j in range(n):
#         for k in range(n):
#             # some function f(x,y,z) is given on a cubic grid indexed by i,j,k
#             f[i][j][k] = i+j+k

# # initialize interpolator with input data on cubic grid
# ip = tricubic.tricubic(list(f), [n, n, n])
# for i in range(100):
#     # interpolate the function f at a random point in space
#     res = ip.ip(list(np.random.rand(3)*(n-1)))
#     print(res)
