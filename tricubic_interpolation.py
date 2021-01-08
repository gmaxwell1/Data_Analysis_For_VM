import numpy as np
import pandas as pd
import tricubic
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

from modules.analysis_tools import *


def interpolateOnArbitraryGrid(input_vector, N=3, gridOrder=1):
    """
    On a NxNxN grid where each 'point' is indexed by 3 integers (i,j,k), the measurement values
    (which are 3 dimensional w/ components (f1,f2,f3)) are each associated with one grid point. Between any two grid points,
    tricubic interpolation determines the behaviour of the function.

    Args:
        input_vector (ndarray): a vector containing any number of scalar measurements.
        N (int, optional): Size of grid. Defaults to 3.
        gridOrder (int, optional): Order in which the grid is set up. default means we start at the bottom left
                                   corner and increase the third component(vertical), then the second and finally
                                   the first.
        
    Returns:
        3 Interpolator objects: ip_1, ip_2, ip_3
        and 3 ndarrays f1, f2, f3
    """

    
    f1 = np.zeros((N, N, N), dtype='float')
    f2 = np.zeros((N, N, N), dtype='float')
    f3 = np.zeros((N, N, N), dtype='float')

    gridSize = N**3 - 1
    m = 0
    for i in range(N):
        for j in range(N):
            for k in range(N):
                # some function f(x,y,z) is given on a cubic grid indexed by i,j,k
                if gridOrder == -1:
                    f1[i][j][k] = input_vector[gridSize-m, 0]
                    f2[i][j][k] = input_vector[gridSize-m, 1]
                    f3[i][j][k] = input_vector[gridSize-m, 2]
                elif gridOrder == 1:
                    f1[i][j][k] = input_vector[m, 0]
                    f2[i][j][k] = input_vector[m, 1]
                    f3[i][j][k] = input_vector[m, 2]
                m = m+1

    # initialize interpolator with input data on cubic grid
    ip_1 = tricubic.tricubic(list(f1), [N, N, N])
    ip_2 = tricubic.tricubic(list(f2), [N, N, N])
    ip_3 = tricubic.tricubic(list(f3), [N, N, N])

    return ip_1, ip_2, ip_3, f1, f2, f3


def plot1Bvs1Icomponent(ip_Bx: tricubic.tricubic, coil_number, measured_vals, I=np.array([-1, 0, 1]), finesse=100, N=3, color='C2',
                        xlabel='current in coil 1, $I_1$ [A]', ylabel='$B_x$ [mT]'):
    
    points = np.linspace(0,N-1,finesse)
    fac = (I[-1] - I[0]) / (N-1)
    i_vals = points*fac + I[0]
    B_x_eval = np.zeros(points.shape)

    for l in range(len(points)):
        eval_point = [(N-1)/2] * 3
        eval_point[coil_number-1] = points[l]
        B_x_eval[l] = ip_Bx.ip(eval_point)

    fig, ax = plt.subplots(1, 1, sharex=True)
    fig.set_size_inches(6, 6)

    ax.plot(i_vals, B_x_eval, color='C2', label='interpolted curve from measurements')

    ax.plot(I, measured_vals, color='C3', linestyle='', marker='.', label='measurements')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.legend()
    ax.yaxis.grid()
    
    plt.tight_layout()
    plt.show()

        


if __name__ == "__main__":
    n = 3

    fileName = r'data_sets\first_dataset_for_tricubic_20_12_17\21_01_08_08-20-13_grid_max30_PointsPerDim13_demag5A_reordered_zyx.csv'
    data = pd.read_csv(fileName).to_numpy()
    currents = data[:, 0:3]
    B_measured = data[:, 3:6]
    print(B_measured[:169])

    # print(len(currents))
    ip_I1, ip_I2, ip_I3, I1, I2, I3 = interpolateOnArbitraryGrid(currents, N=13, gridOrder=1)
    
    I_act = np.array([I1[0][6][6], I1[1][6][6], I1[2][6][6], I1[3][6][6], I1[4][6][6], I1[5][6][6], 
                      I1[6][6][6], I1[7][6][6], I1[8][6][6], I1[9][6][6], I1[10][6][6], I1[11][6][6], I1[12][6][6]])
    plot1Bvs1Icomponent(ip_I1, 1, I_act, I=np.array([-30,-25,-20,-15,-10,-5,0,5,10,15,20,25,30]), N=13,xlabel='$B_x$ [mT]', ylabel='current in coil 1, $I_1$ [A]')
    I_act = np.array([I1[6][0][6], I1[6][1][6], I1[6][2][6], I1[6][3][6], I1[6][4][6], I1[6][5][6], 
                      I1[6][6][6], I1[6][7][6], I1[6][8][6], I1[6][9][6], I1[6][10][6], I1[6][11][6], I1[6][12][6]])
    plot1Bvs1Icomponent(ip_I1, 2, I_act, I=np.array([-30,-25,-20,-15,-10,-5,0,5,10,15,20,25,30]), N=13,xlabel='$B_y$ [mT]', ylabel='current in coil 1, $I_1$ [A]')
    I_act = np.array([I1[6][6][0], I1[6][6][1], I1[6][6][2], I1[6][6][3], I1[6][6][4], I1[6][6][5], 
                      I1[6][6][6], I1[6][6][7], I1[6][6][8], I1[6][6][9], I1[6][6][10], I1[6][6][11], I1[6][6][12]])
    plot1Bvs1Icomponent(ip_I1, 3, I_act, I=np.array([-30,-25,-20,-15,-10,-5,0,5,10,15,20,25,30]), N=13,xlabel='$B_z$ [mT]', ylabel='current in coil 1, $I_1$ [A]')
        
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')

    # # Make data.
    # X = np.arange(-1, 1, 0.05)
    # Y = np.arange(-1, 1, 0.05)
    
    # B_x_eval = np.zeros((len(X),len(Y)))

    # for l in range(len(Y)):
    #     for p in range(len(X)):
    #         B_x_eval[l][p] = ip_Bz.ip([1,1+X[p],1+Y[l]])
    #         # B_y_eval[l] = ip_By.ip([0,0,points[l]])
    #         # B_z_eval[l] = ip_Bz.ip([0,0,points[l]])
            
    # X, Y = np.meshgrid(X, Y)
    # # Plot the surface.
    # surf = ax.plot_surface(X, Y, B_x_eval, cmap=cm.jet,
    #                     linewidth=0, antialiased=False)
    
    # ax.set_xlabel('$I_2$ [A]')
    # ax.set_ylabel('$I_3$ [A]')
    # ax.set_zlabel('$B_z$ [mT]')


    # # Customize the z axis.
    # # ax.set_zlim(-1.01, 1.01)
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # # Add a color bar which maps values to colors.
    # fig.colorbar(surf, shrink=0.5, aspect=5)

    # plt.show()



    