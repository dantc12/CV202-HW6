import numpy as np
import time
from copy import copy


def calc_neighbors_sum(mat, i, j):
    return mat[i - 1][j] + mat[i][j - 1] + mat[i][j + 1] + mat[i + 1][j]


# returns prob for (x_s=+1, x_s=-1)
def calc_prob(mat, Temp, i, j):
    tmp_p_x_plus1 = np.exp((1/Temp) * calc_neighbors_sum(mat, i, j) * 1)
    tmp_p_x_minus1 = np.exp((1/Temp) * calc_neighbors_sum(mat, i, j) * (-1))
    p_x_plus1 = tmp_p_x_plus1 / (tmp_p_x_plus1 + tmp_p_x_minus1)
    p_x_minus1 = tmp_p_x_minus1 / (tmp_p_x_plus1 + tmp_p_x_minus1)
    return p_x_plus1, p_x_minus1


def sample_site(mat, Temp, i, j):
    p_x = calc_prob(mat, Temp, i, j)
    return np.random.choice([1, -1], 1, p=[p_x[0], p_x[1]])


def MRF_iteration(padded_mat, Temp, lat_size):
    prev_padded_mat = copy(padded_mat)
    for i in range(lat_size):
        for j in range(lat_size):
            padded_mat[i+1][j+1] = sample_site(prev_padded_mat, Temp, i+1, j+1)  # Assuming padded
    return padded_mat


def Gibbs_sampler(Temp, iterations, lat_size):
    s = time.time()
    initial_mat = np.random.randint(low=0, high=2, size=(lat_size, lat_size))*2 - 1
    padded_mat = np.pad(initial_mat, ((1, 1), (1, 1)), 'constant')
    for iteration in range(iterations):
        padded_mat = MRF_iteration(padded_mat, Temp, lat_size)
    e = time.time()
#     print("Sampling one sample took " + str(round(e-s, 2)) + " seconds.")
    return padded_mat[1:-1, 1:-1]
