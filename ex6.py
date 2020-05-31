import numpy as np
from scipy.ndimage.interpolation import shift

def calc_neigbhors_sum(mat, i, j, val):
    return mat[i - 1][j - 1]*val + mat[i - 1][j]*val + mat[i - 1][j + 1]*val \
            + mat[i][j - 1]*val + mat[i][j + 1]*val \
            + mat[i + 1][j - 1]*val + mat[i + 1][j]*val + mat[i + 1][j + 1]*val

# returns prob for +1
def calc_prob(mat, Temp, i,j):
    x_plus = calc_neigbhors_sum(mat, i, j, 1)
    x_minus = calc_neigbhors_sum(mat, i, j, -1)
    Z_temp = np.exp((1/Temp) * x_plus) + np.exp((1/Temp) * x_minus)
    return np.exp((1/Temp) * x_plus)/Z_temp

def sample_site(mat, Temp, i, j):
    prob = calc_prob(mat, Temp, i,j)
    return np.random.choice([1, -1], 1, p=[prob, 1-prob])

def MRF_iteration(mat, Temp, lat_size=8):
    for i in range(lat_size):
        for j in range(lat_size):
            mat[i+1][j+1] = sample_site(mat, Temp, i+1, j+1)
    return mat

def Gibbs_sampler(Temp, iterations, lat_size=8):
    initial_mat = np.random.randint(low=0,high=2,size=(lat_size,lat_size))*2 - 1
    padded_mat = np.pad(initial_mat, ((1,1),(1,1)), 'constant')
    for iteration in range(iterations):
        padded_mat = MRF_iteration(padded_mat, Temp)
    return padded_mat

def compute_empirical_expectation(num_samples, Temp, iterations=25):
    normalizing_factor = num_samples
    sum12 = 0
    sum18 = 0
    for sample_step in range(num_samples):
        print("Step: " + str(sample_step))
        sample = Gibbs_sampler(Temp, iterations)
        sum12 += sample[1][1] * sample[2][2]
        sum18 += sample[1][1] * sample[8][8]
    return sum12 / normalizing_factor, sum18 / normalizing_factor

print("Method 1: Independent Samples")
#print(compute_empirical_expectation(1000, 2))


def Gibbs_sampler_ergodicity(Temp, sweeps=25000, lat_size=8, ignore_first_size=100):
    initial_mat = np.random.randint(low=0,high=2,size=(lat_size,lat_size))*2 - 1
    padded_mat = np.pad(initial_mat, ((1,1),(1,1)), 'constant')
    normalizing_factor = sweeps - ignore_first_size
    sum12 = 0
    sum18 = 0
    for sweep in range(sweeps):
        print('Sweep Number: ' + str(sweep))
        padded_mat = MRF_iteration(padded_mat, Temp)
        if sweep >= ignore_first_size:
            sum12 += padded_mat[1][1] * padded_mat[2][2]
            sum18 += padded_mat[1][1] * padded_mat[8][8]
    return sum12 / normalizing_factor, sum18 / normalizing_factor

print("Method 2: Ergodicity")
#print(Gibbs_sampler_ergodicity(1, sweeps=5000))



# Exercise 2: Image Restoration
def Gibbs_sampler(Temp, iterations, lat_size=8):
    initial_mat = np.random.randint(low=0,high=2,size=(lat_size,lat_size))*2 - 1
    padded_mat = np.pad(initial_mat, ((1,1),(1,1)), 'constant')
    for iteration in range(iterations):
        padded_mat = MRF_iteration(padded_mat, Temp)
    return padded_mat

def Gaussian_noise(lat_size=100):
    return 2*np.random.standard_normal(size=(lat_size,lat_size))

# Step 1
x = Gibbs_sampler(1, 50, lat_size=100)

# Step 2
eta = Gaussian_noise()
padded_eta = np.pad(eta, ((1, 1), (1, 1)), 'constant')
y = x + padded_eta

# Step 3
def calc_noise(mat, i, j, val, sigma):
    return 1/(2 * pow(sigma, 2)) * pow((mat[i][j] - val), 2)

# returns prob for +1
def calc_prob_with_noise(mat, Temp, i,j, sigma):
    x_plus = calc_neigbhors_sum(mat, i, j, 1)
    x_minus = calc_neigbhors_sum(mat, i, j, -1)
    noise_plus = calc_noise(mat, i, j, 1, sigma)
    noise_minus = calc_noise(mat, i, j, -1, sigma)
    Z_temp = np.exp((1/Temp) * x_plus - noise_plus) + np.exp((1/Temp) * x_minus - noise_minus)
    return np.exp((1/Temp) * x_plus - noise_plus)/Z_temp

def sample_site_with_noise(mat, Temp, i, j, sigma):
    prob = calc_prob_with_noise(mat, Temp, i,j, sigma)
    return np.random.choice([1, -1], 1, p=[prob, 1-prob])

def MRF_iteration_with_noise(mat, Temp, sigma, lat_size=8):
    for i in range(lat_size):
        for j in range(lat_size):
            mat[i+1][j+1] = sample_site_with_noise(mat, Temp, i+1, j+1, sigma)
    return mat

def Gibbs_sampler_with_noise(Temp, iterations, y, sigma=2, lat_size=8):
    padded_mat = y
    for iteration in range(iterations):
        padded_mat = MRF_iteration_with_noise(padded_mat, Temp, sigma)
    return padded_mat

restored_matrix = Gibbs_sampler_with_noise(1, 50, y, lat_size=100)
pass

#TODO: compare restored_matrix with x (round the values to -1 and 1 respectively?)

# Step 4