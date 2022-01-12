import numpy as np
import matplotlib.pyplot as plt

# Non-normalized probability of state occupation
def boltzmann(energy, temperature):
    beta = 1./temperature
    return np.exp(-beta*energy)

def single_state_partition_function(energy, temperature):
    beta = 1./temperature
    return np.exp(-beta*energy/2) / (1 - np.exp(-beta*energy))

L = 10.0
h = 1.0
c = 1.0
root_frequency = c/(2*L)
def photon_energy(n_tuple):
    return h*root_frequency*np.linalg.norm(n_tuple)

# Calculates a cutoff n, where probabilities become insignificant
def n_max(temperature, frac_thresh=1.e-3):
    max_energy = -temperature*np.log(frac_thresh)
    return int(max_energy/(h*root_frequency))

# Serves as a normalization factor for the boltzmann distribution
def partition_function(temperature, n_max):
    part_func = 0.
    for i in range(n_max):
        for j in range(n_max):
            for k in range(n_max):
                n_tuple = np.array((i+1,j+1,k+1))
                if np.linalg.norm(n_tuple)<=n_max:
                    part_func += single_state_partition_function(photon_energy(n_tuple), temperature)
    return part_func

################################

## Sampling with inverse CDF
# First need to choose a random angle, then a random n, then round to nearest n_tuple
rng = np.random.default_rng()

def rand_theta():
    return 0.5*np.pi*rng.uniform()

# Sample phi via inverse transform sampling
def rand_phi():
    temp = rng.uniform()
    return np.arccos(-temp+1)

# The integral of x^2*exp(-a*x)
def int_x2_e_min_ax(x, a, b):
    return b * (2./a**3 - ((a*x)**2 + 2.*a*x + 2.) * np.exp(-a*x))

# Invert the integral of x^2*exp(-a*x) via binary search
def inv_int_x2_e_min_ax(y, a, b, tolerance=1e-3):
    lower_bound = 0.0
    upper_bound = 2.*a**-3
    midpoint = (lower_bound + upper_bound) / 2.
    
    while (int_x2_e_min_ax(upper_bound, a, b) - y < 0):
        # Upper bound too low, double it
        upper_bound *= 2.0
        midpoint = (lower_bound + upper_bound) / 2.

    temp = int_x2_e_min_ax(midpoint, a, b) - y
    if abs(temp) < tolerance:
        return midpoint

    counter = 0
    while (abs(temp) > tolerance):
        counter +=1
        print(counter)
        if temp>0:
            # too big
            upper_bound = midpoint
        else:
            # too small
            lower_bound = midpoint
        midpoint = (lower_bound + upper_bound) / 2.
        temp = int_x2_e_min_ax(midpoint, a, b) - y
        if abs(temp) < tolerance:
            return midpoint

inversion_testing = False
if inversion_testing:
    y_vals = np.linspace(0., 1.5, 100)
    x_vals = np.linspace(0., 2., 100)
    test_y = np.zeros_like(y_vals)
    test_x = np.zeros_like(x_vals)
    for i in range(len(y_vals)):
        print(i)
        test_y[i] = int_x2_e_min_ax(x_vals[i], 1., 1.e5)
        test_x[i] = inv_int_x2_e_min_ax(y_vals[i], 1., 1.e5)
    plt.plot(x_vals, test_y)
    plt.plot(test_x, y_vals)
    plt.show()


def rand_n(part_func, temperature):
    a = h*root_frequency/temperature
    temp = rng.uniform()
    n = inv_int_x2_e_min_ax(temp*2./a**3, a, part_func)
    return n

def spherical_to_cartesian(r, theta, phi):
    x = r*np.cos(theta)*np.sin(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(phi)
    return (x, y, z)

## Test sampling
sample_number = int(1e4)
test_theta = np.zeros(sample_number)
test_phi = np.zeros(sample_number)
test_x = np.zeros(sample_number)
test_y = np.zeros(sample_number)
test_z = np.zeros(sample_number)

const_radius = False
if not const_radius:
    temperature = 1.
    max_n = n_max(temperature)
    print(max_n)
    part_func = partition_function(temperature, max_n)
    print(part_func)
    test_n = np.zeros(sample_number)

for i in range(sample_number):
    print(i)
    test_theta[i] = rand_theta()
    test_phi[i] = rand_phi()
    if const_radius:
        test_x[i], test_y[i], test_z[i] = spherical_to_cartesian(1., test_theta[i], test_phi[i])
    else:
        test_n[i] = rand_n(part_func, temperature)
        test_x[i], test_y[i], test_z[i] = spherical_to_cartesian(test_n[i], test_theta[i], test_phi[i])

plt.plot(test_n)
plt.show()

'''
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(test_x, test_y, test_z, s=2)
plt.ylabel('y')
plt.xlabel('x')
plt.show()
'''