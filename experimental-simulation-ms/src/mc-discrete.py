import numpy as np
from scipy.stats import rv_histogram
import matplotlib.pyplot as plt

np.random.seed(0)

# Sample N_true items from a N(0,1) distribution
N_true = 100000
rn_true_mean = 0
rn_true_stdev = 1
x = np.random.normal(rn_true_mean, rn_true_stdev, N_true)



# Create a histogram with B bins from the sampled N_true items
B = 80
hist, bins = np.histogram(x, bins=B, density=True)


# *Montecarlo simulation of N_mc samples from the given histogram*
N_mc = 1000000

# Create a random variable from the histogram 
rv = rv_histogram((hist, bins))

# Sample points from the histogram
mc_samples = rv.rvs(size=N_mc, random_state=0)
# mc_samples

# Plot the histogram  together with the sampled points
plt.hist(x, bins=B, density=True, alpha=.3, label='Samples from $X \sim N(0,1)$')
plt.hist(mc_samples, bins=B, density=True, alpha=.4, label='MC samples $\sim Hist(X)$')
plt.legend()
# plt.show()
plt.savefig('./mc-samples.png')
