import numpy as np
import scipy.stats as stats
from scipy.stats import expon
import matplotlib.pyplot as plt

class ExponentialFit(): 
    def __init__(self):
        pass

    def load_data(self, file_path): 
        """load time series data and return the time interval"""
        with open(file_path, "r") as f:
            data = f.readlines()
        time_spike = np.array([float(line.strip()) for line in data])
        time_interval = np.diff(time_spike)
        min_tau = min(time_interval)
        print("minimum tau value:", min_tau)
        return time_interval, min_tau
    
    def fit_expon(self, data): 
        """fit the data into exponential distribution and return the estimated location and scale (1/lambda). """
        loc, scale = expon.fit(data) 
        lambda_hat = 1 / scale
        print("Location:", loc)
        print("Scale:", scale)
        print("Estimated lambda:", lambda_hat)
        return loc, scale, lambda_hat

    def ks_test(self, data, loc, scale): 
        """test whether the data fit the exponential distribution with given location and scale"""
        ks_statistic, ks_p_value = stats.kstest(data, 'expon', args=(loc, scale)) 
        print(f"KS Statistic: {ks_statistic}, p-value: {ks_p_value}")

    def sample_expon(self, loc, scale, size): 
        """sample new data from exponential distrbution with given loc and scale. """
        samples = loc + np.random.exponential(scale=scale, size=size)
        return samples

    def calculate_spiking_date(self, data): 
        mean_tau = np.mean(data)
        spiking_rate = 1/mean_tau
        print(fr"Mean inter-spike interval <$\tau$>", mean_tau)
        print("Average spiking rate:", spiking_rate)

def plot_expon(loc, scale): 
    x = np.linspace(0, 100, 1001)
    y = expon.pdf(x, loc=loc, scale=scale)
    plt.plot(x, y, label=fr'Exponential PDF ($\tau_0$={loc:.2f}, $\lambda$={1/scale:.2f})', color="r", alpha=0.5)

def plot_hist(data, file_path, title=None): 
    """plot the given data in histogram. """
    plt.hist(data, bins=100, density=True, color='skyblue', edgecolor='black', alpha=0.75, label=fr"Time intervals of neuron spike ($\tau$)")
    plt.title(title, fontsize=16)
    plt.xlabel(fr"$\tau$", fontsize=14)
    plt.ylabel(fr"P$(\tau)$", fontsize=14)
    plt.grid()
    plt.legend()
    if file_path: 
        plt.savefig(file_path, dpi=300)
    plt.show()

def plot_time_series(data): 
    """plot the time seties data."""
    x = np.linspace(0, len(data) - 1, len(data))
    plt.figure(dpi=300)
    plt.figure(figsize=(10, 5))
    plt.plot(x, data, color='royalblue', linewidth=-.1, label='Neuron Data')
    plt.title("Neuron Data Over Time", fontsize=16)
    plt.xlabel("Time (arbitrary units)", fontsize=14)
    plt.ylabel("Signal", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()