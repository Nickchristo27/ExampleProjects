# Nicholas Christophides    113319835

import scipy.stats as stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set options for displaying data with pandas
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)  # or a large number like 1000
pd.set_option('display.max_colwidth', None)  # to show long strings fully

# Read the matrix of equity data
equities = pd.read_csv("/Users/nickchristo/Downloads/Spring 2025/AMS 603/Final/Equities.csv", header=None)

# Calculate Log Returns of Assets
rows, columns = equities.shape  # Obtain matrix dimensions
log_returns = np.zeros((rows - 1, columns))  # Initialize a zeros array
names = np.array(["S&P 500", "Nasdaq Composite", "Apple", "Microsoft", "Netflix", "Goldman Sachs", "Nike"])  # Names

for i in range(columns):
    for j in range(rows - 1):
        log_returns[j, i] = np.log(equities.iloc[j + 1, i] / equities.iloc[j, i])

# Estimating normal distribution parameters for each equity ([Mean, Variance])
norm_params = np.zeros((7, 2))
for i in range(norm_params.shape[0]):
    norm_params[i, 0], norm_params[i, 1] = stats.norm.fit(log_returns[:, i])

# Estimating t-distribution parameters for each equity
t_params = np.zeros((7, 3))
for i in range(t_params.shape[0]):
    t_params[i, 0], t_params[i, 1], t_params[i, 2] = stats.t.fit(log_returns[:, i])

# Estimating normal-inverse Gaussian parameters for each equity
NIG_params = np.zeros((7, 4))
for i in range(NIG_params.shape[0]):
    NIG_params[i, 0], NIG_params[i, 1], NIG_params[i, 2], NIG_params[i, 3] = stats.norminvgauss.fit(log_returns[:, i])

# Estimating skewed t-distribution parameters for each equity
skewed_t_params = np.zeros((7, 4))
for i in range(skewed_t_params.shape[0]):
    skewed_t_params[i, 0], skewed_t_params[i, 1], skewed_t_params[i, 2], skewed_t_params[i, 3] = \
        stats.jf_skew_t.fit(log_returns[:, i])


# ********** Kolmogorov-Smirnov Tests **********

# Kolmogorov-Smirnov Test: Normal Distribution
ks_results_norm = np.zeros(7)  # Establish array for storing results

for i in range(7):
    ks_stat, ks_pvalue = stats.kstest(log_returns[:, i], "norm", args=(norm_params[i, 0], norm_params[i, 1]))
    ks_results_norm[i] = ks_pvalue

print('Kolmogorov-Smirnov Normal Test P-Values:\n')
print(pd.DataFrame(np.array([names, ks_results_norm])))

# Kolmogorov-Smirnov Test: T-Distribution
ks_results_t = np.zeros(7)  # Establish array for storing results

for i in range(7):
    ks_stat, ks_pvalue = stats.kstest(log_returns[:, i], "t", args=(t_params[i, 0], t_params[i, 1], t_params[i, 2]))
    ks_results_t[i] = ks_pvalue

print('\n\nKolmogorov-Smirnov T Test P-Values:\n')
print(pd.DataFrame(np.array([names, ks_results_t])))

# Kolmogorov-Smirnov Test: Normal-Inverse Gaussian Distribution
ks_results_NIG = np.zeros(7)  # Establish array for storing results

for i in range(7):
    ks_stat, ks_pvalue = stats.kstest(log_returns[:, i], "norminvgauss", args=(NIG_params[i, 0], NIG_params[i, 1],
                                                                               NIG_params[i, 2], NIG_params[i, 3]))
    ks_results_NIG[i] = ks_pvalue

print('\n\nKolmogorov-Smirnov NIG Test P-Values:\n')
print(pd.DataFrame(np.array([names, ks_results_NIG])))

# Kolmogorov-Smirnov Test: Skewed T-Distribution
ks_results_skew_t = np.zeros(7)  # Establish array for storing results

for i in range(7):
    ks_stat, ks_pvalue = stats.kstest(log_returns[:, i],
                                      "jf_skew_t", args=(skewed_t_params[i, 0], skewed_t_params[i, 1],
                                                         skewed_t_params[i, 2], skewed_t_params[i, 3]))
    ks_results_skew_t[i] = ks_pvalue

print('\n\nKolmogorov-Smirnov Skewed T Test P-Values:\n')
print(pd.DataFrame(np.array([names, ks_results_skew_t])))

# ********** Anderson-Darling Tests **********

# Anderson-Darling Test: Normal Distribution
ad_results_norm = np.zeros(7)  # Establish array for storing results

for i in range(7):
    ad_result = stats.anderson(log_returns[:, i], "norm")  # Find test statistic
    print(f'\n{ad_result.statistic}')

    # Find critical values at each level of significance
    for j in range(len(ad_result.critical_values)):
        sig = ad_result.significance_level[j]
        crit = ad_result.critical_values[j]
        print(f"Significance Level: {sig:.1f}%, Critical Value: {crit:.3f}")
    print("\n")

# ********** Q-Q Plots **********

# Q-Q Plots: Normal Distribution
# Sort the sample and compute theoretical quantiles
for i in range(7):
    returns_sorted = np.sort(log_returns[:, i])  # Sort Returns
    probs = np.linspace(0.01, 0.99, len(returns_sorted))
    theoretical_quantiles = stats.norm.ppf(probs, loc=norm_params[i, 0], scale=norm_params[i, 1])
    sample_quantiles = np.quantile(returns_sorted, probs)

    # Plot
    plt.scatter(theoretical_quantiles, sample_quantiles, label='QQ Points')
    plt.plot(theoretical_quantiles, theoretical_quantiles, 'r--', label='45-degree line')
    plt.title(f'QQ Plot: {names[i]} vs Estimated Normal')
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Sample Quantiles")
    plt.legend()
    plt.show()

# Q-Q Plots: T-Distribution
for i in range(7):
    returns_sorted = np.sort(log_returns[:, i])  # Sort Returns
    probs = np.linspace(0.01, 0.99, len(returns_sorted))
    theoretical_quantiles = stats.t.ppf(probs, df=t_params[i, 0], loc=t_params[i, 1], scale=t_params[i, 2])
    sample_quantiles = np.quantile(returns_sorted, probs)

    # Plot
    plt.scatter(theoretical_quantiles, sample_quantiles, label='QQ Points')
    plt.plot(theoretical_quantiles, theoretical_quantiles, 'r--', label='45-degree line')
    plt.title(f'QQ Plot: {names[i]} vs Estimated T-Dist.')
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Sample Quantiles")
    plt.legend()
    plt.show()

# Q-Q Plots: NIG Distribution
for i in range(7):
    returns_sorted = np.sort(log_returns[:, i])  # Sort Returns
    probs = np.linspace(0.01, 0.99, len(returns_sorted))
    theoretical_quantiles = stats.norminvgauss.ppf(probs, a=NIG_params[i, 0], b=NIG_params[i, 1],
                                                   loc=NIG_params[i, 2], scale=NIG_params[i, 3])
    sample_quantiles = np.quantile(returns_sorted, probs)

    # Plot
    plt.scatter(theoretical_quantiles, sample_quantiles, label='QQ Points')
    plt.plot(theoretical_quantiles, theoretical_quantiles, 'r--', label='45-degree line')
    plt.title(f'QQ Plot: {names[i]} vs Estimated NIG Dist.')
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Sample Quantiles")
    plt.legend()
    plt.show()

# Q-Q Plots: Skewed T-Distribution
for i in range(7):
    returns_sorted = np.sort(log_returns[:, i])  # Sort Returns
    probs = np.linspace(0.01, 0.99, len(returns_sorted))
    theoretical_quantiles = stats.jf_skew_t.ppf(probs, a=skewed_t_params[i, 0], b=skewed_t_params[i, 1],
                                                loc=skewed_t_params[i, 2], scale=skewed_t_params[i, 3])
    sample_quantiles = np.quantile(returns_sorted, probs)

    # Plot
    plt.scatter(theoretical_quantiles, sample_quantiles, label='QQ Points')
    plt.plot(theoretical_quantiles, theoretical_quantiles, 'r--', label='45-degree line')
    plt.title(f'QQ Plot: {names[i]} vs Estimated Skewed T-Dist.')
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Sample Quantiles")
    plt.legend()
    plt.show()
