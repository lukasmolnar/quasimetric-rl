import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

folder = '5_results_csv/'
files = [
    'novel_eps_greedy_state',
    'novel_eps_greedy_latent',
    'novel_pure_latent',
    'novel_pure_state',
]
# idx 1, 2, 3, 4, 5
idx = 1
indices = range(1, 6)
file_dir = [folder + file + f'_{idx}.csv' for file in files]
dfs = [pd.read_csv(file) for file in file_dir]

min_x = -1.2
max_x = 0.6
delta_x = max_x - min_x
min_v = -0.07
max_v = 0.07
delta_v = max_v - min_v
# remove is_success (x >= filter_x)
filter_x = 0.5
# remove first prefill samples
prefill = 20_000

def filter_df(df):
    n = len(df)
    # remove rows with is_success
    # df = df[df['observations_x'] < filter_x]
    df = df[prefill:]
    return df.reset_index(drop=True)

def load_and_filter_data(files, indices):
    dfs = []
    for idx in indices:
        current_files = [f"{folder}{file}_{idx}.csv" for file in files]
        current_dfs = [pd.read_csv(file) for file in current_files]
        filtered_dfs = [filter_df(df) for df in current_dfs]
        dfs.append(filtered_dfs)
    return dfs

def discretize(df, n=1000):
    bins = np.linspace(min_x, max_x, n)
    df['observations_x'] = pd.cut(df['observations_x'], bins=bins, labels=False)
    bins = np.linspace(min_v, max_v, n)
    df['observations_v'] = pd.cut(df['observations_v'], bins=bins, labels=False)
    df = df.dropna()
    return df

def combine_v_x(df):
    df = df.copy()
    # combine x and v strings
    df.loc[:, 'combined'] = df['observations_x'].astype(int).astype(str) + df['observations_v'].astype(int).astype(str)
    # turn into integers
    df.loc[:, 'combined'] = df['combined'].astype(int)
    # drop old columns
    df = df.drop(columns=['observations_x', 'observations_v'])
    return df

# entropy of a discrete distribution
def entropy(df):
    p = df.value_counts(normalize=True)
    # remove NaNs
    p = p.dropna()
    return -np.sum(p * np.log2(p))

def running_entropy(df, column, window=1000):
    return df[column].rolling(window=window, step=window).apply(lambda x: entropy(x))

def smooth(df, window=10):
    return df.rolling(window=window).mean()

def plot_and_save(data, labels, title, filename):
    plt.figure(figsize=(15, 10))
    for i, d in enumerate(data):
        plt.plot(d, label=labels[i])
    plt.legend()
    plt.title(title)
    plt.savefig(filename, dpi=300)
    plt.close()

def calculate_entropies(dfs):
    dfs = [df.copy() for df in dfs]
    dfs_discretized = [discretize(df) for df in dfs]
    dfs_combined = [combine_v_x(df) for df in dfs_discretized]
    entropies_combined = [running_entropy(df, 'combined') for df in dfs_combined]
    entropies_combined_smooth = [smooth(r, window=10) for r in entropies_combined]
    return entropies_combined_smooth

def trim_entropies(entropies):
    min_shape = min(e_i.shape[0] for e in entropies for e_i in e)
    return [[e_i[:min_shape] for e_i in e] for e in entropies]

# calculate mean and std of entropies
def calculate_mean_std(entropies):
    mean = np.mean(entropies, axis=0)
    std = np.std(entropies, axis=0)
    return mean, std

def plot_mean_std(mean, std, files, folder, filename='mean_std'):
    plt.figure(figsize=(15, 10))
    for i, m in enumerate(mean):
        plt.plot(m, label=files[i])
        # cap the std at 10 and 0
        # plt.fill_between(range(len(m)), m - std[i], m + std[i], alpha=0.3)
        plt.fill_between(range(len(m)), np.maximum(m - std[i], 0), np.minimum(m + std[i], 10), alpha=0.3)
    plt.legend()
    plt.title('Mean and std of entropy')
    plt.savefig(folder + filename, dpi=300)
    plt.close()

# ENTROPY 
dfs_all_indices = load_and_filter_data(files, indices)
entropies_all_indices = [calculate_entropies(dfs) for dfs in dfs_all_indices]
trimmed_entropies_all_indices = trim_entropies(entropies_all_indices)
mean, std = calculate_mean_std(trimmed_entropies_all_indices)
plot_mean_std(mean, std, files, folder, 'Entropy_mean_std')

def plot_entropy(name, dfs, folder, files):
    dfs_discretized = [discretize(df) for df in dfs]

    print('Plotting smoothed entropy of x ...')
    running_entropies_x = [running_entropy(df, 'observations_x') for df in dfs_discretized]
    running_entropies_x_smooth = [smooth(r, window=10) for r in running_entropies_x]
    plot_and_save(running_entropies_x_smooth, files, name + ' of x (smoothed)', folder + name.replace(' ', '_') + '_x_smooth')
    running_entropies_v = [running_entropy(df, 'observations_v') for df in dfs_discretized]
    running_entropies_v_smooth = [smooth(r, window=10) for r in running_entropies_v]
    print('Plotting smoothed entropy of v ...')
    plot_and_save(running_entropies_v_smooth, files, name + ' of v (smoothed)', folder + name.replace(' ', '_') + '_v_smooth')

    # combine x and v
    print('Plotting smoothed entropy of x + v ...')
    dfs_combined = [combine_v_x(df) for df in dfs_discretized]
    running_entropies_combined = [running_entropy(df, 'combined') for df in dfs_combined]
    running_entropies_combined_smooth = [smooth(r, window=10) for r in running_entropies_combined]
    plot_and_save(running_entropies_combined_smooth, files, name + ' of x + v (smoothed)', folder + name.replace(' ', '_') + '_combined_smooth')


# plot histogram of x and v
def plot_hist(name, dfs, folder, files, column='observations_x'):
    plt.figure()
    plt.figure(figsize=(15, 10))
    for i, df in enumerate(dfs):
        plt.hist(df[column], bins=100, alpha=0.5, label=files[i], density=True)
    plt.legend()
    plt.title(name)
    plt.savefig(folder + name.replace(' ', '_') , dpi=300)
    plt.close()


# calculate running variance over the last 10k samples
def running_variance(dfs, window = 20000):
    running_variance_x = []
    running_variance_v = []
    running_variance = []
    for df in dfs:
        running_variance_x.append(df['observations_x'].rolling(window=window).var())
        running_variance_v.append(df['observations_v'].rolling(window=window).var())
        running_variance.append((df['observations_x'].rolling(window=window).var())/delta_x + (df['observations_v'].rolling(window=window).var())/delta_v)
    return running_variance_x, running_variance_v, running_variance


# plot running variance
def plot(name, running_variance_x, folder, files):
    plt.figure()
    plt.figure(figsize=(15, 10))
    for i, r in enumerate(running_variance_x):
        plt.plot(r, label=files[i])
    plt.legend()
    plt.title(name)
    plt.savefig(folder + name.replace(' ', '_') , dpi=300)
    plt.close()

dfs_filtered = [filter_df(df) for df in dfs]
plot_entropy('Running entropy', dfs_filtered.copy(), folder, files)
plot_hist('Histogram of x', dfs_filtered.copy(), folder, files, 'observations_x')
plot_hist('Histogram of v', dfs_filtered.copy(), folder, files, 'observations_v')
running_variance_x, running_variance_v, running_variance_x_v = running_variance(dfs)
plot('Running variance of x', running_variance_x, folder, files)
plot('Running variance of v', running_variance_v, folder, files)
plot('Running variance of x + v', running_variance_x_v, folder, files)