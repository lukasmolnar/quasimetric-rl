import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

folder = 'results_csv/'
files = [
    'novel_eps_greedy_state',
    'novel_eps_greedy_latent',
    'novel_latent',
    'novel_state',
]

dfs = [pd.read_csv(folder + file + '.csv') for file in files]

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
    df = df[df['observations_x'] < filter_x]
    df = df[prefill:]
    print(f'Filtered {len(df)} out of {n} samples')
    return df.reset_index(drop=True)

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

def plot(name, dfs, folder, files):
    dfs_discretized = [discretize(df) for df in dfs]

    print('Plotting entropy of x ...')
    running_entropies_x = [running_entropy(df, 'observations_x') for df in dfs_discretized]
    plot_and_save(running_entropies_x, files, name + ' of x', folder + name.replace(' ', '_') + '_x')
    print('Plotting entropy of v ...')
    running_entropies_v = [running_entropy(df, 'observations_v') for df in dfs_discretized]
    plot_and_save(running_entropies_v, files, name + ' of v', folder + name.replace(' ', '_') + '_v')
    print('Plotting smoothed entropy of x ...')
    running_entropies_x_smooth = [smooth(r, window=10) for r in running_entropies_x]
    plot_and_save(running_entropies_x_smooth, files, name + ' of x (smoothed)', folder + name.replace(' ', '_') + '_x_smooth')
    print('Plotting smoothed entropy of v ...')
    running_entropies_v_smooth = [smooth(r, window=10) for r in running_entropies_v]
    plot_and_save(running_entropies_v_smooth, files, name + ' of v (smoothed)', folder + name.replace(' ', '_') + '_v_smooth')

    # combine x and v
    print('Plotting entropy of x + v ...')
    dfs_combined = [combine_v_x(df) for df in dfs_discretized]
    running_entropies_combined = [running_entropy(df, 'combined') for df in dfs_combined]
    plot_and_save(running_entropies_combined, files, name + ' of x + v', folder + name.replace(' ', '_') + '_combined')
    print('Plotting smoothed entropy of x + v ...')
    running_entropies_combined_smooth = [smooth(r, window=10) for r in running_entropies_combined]
    plot_and_save(running_entropies_combined_smooth, files, name + ' of x + v (smoothed)', folder + name.replace(' ', '_') + '_combined_smooth')

plot('Running entropy', [df.reset_index(drop=True) for df in dfs], folder, files)

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

plot_hist('Histogram of x', dfs, folder, files, 'observations_x')
plot_hist('Histogram of v', dfs, folder, files, 'observations_v')

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

running_variance_x, running_variance_v, running_variance_x_v = running_variance(dfs)

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

plot('Running variance of x', running_variance_x, folder, files)
plot('Running variance of v', running_variance_v, folder, files)
plot('Running variance of x + v', running_variance_x_v, folder, files)