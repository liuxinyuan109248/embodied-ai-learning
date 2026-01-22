import matplotlib.pyplot as plt
import numpy as np

# Data for Ant-v4
ant_steps = [500, 1000, 2000]
ant_means = [4196.36, 4587.31, 4633.28]
ant_stds = [122.10, 127.16, 61.89]

# Data for Walker2d-v4
walker_steps = [500, 1000, 2000]
walker_means = [450.32, 1506.97, 1299.48]
walker_stds = [354.10, 1456.65, 2004.38]

def plot_experiment(steps, means, stds, title, ylabel, filename, color):
    plt.figure(figsize=(8, 6))
    
    # Plot mean line
    plt.plot(steps, means, marker='o', linestyle='-', color=color, linewidth=2, label='Mean Return')
    
    # Plot standard deviation as shaded area
    means = np.array(means)
    stds = np.array(stds)
    plt.fill_between(steps, means - stds, means + stds, color=color, alpha=0.2, label='Standard Deviation')
    
    plt.title(title, fontsize=14)
    plt.xlabel('Training Steps per Iteration', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(steps)  # Ensure x-axis shows exactly our step values
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved plot to {filename}")
    plt.close()

# Generate Ant-v4 plot
plot_experiment(
    ant_steps, 
    ant_means, 
    ant_stds, 
    'BC Performance vs Training Steps (Ant-v4)', 
    'Average Return', 
    'ant_hyperparam_plot.png',
    'blue'
)

# Generate Walker2d-v4 plot
plot_experiment(
    walker_steps, 
    walker_means, 
    walker_stds, 
    'BC Performance vs Training Steps (Walker2d-v4)', 
    'Average Return', 
    'walker2d_hyperparam_plot.png',
    'red'
)
