import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# import imageio
import imageio
import numpy as np
import os
from tqdm import tqdm

GIF = True
DIR = './online/results/gcrl_MountainCar-v0/run_novel_1M_downsample/'
PATH = DIR + 'visited_states50000.pth'

# Load the tensor from a file
visited_states = torch.load(PATH)
visited_states_np = visited_states.cpu().numpy()

if 'CartPole' in DIR:
    # Only plot pole angle and angular velocity
    visited_states_np = visited_states_np[:, 2:]

# Initialize scatter plot
fig, ax = plt.subplots(figsize=(8, 5))
scat = ax.scatter([], [], s=1)

ax.set_title('Visited states')
ax.set_xlabel('Position')
ax.set_ylabel('Velocity')

if 'MountainCar' in DIR:
    ax.set_xlim(-1.2, 0.6)
    ax.set_ylim(-0.07, 0.07)

    # Mark initial states and goal state
    ax.axvline(x=0.5, color='green')
    ax.axvline(x=-0.5, color='gray')
    ax.plot([-0.6, -0.4], [0, 0], color='red')


# Genereate scatter plot
scat.set_offsets(visited_states_np)

img_path = PATH.replace('.pth', '.png')
plt.savefig(img_path)
plt.close()

if GIF:
    images = []
    for i in tqdm(range(0, len(visited_states_np), int(len(visited_states_np)/200))):
        data = visited_states_np[:i, :]
        fig, ax = plt.subplots()
        scat = ax.scatter(data[:,0], data[:,1], s=5)
        ax.set_title('Novel Exploration')
        ax.set_xlabel('Position')
        ax.set_ylabel('Velocity')
        ax.set_xlim(-1.2, 0.6)
        ax.set_ylim(-0.07, 0.07)
        ax.axvline(x=0.5, color='green')
        ax.axvline(x=-0.5, color='gray')
        ax.plot([-0.6, -0.4], [0, 0], color='red')
        img_dir = PATH.replace('.pth', '_img/')
        os.makedirs(img_dir, exist_ok=True)
        img = img_dir + f'{i}.png'
        plt.savefig(img)
        images.append(imageio.imread(img))
        plt.close()

    imageio.mimsave(PATH.replace('.pth', '.gif'), images, fps=30)