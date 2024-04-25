import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# import imageio
import imageio.v2 as imageio
import numpy as np
import os
from tqdm import tqdm

MOVIE = True
PATH = './online/results/gcrl_CartPole-v1/run_novel_20k/visited_states.pth'

# Load the tensor from a file
visited_states = torch.load(PATH)
visited_states_np = visited_states.cpu().numpy()

if 'CartPole' in PATH:
    # Only plot pole angle and angular velocity
    visited_states_np = visited_states_np[:, 2:]

# Initialize scatter plot
fig, ax = plt.subplots()
scat = ax.scatter([], [], s=5)

ax.set_title('Visited states')
ax.set_xlabel('Position')
ax.set_ylabel('Velocity')

if 'MountainCar' in PATH:
    ax.set_xlim(-1.2, 0.6)
    ax.set_ylim(-0.07, 0.07)

    # Mark initial states and goal state
    ax.axvline(x=0.5, color='green')
    ax.axvline(x=-0.5, color='gray')
    ax.plot([-0.6, -0.4], [0, 0], color='red')


if MOVIE:
    # Update function for animation
    def update(i):
        data = visited_states_np[:i, :]
        scat.set_offsets(data)

    # Create animation
    ani = FuncAnimation(fig, update, frames=range(len(visited_states_np)), repeat=False, interval=1)

else:
    # Regular scatter plot
    scat.set_offsets(visited_states_np)

plt.show()
plt.close()

# 5. How to create a gif
images = []
for i in tqdm(range(0, len(visited_states_np), int(len(visited_states_np)/200))):
    data = visited_states_np[:i, :]
    fig, ax = plt.subplots()
    scat = ax.scatter(data[:,0], data[:,1], s=5)
    ax.set_title('Visited states')
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