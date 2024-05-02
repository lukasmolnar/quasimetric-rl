import os
import torch
from omegaconf import OmegaConf, SCMode
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from quasimetric_rl.data import Dataset
from quasimetric_rl.modules import QRLAgent, QRLConf

NOVEL = True
EPISODE_LENGTH = 1000
CHECKPOINT_DIR = './online/results/gcrl_MountainCar-v0/run_novel_200k_downsample/'
CHECKPOINT = 'checkpoint_env00200000_opt00009500_final.pth'

# find all .pth files in the directory
checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.pth')]
# filter out the checkpoint
checkpoints = [f for f in checkpoints if 'checkpoint' in f]

# sort the checkpoints by the number of steps
checkpoints.sort(key=lambda x: int(x.split('_')[1][3:]))

with open(CHECKPOINT_DIR + '/config.yaml', 'r') as f:
    # load saved conf
    conf = OmegaConf.create(yaml.safe_load(f))


# 1. How to create env
dataset: Dataset = Dataset.Conf(kind=conf.env.kind, name=conf.env.name).make(dummy=True)  # dummy: don't load data
env = dataset.create_env()  # <-- you can use this now!
env.episode_length = EPISODE_LENGTH
# episodes = list(dataset.load_episodes())  # if you want to load episodes for offline data


# 2. How to re-create QRL agent
agent_conf: QRLConf = OmegaConf.to_container(
  OmegaConf.merge(OmegaConf.structured(QRLConf()), conf.agent),  # overwrite with loaded conf
  structured_config_mode=SCMode.INSTANTIATE,  # create the object
)

for checkpoint in checkpoints:
    agent: QRLAgent = agent_conf.make(env_spec=dataset.env_spec, total_optim_steps=1)[0]  # you can move to your fav device


    # 3. Load checkpoint
    agent.load_state_dict(torch.load(CHECKPOINT_DIR + checkpoint, map_location='cpu')['agent'])


    # 4. Simulate
    obs_dict = env.reset()
    goal = torch.tensor(obs_dict['desired_goal'])
    latent_goal = agent.critics[0].encoder(goal)
    min_pos = -1.2
    max_pos = 0.6
    min_vel = -0.07
    max_vel = 0.07

    # create grid of positions and velocities
    positions = np.linspace(min_pos, max_pos, 100)
    velocities = np.linspace(min_vel, max_vel, 100)
    grid = np.meshgrid(positions, velocities)
    grid = np.stack(grid, axis=-1)
    grid = grid.reshape(-1, 2)
    # get the latent representation of the grid
    grid = torch.tensor(grid).float()
    latent_grid = agent.critics[0].encoder(grid)
    # get the latent representation of the goal
    latent_goal_grid = latent_goal.repeat(latent_grid.shape[0], 1)

    distances = torch.norm(latent_grid - latent_goal_grid, dim=-1)

    # now visualize the distances in a heatmap
    dir = CHECKPOINT_DIR + 'heatmap/'
    os.makedirs(dir, exist_ok=True)

    fig, ax = plt.subplots()
    cmap = plt.get_cmap('viridis')
    norm = Normalize(vmin=0, vmax=1)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    np_d = distances.detach().numpy()
    colors = sm.to_rgba(np_d)
    ax.scatter(grid[:,0], grid[:,1], c=colors)
    plt.colorbar(sm)
    n = int(checkpoint.split('_')[1][3:])
    plt.title(f'Checkpoint {n}')
    plt.xlabel('Position')
    plt.ylabel('Velocity')
    plt.savefig(dir + f'{n}_norm.png')
    plt.close()

    # correct way of calculating the quasimetric distance with the model
    distances = agent.critics[0].quasimetric_model(latent_grid, latent_goal_grid)

    # now visualize the log distances in a heatmap
    fig, ax = plt.subplots()
    cmap = plt.get_cmap('viridis')
    norm = Normalize(vmin=0, vmax=1)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    np_d = np.log(distances.detach().numpy())
    colors = sm.to_rgba(np_d)
    ax.scatter(grid[:,0], grid[:,1], c=colors)
    plt.colorbar(sm)
    n = int(checkpoint.split('_')[1][3:])
    plt.title(f'Checkpoint {n}')
    plt.xlabel('Position')
    plt.ylabel('Velocity')
    plt.savefig(dir + f'{n}_quasimetric.png')
    plt.close()

# now generate a gif
import imageio

images = []
images_log = []
for checkpoint in checkpoints:
    n = int(checkpoint.split('_')[1][3:])
    images.append(imageio.imread(CHECKPOINT_DIR + f'/heatmap/{n}_norm.png'))
    images_log.append(imageio.imread(CHECKPOINT_DIR + f'/heatmap/{n}_quasimetric.png'))

imageio.mimsave(CHECKPOINT_DIR + '/heatmap_norm.gif', images, duration=0.5)
imageio.mimsave(CHECKPOINT_DIR + '/heatmap_quasimetric.gif', images_log, duration=0.5)