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
CHECKPOINT_DIR = '/Users/hanno/Documents/_Studium/MIT2/quasimetric-rl/results_csv_all/final_checkpoints'
CHECKPOINT = '/Users/hanno/Documents/_Studium/MIT2/quasimetric-rl/results_csv_all/final_checkpoints/random_5.pth'

checkpoints = [CHECKPOINT]

# name of checkpoint
checkpoint_name = CHECKPOINT.split('/')[-1].split('.')[0]

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
    agent.load_state_dict(torch.load(checkpoint, map_location='cpu')['agent'])


    # 4. Simulate
    obs_dict = env.reset()
    goal = torch.tensor(obs_dict['desired_goal'])
    latent_goal = agent.critics[0].encoder(goal)
    min_pos = -1.2
    max_pos = 0.6
    min_vel = -0.07
    max_vel = 0.07

    # now visualize the distances in a heatmap
    dir = CHECKPOINT_DIR.split('final_checkpoints')[0] + 'heatmaps' + '/'
    os.makedirs(dir, exist_ok=True)

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

    # distances = torch.norm(latent_grid - latent_goal_grid, dim=-1)
    # fig, ax = plt.subplots()
    # cmap = plt.get_cmap('viridis')
    # norm = Normalize(vmin=0, vmax=1)
    # sm = ScalarMappable(norm=norm, cmap=cmap)
    # np_d = distances.detach().numpy()
    # np_d = np.emath.logn(100, np_d + 1)
    # colors = sm.to_rgba(np_d)
    # ax.scatter(grid[:,0], grid[:,1], c=colors)
    # plt.colorbar(sm)
    # plt.title('Epsilon-greedy Novel-state Exploration (ε-NS)')
    # plt.xlabel('Position')
    # plt.ylabel('Velocity')
    # plt.savefig(dir + f'log100_norm_{checkpoint_name}.png', dpi=300)
    # plt.close()

    distances = agent.critics[0].quasimetric_model(latent_grid, latent_goal_grid)
    fig, ax = plt.subplots()
    cmap = plt.get_cmap('viridis')
    norm = Normalize(vmin=0, vmax=1)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    np_d = distances.detach().numpy()
    np_d = np.emath.logn(100, np_d + 1)
    colors = sm.to_rgba(np_d)
    ax.scatter(grid[:,0], grid[:,1], c=colors)
    plt.colorbar(sm)
    plt.title('Random Baseline (R):')
    plt.xlabel('Position')
    plt.ylabel('Velocity')
    plt.savefig(dir + f'log100_quasimetric_{checkpoint_name}.png', dpi=300)
    plt.close()

# # now generate a gif
# import imageio

# images = []
# images_log = []
# n = 1
# for checkpoint in checkpoints:
#     # n = int(checkpoint.split('_')[1][3:])
#     n += 1
#     images.append(imageio.imread(CHECKPOINT_DIR + f'/heatmap/{n}_norm.png'))
#     images_log.append(imageio.imread(CHECKPOINT_DIR + f'/heatmap/{n}_quasimetric.png'))

# imageio.mimsave('heatmap_norm.gif', images, duration=0.5)
# imageio.mimsave('heatmap_quasimetric.gif', images_log, duration=0.5)