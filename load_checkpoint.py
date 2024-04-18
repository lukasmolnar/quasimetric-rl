import os
import torch
from omegaconf import OmegaConf, SCMode
import yaml
import numpy as np

from quasimetric_rl.data import Dataset
from quasimetric_rl.modules import QRLAgent, QRLConf

EPISODE_LENGTH = 1000
CHECKPOINT = './online/results/gcrl_MountainCarContinuous-v0/run_1/checkpoint_env00015000_opt00005500_final.pth'

checkpoint_dir = os.path.dirname(CHECKPOINT)
with open(checkpoint_dir + '/config.yaml', 'r') as f:
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
agent: QRLAgent = agent_conf.make(env_spec=dataset.env_spec, total_optim_steps=1)[0]  # you can move to your fav device


# 3. Load checkpoint
agent.load_state_dict(torch.load(CHECKPOINT, map_location='cpu')['agent'])


# 4. Simulate
obs_dict = env.reset()
done = False
while not done:
    obs = torch.tensor(obs_dict['observation'])
    goal = torch.tensor(obs_dict['desired_goal'])
    dist = agent.actor(obs, goal)
    action = dist.sample()
    obs_dict, reward, terminal, info = env.step(np.asarray(action))
    env.render()
    if done:
        break