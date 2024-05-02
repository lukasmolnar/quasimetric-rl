from typing import *

import functools

import gym
import gym.spaces
import numpy as np

from ..memory import register_online_env


class GoalCondEnvWrapper(gym.ObservationWrapper):
    r"""
    Convert the concatenated observation space in GCRL into a better format with
    dict observations.
    """

    episode_length: int
    fixed_goal: Optional[np.ndarray]
    is_image_based: bool
    create_kwargs: Mapping[str, Any]

    def __init__(self, env: gym.Env, episode_length: int, fixed_goal: Optional[np.ndarray], is_image_based: bool):
        super().__init__(gym.wrappers.TimeLimit(env.unwrapped, episode_length))
        if is_image_based:
            single_ospace = gym.spaces.Box(
                low=np.full((64, 64, 3), 0),
                high=np.full((64, 64, 3), 255),
                dtype=np.uint8,
            )
        else:
            assert isinstance(env.observation_space, gym.spaces.Box)
            ospace: gym.spaces.Box = env.observation_space
            assert len(ospace.shape) == 1
            if fixed_goal is None:
                # Fetch environments have a concatenated observation space
                single_ospace = gym.spaces.Box(
                    low=np.split(ospace.low, 2)[0],
                    high=np.split(ospace.high, 2)[0],
                    dtype=ospace.dtype,
                )
            else:
                single_ospace = ospace
        self.observation_space = gym.spaces.Dict(dict(
            observation=single_ospace,
            achieved_goal=single_ospace,
            desired_goal=single_ospace,
        ))
        self.episode_length = episode_length
        self.fixed_goal = fixed_goal
        self.is_image_based = is_image_based

    def observation(self, observation):
        if self.fixed_goal is None:
            # Fetch environments have a concatenated observation space
            o, g = np.split(observation, 2)
        else:
            o = observation
            g = self.fixed_goal
            
        if self.is_image_based:
            o = o.reshape(64, 64, 3)
            g = g.reshape(64, 64, 3)
        odict = dict(
            observation=o,
            achieved_goal=o,
            desired_goal=g,
        )
        return odict



def create_env_from_spec(name: str):
    from . import fetch_envs  # lazy init mujoco/mujoco_py, which has a range of installation issues

    is_image_based = name.endswith('Image')

    if name in fixed_goals:
        fixed_goal = fixed_goals[name]
        env = gym.make(name)
    else:
        # Fetch environments do not have fixed goals
        fixed_goal = None
        env: gym.Env = getattr(fetch_envs, name + 'Env')()
    
    return GoalCondEnvWrapper(env, episode_length=1000, fixed_goal=fixed_goal, 
                              is_image_based=is_image_based)


valid_names = (
    'FetchReach',
    'FetchReachImage',
    'FetchPush',
    'FetchPushImage',
    'FetchSlide',
    'MountainCar-v0',
    'MountainCarContinuous-v0',
    'Pendulum-v0',
    'CartPole-v1'
)

fixed_goals = {
    'MountainCar-v0': np.array([0.5, 0.0]), # Discrete mountain car (different goal to continuous)
    'MountainCarContinuous-v0': np.array([0.45, 0.0]),
    'Pendulum-v0': np.array([1.0, 0.0, 0.0]),
    'CartPole-v1': np.array([0.0, 0.0, 0.0, 0.0])
}

for name in valid_names:
    register_online_env(
        'gcrl', name,
        create_env_fn=functools.partial(create_env_from_spec, name),
        episode_length=1000,
    )
