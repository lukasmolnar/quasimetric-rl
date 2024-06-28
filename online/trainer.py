from __future__ import annotations
from typing import *
from typing_extensions import Self

import time
import attrs
import logging

import gym.spaces
import numpy as np
import torch
import torch.utils.data
import random

from quasimetric_rl.modules import QRLConf, QRLAgent, QRLLosses, InfoT
from quasimetric_rl.data import BatchData, EpisodeData, MultiEpisodeData
from quasimetric_rl.data.online import ReplayBuffer, FixedLengthEnvWrapper, LatentCollection
from quasimetric_rl.utils import tqdm


def first_nonzero(arr: torch.Tensor, dim: bool = -1, invalid_val: int = -1):
    mask = (arr != 0)
    return torch.where(mask.any(dim=dim), mask.to(torch.uint8).argmax(dim=dim), invalid_val)


@attrs.define(kw_only=True)
class EvalEpisodeResult:
    timestep_reward: torch.Tensor
    episode_return: torch.Tensor
    timestep_is_success: torch.Tensor
    is_success: torch.Tensor
    hitting_time: torch.Tensor

    @classmethod
    def from_timestep_reward_is_success(cls, timestep_reward: torch.Tensor, timestep_is_success: torch.Tensor) -> Self:
        return cls(
            timestep_reward=timestep_reward,
            episode_return=timestep_reward.sum(-1),
            timestep_is_success=timestep_is_success,
            is_success=timestep_is_success.any(dim=-1),
            hitting_time=first_nonzero(timestep_is_success, dim=-1),  # NB this is off by 1
        )


@attrs.define(kw_only=True)
class InteractionConf:
    total_env_steps: int = attrs.field(default=int(300_000), validator=attrs.validators.gt(0))

    num_prefill_episodes: int = attrs.field(default=20, validator=attrs.validators.ge(0))
    num_samples_per_cycle: int = attrs.field(default=500, validator=attrs.validators.ge(0))
    num_rollouts_per_cycle: int = attrs.field(default=10, validator=attrs.validators.ge(0))
    num_eval_episodes: int = attrs.field(default=20, validator=attrs.validators.ge(0))

    exploration_method: str = attrs.field(default='novel', validator=attrs.validators.in_(['eps_greedy', 'novel', 'novel_eps_greedy', 'softmax', 'random']))
    exploration_eps: float = attrs.field(default=0.3, validator=attrs.validators.ge(0))

    downsample: str = attrs.field(default='downsample', validator=attrs.validators.in_(['downsample', 'cluster_latents', 'cluster_states']))
    downsample_n: int = attrs.field(default=40_000, validator=attrs.validators.gt(0))
    novel_k: int = attrs.field(default=50, validator=attrs.validators.gt(0))
    novelty_mode: str = attrs.field(default='state', validator=attrs.validators.in_(['state', 'latent']))


class Trainer(object):
    agent: QRLAgent
    losses: QRLLosses
    device: torch.device
    replay: ReplayBuffer
    batch_size: int
    latent_collection: LatentCollection

    total_env_steps: int

    num_prefill_episodes: int
    num_samples_per_cycle: int
    num_rollouts_per_cycle: int
    num_eval_episodes: int

    exploration_method: str
    exploration_eps: float

    def get_total_optim_steps(self, total_env_steps: int):
        total_env_steps -= self.replay.num_episodes_realized * self.replay.episode_length
        total_env_steps -= self.num_prefill_episodes * self.replay.episode_length
        num_cycles = 1

        if total_env_steps != 0:
            assert self.num_rollouts_per_cycle > 0
            num_cycles = int(np.ceil(total_env_steps / (self.num_rollouts_per_cycle * self.replay.episode_length)))

        return self.num_samples_per_cycle * num_cycles

    def __init__(self, *, agent_conf: QRLConf,
                 device: torch.device,
                 replay: ReplayBuffer,
                 batch_size: int,
                 interaction_conf: InteractionConf,
                 eval_seed: int = 416923159):

        self.device = device
        self.replay = replay
        self.eval_seed = eval_seed
        self.batch_size = batch_size
        self.latent_collection = LatentCollection(
            device=self.device,
            downsample_n=interaction_conf.downsample_n, 
            novel_k=interaction_conf.novel_k
            )
        self.downsample = interaction_conf.downsample
        self.novelty_mode = interaction_conf.novelty_mode
        self.exploration_method = interaction_conf.exploration_method
        self.exploration_eps = interaction_conf.exploration_eps

        self.total_env_steps = interaction_conf.total_env_steps
        self.num_samples_per_cycle = interaction_conf.num_samples_per_cycle
        self.num_rollouts_per_cycle = interaction_conf.num_rollouts_per_cycle
        self.num_eval_episodes = interaction_conf.num_eval_episodes
        self.num_prefill_episodes = interaction_conf.num_prefill_episodes

        self.agent, self.losses = agent_conf.make(
            env_spec=replay.env_spec,
            total_optim_steps=self.get_total_optim_steps(interaction_conf.total_env_steps))
        self.agent.to(device)
        self.losses.to(device)

        logging.info('Agent:\n\t' + str(self.agent).replace('\n', '\n\t') + '\n\n')
        logging.info('Losses:\n\t' + str(self.losses).replace('\n', '\n\t') + '\n\n')

    def make_collect_env(self) -> FixedLengthEnvWrapper:
        return self.replay.create_env()

    def make_evaluate_env(self) -> FixedLengthEnvWrapper:
        env = self.replay.create_env()
        # a hack to expose more signal from some envs :)
        if hasattr(env, 'reward_mode') and len(self.replay.env_spec.observation_shape) == 1:
            env.unwrapped.reward_mode = 'dense'
        env.seed(self.eval_seed)
        return env

    def sample(self) -> BatchData:
        return self.replay.sample(
            self.batch_size,
        ).to(self.device)
    
    def env_dynamic(self, env, obs, action):
        assert env.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"
        if env.spec.id == "MountainCar-v0":
            position, velocity = obs
            velocity += (action - 1) * env.force + np.cos(3 * position) * (-env.gravity)
            velocity = np.clip(velocity, -env.max_speed, env.max_speed)
            position += velocity
            position = np.clip(position, env.min_position, env.max_position)
            if position == env.min_position and velocity < 0:
                velocity = 0

            terminated = bool(
                position >= env.goal_position and velocity >= env.goal_velocity
            )
            # return tensor of next state
            next_obs = (position, velocity)

        else:
            raise NotImplementedError
        return next_obs, terminated
    
    def greedy_actor(self, env, obs: torch.Tensor, goal: torch.Tensor, space: gym.spaces.Space):
        num_actions = env.action_space.n
        actions = torch.tensor([i for i in range(num_actions)]).to(self.device)
        critic_0 = self.agent.critics[0]
        latent_goal = critic_0.encoder(goal.to(self.device))
        dist_to_goal = np.inf
        best_actions = []
        # Iterate over all possible actions
        # Get the latent representation of the next state given the current state and the one-hot encoded action
        latent_state = critic_0.encoder(obs.to(self.device))
        next_latent_states = critic_0.latent_dynamics(latent_state, actions)
        for i in range(num_actions):
            a = actions[i]
            next_state = next_latent_states[i,:]
            d = critic_0.quasimetric_model(next_state, latent_goal)
            if d < dist_to_goal:
                dist_to_goal = d
                best_actions = [a]
            elif d == dist_to_goal:
                best_actions.append(a)
        best = best_actions[torch.randint(len(best_actions), (1,)).item()]
        return best


    def collect_random_rollout(self, *, store: bool = True, env: Optional[FixedLengthEnvWrapper] = None) -> EpisodeData:
        rollout = self.replay.collect_rollout(
            lambda obs, goal, space: space.sample(),
            env=env,
        )
        if store:
            self.replay.add_rollout(rollout)
            self.latent_collection.add_rollout(rollout, self.agent.critics[0])
        return rollout

    def collect_novel_rollout(self, *, eval: bool = False, store: bool = True,
                        env: Optional[FixedLengthEnvWrapper] = None) -> EpisodeData:
        @torch.no_grad()
        def actor(obs: torch.Tensor, goal: torch.Tensor, space: gym.spaces.Space):
            def novel_actor(obs: torch.Tensor, goal: torch.Tensor, space: gym.spaces.Space, mode = 'state'):
                action_novelty = {}
                num_actions = env.action_space.n
                # Iterate over all possible actions
                actions = torch.tensor([i for i in range(num_actions)]).to(self.device)
                critic_0 = self.agent.critics[0]
                if mode == 'latent':
                    # Get the latent representation of the next state given the current state and the one-hot encoded action
                    # latent_state = critic_0.encoder(obs[None].to(self.device))
                    # next_latent_states = critic_0.latent_dynamics(latent_state, actions)
                    # Calculate the novelty of the next state
                    for i in range(num_actions):
                        a = actions[i]
                        # next_state = next_latent_states[i,:].to(self.device)
                        # HACK: we might need to use the environment dynamics to get the next state
                        next_state, _ = self.env_dynamic(env, obs.cpu().numpy(), a.cpu().numpy())
                        next_state = critic_0.encoder(torch.tensor(next_state).to(self.device))
                        nov = self.latent_collection.novelty(next_state, critic_0, mode = mode)
                        # Store the novelty of the next state with the action
                        action_novelty[a] = nov
                    # Get the action with the highest novelty
                    action = max(action_novelty, key=action_novelty.get).cpu()
                elif mode == 'state':
                    for i in range(num_actions):
                        a = actions[i]
                        next_state, _ = self.env_dynamic(env, obs.cpu().numpy(), a.cpu().numpy())
                        next_state = torch.tensor(next_state).to(self.device)
                        nov = self.latent_collection.novelty(next_state, critic_0, mode = mode)
                        action_novelty[a] = nov
                    action = max(action_novelty, key=action_novelty.get).cpu()
                else:
                    raise NotImplementedError(f"Mode {mode} not implemented")
                return action.cpu()
            
            # Epsilon-greedy action selection
            if not eval and random.random() < self.exploration_eps:
                best = novel_actor(obs, goal, space, mode = self.novelty_mode)
                # print("Novel action: ", best)
            else:
                best = self.greedy_actor(env, obs, goal, space)
                # print("Greedy action: ", best)
            return best.cpu()

        rollout = self.replay.collect_rollout(actor, env=env)
        if store:
            self.replay.add_rollout(rollout)
            self.latent_collection.add_rollout(rollout, self.agent.critics[0])
        return rollout

    def collect_softmax_rollout(self, *, eval: bool = False, store: bool = True,
                        env: Optional[FixedLengthEnvWrapper] = None) -> EpisodeData:
        @torch.no_grad()
        def actor(obs: torch.Tensor, goal: torch.Tensor, space: gym.spaces.Space):
            num_actions = env.action_space.n
            actions = torch.tensor([i for i in range(num_actions)]).to(self.device)
            critic_0 = self.agent.critics[0]
            latent_goal = critic_0.encoder(goal.to(self.device))
            action_distances = torch.zeros(num_actions)
            # Iterate over all possible actions
            # Get the latent representation of the next state given the current state and the one-hot encoded action
            latent_state = critic_0.encoder(obs.to(self.device))
            next_latent_states = critic_0.latent_dynamics(latent_state, actions)
            for i in range(num_actions):
                a = actions[i]
                next_state = next_latent_states[i,:]
                d = critic_0.quasimetric_model(next_state, latent_goal)
                action_distances[a] = -d # negative distance

            # Calculate the temperature based on the range of action distances
            temperature = (action_distances.max() - action_distances.min()).item()

            action_probs = torch.nn.functional.softmax(action_distances/temperature, dim=0)
            sample = torch.multinomial(action_probs, 1)
            # print("Softmax probs:", action_probs)
            # print("Softmax action: ", sample)
            return sample.numpy()[0]

        rollout = self.replay.collect_rollout(actor, env=env)
        if store:
            self.replay.add_rollout(rollout)
            self.latent_collection.add_rollout(rollout, self.agent.critics[0])
        return rollout

    def collect_rollout(self, *, eval: bool = False, store: bool = True,
                        env: Optional[FixedLengthEnvWrapper] = None) -> EpisodeData:
        if self.agent.actor is None:
            @torch.no_grad()
            def actor(obs: torch.Tensor, goal: torch.Tensor, space: gym.spaces.Space):
                # Epsilon-greedy action selection
                if not eval and random.random() < self.exploration_eps:
                    num_actions = env.action_space.n
                    actions = torch.tensor([i for i in range(num_actions)])
                    best = random.choice(actions)
                    # print("Random action: ", best)
                else:
                    best = self.greedy_actor(env, obs, goal, space)
                    # print("Greedy action: ", best)
                return best.cpu()
            
        else:
            @torch.no_grad()
            def actor(obs: torch.Tensor, goal: torch.Tensor, space: gym.spaces.Space):
                with self.agent.mode(False):
                    adistn = self.agent.actor(obs[None].to(self.device), goal[None].to(self.device))
                if eval:
                    a = adistn.mode.cpu().numpy()[0]
                else:
                    a_t = adistn.sample()
                    if self.exploration_eps != 0:
                        # FIXME: this only works with [-1, 1] range!  # a hack :)
                        a_t += torch.randn_like(a_t).mul_(self.exploration_eps)
                        a_t.clamp_(-1, 1)
                    a = a_t.cpu().numpy()[0]
                return a

        rollout = self.replay.collect_rollout(actor, env=env)
        if store:
            self.replay.add_rollout(rollout)
            self.latent_collection.add_rollout(rollout, self.agent.critics[0])
        return rollout

    def evaluate(self) -> EvalEpisodeResult:
        env = self.make_evaluate_env()
        rollouts = []
        for _ in tqdm(range(self.num_eval_episodes), desc='evaluate'):
            # Collect rollout in evaluation mode
            rollouts.append(self.collect_rollout(eval=True, store=False, env=env))

        mrollouts = MultiEpisodeData.cat(rollouts)
        return EvalEpisodeResult.from_timestep_reward_is_success(
            mrollouts.rewards.reshape(
                self.num_eval_episodes, env.episode_length,
            ),
            mrollouts.transition_infos['is_success'].reshape(
                self.num_eval_episodes, env.episode_length,
            ),
        )
    
    def novelty_update(self):
        # TODO: how do we handle the 2 critics?
        print("******** novelty_update ********")
        print("LatentCollection size: ", len(self.latent_collection.latent))
        self.latent_collection.reduceCollection(mode = self.downsample)
        self.latent_collection.update(self.agent.critics[0])
        print("LatentCollection size: ", len(self.latent_collection.latent))


    def iter_training_data(self) -> Iterator[Tuple[int, bool, BatchData, InfoT]]:
        r"""
        Yield data to train on for each optimization iteration.

        yield (
            env steps,
            whether this is last yield before collecting new env steps,
            data,
            info,
        )
        """
        def yield_data():
            num_transitions = self.replay.num_transitions_realized
            for icyc in tqdm(range(self.num_samples_per_cycle), desc=f"{num_transitions} env steps, train batches"):
                data_t0 = time.time()
                data = self.sample()
                info = dict(
                    data_time=(time.time() - data_t0),
                    num_episodes=self.replay.num_episodes_realized,
                    num_regular_transitions=self.replay.num_transitions_realized,
                    num_successes=self.replay.num_successful_transitions,
                    replay_capacity=self.replay.episodes_capacity,
                    reward=data.rewards,
                )

                yield num_transitions, (icyc == self.num_samples_per_cycle - 1), data, info

        total_env_steps = self.total_env_steps

        env = self.make_collect_env()  # always make fresh collect env before collecting. GCRL envs don't like reusing.
        for _ in tqdm(range(self.num_prefill_episodes), desc='prefill'):
            self.collect_random_rollout(env=env)
        assert self.replay.num_transitions_realized <= total_env_steps

        yield from yield_data()

        while self.replay.num_transitions_realized < total_env_steps:
            env = self.make_collect_env()
            for _ in tqdm(range(self.num_rollouts_per_cycle), desc='rollout'):
                # TODO: This is where we change the training data to be novel
                if self.agent.actor is not None:
                    self.collect_rollout(env=env)
                
                expl = self.exploration_method
                if expl == 'novel':
                    self.exploration_eps = 1 # always novel
                    self.collect_novel_rollout(env=env)
                elif expl == 'novel_eps_greedy':
                    self.collect_novel_rollout(env=env)
                elif expl == 'eps_greedy':
                    self.collect_rollout(env=env)
                elif expl == 'softmax':
                    self.collect_softmax_rollout(env=env)
                else:
                    raise NotImplementedError(f"exploration_method {expl} not implemented")

                if self.replay.num_transitions_realized >= total_env_steps:
                    break

            yield from yield_data()

    def train_step(self, data: BatchData, *, optimize: bool = True) -> InfoT:
        return self.losses(self.agent, data, optimize=optimize).info
