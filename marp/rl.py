import time
import random
from itertools import product

import numpy as np
import supersuit as ss
from gymnasium import spaces, Env
from stable_baselines3 import PPO, A2C, DQN
from tqdm import tqdm


class SingleAgentLearningWrapper(Env):
    """
    Formulate a single agent learning problem

    Args:
        ma_env (MARP): an intialized multi-agent environment
        agent (str): the agent the the problem is induced for
    """

    metadata = {
        'name': 'MA-Single-RL'
    }

    def __init__(self, ma_env, agent):
        self.ma_env = ma_env
        self.agent = agent
        self.action_space = self.ma_env.action_space(agent)
        self.observation_space = self.ma_env.observation_space(agent)

    def reset(self, seed=None, options=None):
        """
        Reset the location of the agent
        """
        obs_n, info_n = self.ma_env.reset()
        return obs_n[self.agent], info_n[self.agent]

    def step(self, action):
        """
        Proceed to the next step by the given action

        Args:
            action (Action): the next action

        Returns:
            obs (dict): local observation
            reward (float): reward
            termination (bool): whether the episode terminates
            truncation (bool): whether the maximum number of steps is exceeded
            info (dict): auxiliary infomation including collision situations and action masks
        """
        fake_actions = {
            agent: 0 if agent != self.agent else action
            for agent in self.ma_env.agents
        }
        obs_n, r_n, term_n, trunc_n, info_n = self.ma_env.step(fake_actions)

        obs = obs_n[self.agent]
        r = r_n[self.agent]
        term = term_n[self.agent]
        trunc = trunc_n[self.agent]

        return obs, r, term, trunc, info_n


class MultiAgentJointLearningWrapper(Env):
    """
    Formulate a multi-agent joint learning problem

    Args:
        ma_env (MARP): an intialized multi-agent environment
    """

    metadata = {
        'name': 'MA-Joint-RL'
    }

    def __init__(self, ma_env):
        self.ma_env = ma_env
        self.agents = self.ma_env.agents
        self.action_space = spaces.Discrete(
            np.prod([self.ma_env.action_space(agent).n for agent in self.agents])
        )
        self.joint_actions = tuple(product(
            *[range(self.ma_env.action_space(agent).n) for agent in self.agents]
        ))
        self.observation_space = self.ma_env.observation_space(self.agents[0])

    def reset(self, seed=None, options=None):
        """
        Reset the locations of agents
        """
        obs_n, info_n = self.ma_env.reset()
        obs = obs_n[self.agents[0]]
        return obs, info_n

    def step(self, action):
        """
        Proceed to the next step by the given joint action

        Args:
            action (Action): the next joint action

        Returns:
            obs (dict): joint observations
            reward (float): aggregated reward by simple summation
            termination (bool): whether the episode terminates for all agents
            truncation (bool): whether the maximum number of steps is exceeded
            info (dict): auxiliary infomation including collision situations and action masks
        """
        action = self.joint_actions[action]
        ja = dict(zip(self.agents, action))
        obs_n, r_n, term_n, trunc_n, info_n = self.ma_env.step(ja)

        obs = obs_n[self.agents[0]]
        r = np.sum(np.array([r_n[agent] for agent in self.agents]))
        term = np.all(np.array([term_n[agent] for agent in self.agents]))
        trunc = np.all(np.array([trunc_n[agent] for agent in self.agents]))

        return obs, r, term, trunc, info_n


def MultiAgentIndividualLearningWrapper(ma_env):
    training_env = ss.pettingzoo_env_to_vec_env_v1(ma_env)
    training_env = ss.concat_vec_envs_v1(training_env, 8, num_cpus=1, base_class="stable_baselines3")
    training_env.reset()
    return training_env


def Qlearning(env, num_it=1e3, epsilon=0.5, alpha=0.3, gamma=0.9):
    """
    Tabular Q learning

    Args:
        env (RLEnv): a single/joint-agent RL environment
        num_it (int or float): the number of learning iterations
        epsilon (float): the initial exploration rate,
            will linearly decay to 0.1 in the first half of iterations
        alpha (float): learning rate
        gamma (float): discount factor

    Returns:
        policy (dict): the learned policy
    """
    Qs = {}
    for i in tqdm(range(int(num_it))):
        ep = max((epsilon - 0.1) * (0.5 * num_it - i) / num_it, 0) + 0.1
        obs, info = env.reset()
        term, trunc = False, False
        while not (term or trunc):
            if str(obs) not in Qs:
                Qs[str(obs)] = np.zeros(env.action_space.n)

            if random.random() < ep:
                action = random.choice(range(env.action_space.n))
            else:
                action = np.argmax(Qs[str(obs)])
            succ_obs, r, term, trunc, _ = env.step(action)

            # update Qs
            if str(succ_obs) not in Qs:
                Qs[str(succ_obs)] = np.zeros(env.action_space.n)
            Qs[str(obs)][action] = (1 - alpha) * Qs[str(obs)][action] +\
                alpha * (r + gamma * np.max(Qs[str(succ_obs)]))

            obs = succ_obs

    policy = {}
    for state in Qs:
        policy[state] = np.argmax(Qs[state])

    return policy


def individualQlearning(env, num_it=1e3, epsilon=0.5, alpha=0.3, gamma=0.9):
    """
    Tabular Q learning for multiple individual learners

    Args:
        env (MARLEnv): a multi-agent RL environment
        num_it (int or float): the number of learning iterations
        epsilon (float): the initial exploration rate,
            will linearly decay to 0.1 in the first half of iterations
        alpha (float): learning rate
        gamma (float): discount factor

    Returns:
        policies (dict[str, dict]): a policy profile
    """
    Qs = {agent: {} for agent in env.agents}
    for i in tqdm(range(int(num_it))):
        ep = max((epsilon - 0.1) * (0.5 * num_it - i) / num_it, 0) + 0.1
        obs_n, info_n = env.reset()
        while env.agents:
            for agent in obs_n:
                if str(obs_n[agent]) not in Qs[agent]:
                    Qs[agent][str(obs_n[agent])] = np.zeros(env.action_space(agent).n)

            if random.random() < ep:
                actions = {
                    agent: env.action_space(agent).sample()
                    for agent in env.agents
                }
            else:
                actions = {
                    agent: np.argmax(Qs[agent][str(obs_n[agent])])
                    for agent in env.agents
                }
            succ_obs_n, r_n, _, _, _ = env.step(actions)

            # update Qs
            for agent in succ_obs_n:
                if str(succ_obs_n[agent]) not in Qs[agent]:
                    Qs[agent][str(succ_obs_n[agent])] = np.zeros(env.action_space(agent).n)
                Qs[agent][str(obs_n[agent])][actions[agent]] =\
                    (1 - alpha) * Qs[agent][str(obs_n[agent])][actions[agent]] +\
                    alpha * (r_n[agent] + gamma * np.max(Qs[agent][str(succ_obs_n[agent])]))

            obs_n = succ_obs_n

    policy = {}
    for agent in Qs:
        policy[agent] = {}
        for state in Qs[agent]:
            policy[agent][state] = np.argmax(Qs[agent][state])

    return policy
