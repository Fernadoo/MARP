from copy import deepcopy

from pettingzoo import ParallelEnv

from marp.utils import parse_map_from_file


class MARP(ParallelEnv):
    metadata = {
        "name": "multiagent_route_planning_v0",
    }

    def __init__(self, N=2, layout='small', orthogonal_actions=True, one_shot=True, render_mode=None):
        layout = parse_map_from_file(layout)
        if orthogonal_actions:
            from marp.marp_orth import MARPOrth
            self.world = MARPOrth(N, layout, one_shot, render_mode=render_mode)
        else:
            from marp.marp_spin import MARPSpin
            self.world = MARPSpin(N, layout, one_shot)

    def reset(self, seed=None, options=None):
        observations, infos = self.world._reset(seed, options)
        self.agents = self.world.agents
        self.possible_agents = deepcopy(self.agents)
        return observations, infos

    def step(self, actions):
        observations, rewards, terminations, truncations, infos = self.world._step(actions)
        self.agents = self.world.agents
        return observations, rewards, terminations, truncations, infos

    def render(self):
        return self.world._render()

    def get_state(self):
        return self.world._get_state()

    def get_observation(self, agent=None):
        pass

    def transit(self, state, actions):
        return self.world._transit(state, actions)

    def is_goal_state(self, state):
        return self.world._is_goal_state(state)

    def observation_space(self, agent):
        return self.world._observation_space(agent)

    def action_space(self, agent):
        return self.world._action_space(agent)
