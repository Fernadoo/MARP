from copy import deepcopy

from pettingzoo import ParallelEnv

from marp.utils import parse_map_from_file


class MARP(ParallelEnv):
    """
    The base MARP environment that unifies the APIs of the supporting tasks

    Args:
        N (int): the number of agents
        layout (str): the file name of the layout configuration
        one_shot (bool): one-shot path finding or lifelong
        render_mode (str or None): will visualize if 'human', otherwise, only print in the console
    """
    metadata = {
        "name": "multiagent_route_planning_v0",
    }

    def __init__(self, N=2, layout='small', orthogonal_actions=True, one_shot=True, battery=False, render_mode=None):
        layout = parse_map_from_file(layout)
        if orthogonal_actions:
            if one_shot:
                from marp.mapf import MAPF
                self.world = MAPF(N, layout, render_mode=render_mode)
            else:
                if not battery:
                    from marp.mapd import MAPD
                    self.world = MAPD(N, layout, render_mode=render_mode)
                else:
                    from marp.warehouse import Warehouse
                    self.world = Warehouse(N, layout, render_mode=render_mode)
        else:
            if one_shot:
                from marp.mapf import MAPF_R
                self.world = MAPF_R(N, layout, render_mode=render_mode)
            else:
                if not battery:
                    from marp.mapd import MAPD_R
                    self.world = MAPD_R(N, layout, render_mode=render_mode)
                else:
                    from marp.warehouse import Warehouse_R
                    self.world = Warehouse_R(N, layout, render_mode=render_mode)

    def reset(self, seed=None, options=None):
        """
        Reset agent loations
        """
        observations, infos = self.world._reset(seed, options)
        self.agents = self.world.agents
        self.possible_agents = deepcopy(self.agents)
        return observations, infos

    def step(self, actions):
        """
        Proceed to the next step by the given action profile according to the underlying specific task.

        Args:
            actions (dict[str, Action]): the action profile, i.e., joint actions.

        Returns:
            observations (dict): observation profile
            rewards (dict): rewards for each agent
            terminations (dict): whether the task is accomplished for each agent
            truncations (dict): True if timeout, otherwise False
            infos (dict): auxiliary infomation including collisions and action masks
        """
        observations, rewards, terminations, truncations, infos = self.world._step(actions)
        self.agents = self.world.agents
        return observations, rewards, terminations, truncations, infos

    def render(self):
        """
        Render the history
        """
        return self.world._render()

    def save(self, file_name, speed=1):
        """
        Save the visualized result

        .. note::

            Run ``conda install conda-forge::ffmpeg`` first, if ``ffmpeg`` is not installed

        Args:
            file_name (str): output file path
            speed (int): speedup rate
        """
        self.world._save(file_name, speed)

    def get_state(self):
        """
        Summarize the current system state
        """
        return self.world._get_state()

    def get_observation(self, agent=None):
        pass

    def transit(self, state, actions):
        """
        Given a state and an action profile, return the successor state.

        .. note::
            Calls to this method will not change any internal state of the environment

        Args:
            state (dict): system state obtained by enquiring :py:func:`get_state`
            actions (dict[str, Action]): an action profile

        Returns:
            succ_state (dict): the successor state
        """
        return self.world._transit(state, actions)

    def is_goal_state(self, state):
        """
        Check if each agent has reached her designated goal.
        """
        return self.world._is_goal_state(state)

    def observation_space(self, agent):
        return self.world._observation_space(agent)

    def action_space(self, agent):
        return self.world._action_space(agent)
