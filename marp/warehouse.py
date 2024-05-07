import random
from copy import deepcopy

import numpy as np

from marp.mapf import move, get_avai_actions, check_collision
from marp.mapd import MAPD


STARTS = [(4, 1), (1, 4), (1, 6)]
GOALS = [
    [(6, 6), ],
    [(4, 4), (4, 1), (3, 6), ],
    [(3, 2), (6, 2), ],
]
REWARDS = {
    'illegal': -10000,
    'normal': -1,
    'collision': -1000,
    'goal': 10000
}
BATTERY = 15
CONTINGENCY = 0.0


class Warehouse(MAPD):
    """docstring for Warehouse"""

    def __init__(self, N, layout,
                 starts=STARTS, goals=GOALS, rewards=REWARDS,
                 battery=BATTERY, contingency_rate=CONTINGENCY,
                 obs_fn=None, render_mode='human'):
        super().__init__(N, layout,
                         starts, goals, rewards,
                         obs_fn, render_mode)
        self.MAX_NUM_STEP = 30
        self.full_battery = battery
        self.contingency_rate = contingency_rate

    def _reset(self, seed=None, options=None):
        observations, infos = super()._reset(seed, options)
        self.batteries = {agent: self.full_battery for agent in self.agents}
        for agent in self.agents:
            infos[agent]['battery'] = self.full_battery
        self.history = {
            'paths': [self.starts],
            'batteries': [tuple(self.full_battery for agent in self.agents)]
        }
        return observations, infos

    def _step(self, actions):
        succ_locations = []
        if random.random() < self.contingency_rate:
            actions = {
                agent: self._action_space(agent).sample() for agent in self.agents
            }
        rewards = {agent: self.REWARDS['normal'] for agent in self.agents}
        for i, agent in enumerate(self.agents):
            _a = actions[agent]
            if self.batteries[agent] <= 0:
                _a = 'stop'
                self.batteries[agent] += 1  # restore
            elif self.terminations[agent]:
                _a = 'stop'
                rewards[agent] = self.REWARDS['goal']
                self.batteries[agent] += 1  # restore
            elif not self.info_n[agent]['action_mask'][_a]:
                _a = 'stop'
                rewards[agent] = self.REWARDS['illegal']
            succ_loc = move(self.locations[i], _a)
            succ_locations.append(succ_loc)
            self.batteries[agent] -= 1

            if self.layout[succ_loc] == 8:
                self.batteries[agent] = self.full_battery

        collisions = check_collision(self.locations, succ_locations)
        self.locations = succ_locations
        self.history['paths'].append(succ_locations)
        self.history['batteries'].append(tuple(self.batteries[agent] for agent in self.agents))

        observations = {}
        infos = {}
        for i, agent in enumerate(self.agents):
            c_i = collisions[i]
            if c_i and not self.terminations[agent]:
                # TODO: incur collision penalty even if the goal is reached in this step
                rewards[agent] = self.REWARDS['collision']
            observations[agent] = self.obs_fn(succ_locations, agent)
            infos[agent] = {
                'action_mask': get_avai_actions(succ_locations[i], self.layout)[1],
                'collide_with': c_i,
                'battery': self.batteries[agent]
            }
        self.obs_n = deepcopy(observations)
        self.info_n = deepcopy(infos)

        for i, agent in enumerate(self.agents):
            if self.locations[i] == self.goals[i][self.next_goals[agent]]:
                rewards[agent] = self.REWARDS['goal']
                if not self.terminations[agent] and self.next_goals[agent] == len(self.goals[i]) - 1:
                    self.terminations[agent] = True
                self.next_goals[agent] = min(self.next_goals[agent] + 1, len(self.goals[i]) - 1)

        terminations = deepcopy(self.terminations)

        self.step_cnt += 1
        if self.step_cnt >= self.MAX_NUM_STEP:
            truncations = {agent: True for agent in self.agents}
        else:
            truncations = {agent: False for agent in self.agents}

        if np.all(list(terminations.values())) or self.step_cnt >= self.MAX_NUM_STEP:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def _render(self):
        if self.render_mode == 'human':
            from marp.animator import WarehouseAnimation
            paths = []
            aux = {}
            for _field in self.history:
                if _field == 'paths':
                    for step in self.history['paths']:
                        paths.append(step)
                else:
                    aux[_field] = []
                    for step in self.history[_field]:
                        aux[_field].append(step)
            self.animator = WarehouseAnimation(
                range(self.N),
                self.layout,
                self.starts,
                self.goals,
                paths,
                aux,
                FPS=60
            )
            self.animator.show()
        else:
            for step in self.history['paths']:
                print(step)

    def _get_state(self):
        # summarize the information state
        state = {
            'locations': deepcopy(self.locations),
            'infos': deepcopy(self.info_n),
            'goals': deepcopy(self.goals),
            'next_goals': deepcopy(self.next_goals),
            'batteries': deepcopy(self.batteries),
        }
        return state

    def _transit(self, state, actions):
        if np.all(self._is_goal_state(state)):
            state, True

        locations = state['locations']
        infos = state['infos']
        goals = state['goals']
        next_goals = state['next_goals']
        batteries = state['batteries']

        succ_locations = []
        succ_next_goals = deepcopy(next_goals)
        succ_batteries = deepcopy(batteries)
        for i, agent in enumerate(self.agents):
            _a = actions[agent]
            succ_batteries[agent] = batteries[agent] - 1
            if not infos[agent]['action_mask'][_a]:
                _a = 'stop'
            succ_loc = move(locations[i], _a)
            if succ_loc == goals[i][next_goals[agent]]:
                succ_next_goals[agent] = min(next_goals[agent] + 1, len(goals[i]) - 1)
            succ_locations.append(move(locations[i], _a))

            if self.layout[succ_loc] == 8:
                succ_batteries[agent] = self.full_battery

        collision_free = True
        succ_infos = {}
        collisions = check_collision(locations, succ_locations)
        for i, agent in enumerate(self.agents):
            c_i = collisions[i]
            if c_i:
                collision_free = False
            succ_infos[agent] = {
                'action_mask': get_avai_actions(succ_locations[i], self.layout)[1],
                'collide_with': c_i,
            }

        succ_state = {
            'locations': succ_locations,
            'infos': succ_infos,
            'goals': goals,
            'next_goals': succ_next_goals,
            'batteries': succ_batteries,
        }

        return succ_state, collision_free
