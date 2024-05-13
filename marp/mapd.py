from copy import deepcopy

import numpy as np

from marp.mapf import MAPF, move, get_avai_actions, check_collision
from marp.mapf import MAPF_R, r_move, r_get_avai_actions


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


class MAPD(MAPF):
    """docstring for MAPD"""

    def __init__(self, N, layout,
                 starts=STARTS, goals=GOALS, rewards=REWARDS,
                 obs_fn=None, render_mode='human'):
        super().__init__(N, layout,
                         starts, goals, rewards,
                         obs_fn, render_mode)
        max_len = max(list(map(lambda x: len(x), goals)))
        self.MAX_NUM_STEP = np.product(layout.shape) * max_len

    def _reset(self, seed=None, options=None):
        observations, infos = super()._reset(seed, options)
        self.next_goals = {agent: 0 for agent in self.agents}
        return observations, infos

    def _step(self, actions):
        succ_locations = []
        rewards = {agent: self.REWARDS['normal'] for agent in self.agents}
        for i, agent in enumerate(self.agents):
            _a = actions[agent]
            if self.terminations[agent]:
                _a = 'stop'
                rewards[agent] = self.REWARDS['goal']
            elif not self.info_n[agent]['action_mask'][_a]:
                _a = 'stop'
                rewards[agent] = self.REWARDS['illegal']
            succ_locations.append(move(self.locations[i], _a))

        collisions = check_collision(self.locations, succ_locations)
        self.locations = succ_locations
        self.history['paths'].append(succ_locations)

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
            from marp.animator import StreamAnimation
            paths = []
            for step in self.history['paths']:
                paths.append(step)
            self.animator = StreamAnimation(
                range(self.N),
                self.layout,
                self.starts,
                self.goals,
                paths,
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
        }
        return state

    def _transit(self, state, actions):
        if np.all(self._is_goal_state(state)):
            state, True

        locations = state['locations']
        infos = state['infos']
        goals = state['goals']
        next_goals = state['next_goals']

        succ_locations = []
        succ_next_goals = deepcopy(next_goals)
        for i, agent in enumerate(self.agents):
            _a = actions[agent]
            if not infos[agent]['action_mask'][_a]:
                _a = 'stop'
            succ_loc = move(locations[i], _a)
            if succ_loc == goals[i][next_goals[agent]]:
                succ_next_goals[agent] = min(next_goals[agent] + 1, len(goals[i]) - 1)
            succ_locations.append(move(locations[i], _a))

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
        }

        return succ_state, collision_free

    def _is_goal_state(self, state):
        ret = []
        locations = state['locations']
        goals = state['goals']
        next_goals = state['next_goals']

        for i, agent in enumerate(self.agents):
            if next_goals[agent] != len(goals[i]) - 1:
                ret.append(False)
            else:
                if locations[i] == goals[i][-1]:
                    ret.append(True)
                else:
                    ret.append(False)
        return ret


class MAPD_R(MAPF_R):
    """docstring for MAPD_R"""

    def __init__(self, N, layout,
                 starts=STARTS, goals=GOALS, rewards=REWARDS,
                 obs_fn=None, render_mode='human'):
        super().__init__(N, layout,
                         starts, goals, rewards,
                         obs_fn, render_mode)
        max_len = max(list(map(lambda x: len(x), goals)))
        self.MAX_NUM_STEP = np.product(layout.shape) * max_len

    def _reset(self, seed=None, options=None):
        observations, infos = super()._reset(seed, options)
        self.next_goals = {agent: 0 for agent in self.agents}
        return observations, infos

    def _step(self, actions):
        succ_locations = []
        succ_directions = []
        rewards = {agent: self.REWARDS['normal'] for agent in self.agents}
        for i, agent in enumerate(self.agents):
            _a = actions[agent]
            if self.terminations[agent]:
                _a = 'stop'
                rewards[agent] = self.REWARDS['goal']
            elif not self.info_n[agent]['action_mask'][_a]:
                _a = 'stop'
                rewards[agent] = self.REWARDS['illegal']
            succ_loc, succ_drct = r_move((self.locations[i], self.directions[i]), _a)
            succ_locations.append(succ_loc)
            succ_directions.append(succ_drct)

        collisions = check_collision(self.locations, succ_locations)
        self.locations = succ_locations
        self.directions = succ_directions
        self.history['paths'].append(succ_locations)
        self.history['directions'].append(succ_directions)

        observations = {}
        infos = {}
        for i, agent in enumerate(self.agents):
            c_i = collisions[i]
            if c_i and not self.terminations[agent]:
                # TODO: incur collision penalty even if the goal is reached in this step
                rewards[agent] = self.REWARDS['collision']
            observations[agent] = self.obs_fn(succ_locations, succ_directions, agent)
            infos[agent] = {
                'action_mask': r_get_avai_actions((succ_locations[i], succ_directions[i]),
                                                  self.layout)[1],
                'collide_with': c_i,
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
            from marp.animator import StreamAnimation_R
            self.animator = StreamAnimation_R(
                range(self.N),
                self.layout,
                self.starts,
                self.goals,
                self.history['paths'],
                self.history['directions'],
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
            'directions': deepcopy(self.directions),
            'infos': deepcopy(self.info_n),
            'goals': deepcopy(self.goals),
            'next_goals': deepcopy(self.next_goals),
        }
        return state

    def _transit(self, state, actions):
        if np.all(self._is_goal_state(state)):
            state, True

        locations = state['locations']
        directions = state['directions']
        infos = state['infos']
        goals = state['goals']
        next_goals = state['next_goals']

        succ_locations = []
        succ_directions = []
        succ_next_goals = deepcopy(next_goals)
        for i, agent in enumerate(self.agents):
            _a = actions[agent]
            if not infos[agent]['action_mask'][_a]:
                _a = 'stop'
            succ_loc, succ_drct = r_move((locations[i], directions[i]), _a)
            if succ_loc == goals[i][next_goals[agent]]:
                succ_next_goals[agent] = min(next_goals[agent] + 1, len(goals[i]) - 1)
            succ_locations.append(succ_loc)
            succ_directions.append(succ_drct)

        collision_free = True
        succ_infos = {}
        collisions = check_collision(locations, succ_locations)
        for i, agent in enumerate(self.agents):
            c_i = collisions[i]
            if c_i:
                collision_free = False
            succ_infos[agent] = {
                'action_mask': r_get_avai_actions((succ_locations[i], succ_directions[i]),
                                                  self.layout)[1],
                'collide_with': c_i,
            }

        succ_state = {
            'locations': succ_locations,
            'directions': succ_directions,
            'infos': succ_infos,
            'goals': goals,
            'next_goals': succ_next_goals,
        }

        return succ_state, collision_free

    def _is_goal_state(self, state):
        ret = []
        locations = state['locations']
        goals = state['goals']
        next_goals = state['next_goals']

        for i, agent in enumerate(self.agents):
            if next_goals[agent] != len(goals[i]) - 1:
                ret.append(False)
            else:
                if locations[i] == goals[i][-1]:
                    ret.append(True)
                else:
                    ret.append(False)
        return ret
