from copy import deepcopy

import numpy as np
from gymnasium import spaces


ORTH_ACTIONS = [
    0,  # stop
    1,  # up
    2,  # right
    3,  # down
    4,  # left
]


def move(loc, action):
    if action == 'stop' or action == 0:
        return tuple(loc)
    elif action == 'up' or action == 1:
        return tuple(np.add(loc, [-1, 0]))
    elif action == 'right' or action == 2:
        return tuple(np.add(loc, [0, 1]))
    elif action == 'down' or action == 3:
        return tuple(np.add(loc, [1, 0]))
    elif action == 'left' or action == 4:
        return tuple(np.add(loc, [0, -1]))


def get_avai_actions(loc, layout):
    nrows, ncols = layout.shape
    avai_actions = []
    action_mask = np.zeros(len(ORTH_ACTIONS), dtype=np.int8)
    for a in ORTH_ACTIONS:
        succ_loc = move(loc, a)
        if succ_loc[0] not in range(nrows) or succ_loc[1] not in range(ncols):
            continue  # go outside the map
        elif layout[succ_loc] == 1:
            continue  # hit an obstacle
        else:
            avai_actions.append(a)
            action_mask[a] = 1
    return avai_actions, action_mask


def check_collision(prev_locations, curr_locations):
    N = len(prev_locations)
    collisions = [[] for _ in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            collide = False
            if (prev_locations[i], prev_locations[j]) == (curr_locations[j], curr_locations[i]):
                collide = True
            if curr_locations[i] == curr_locations[j]:
                collide = True
            if collide:
                collisions[i].append(j)
                collisions[j].append(i)
    return collisions


STARTS = [(4, 1), (1, 4), (1, 6)]
GOALS = [(2, 3), (6, 1), (6, 4)]
REWARDS = {
    'illegal': -10000,
    'normal': -1,
    'collision': -1000,
    'goal': 10000
}


class MAPF():
    def __init__(self, N, layout,
                 starts=STARTS, goals=GOALS, rewards=REWARDS,
                 obs_fn=None, render_mode='human'):
        self.N = N
        self.agents = []
        self.A = ORTH_ACTIONS

        self.layout = layout
        self.starts = starts
        self.goals = goals
        self.REWARDS = REWARDS
        self.MAX_NUM_STEP = np.product(layout.shape)

        self.obs_fn = obs_fn  # (state, agent_i) -> obs_i
        self.render_mode = render_mode

    def _reset(self, seed=None, options=None):
        self.agents = [f"robot_{i}" for i in range(self.N)]
        self.action_spaces = {
            agent: spaces.Discrete(len(self.A))
            for agent in self.agents
        }
        self.observation_spaces = {
            agent: spaces.MultiDiscrete([*self.N * self.layout.shape])
            for agent in self.agents
        }
        if self.obs_fn is None:
            def self_loc_first(locations, agent):
                i = self.agents.index(agent)
                loc = locations[i]
                rearranged = [loc] + locations[:i] + locations[i + 1:]
                return np.array(rearranged).reshape(-1)
            self.obs_fn = self_loc_first

        observations = {}
        infos = {}
        for i, agent in enumerate(self.agents):
            observations[agent] = self.obs_fn(self.starts, agent)
            infos[agent] = {
                'action_mask': get_avai_actions(self.starts[i], self.layout)[1],
                'collide_with': [],  # TODO: no collision check initially
            }

        self.locations = self.starts
        self.step_cnt = 0
        self.obs_n = deepcopy(observations)
        self.info_n = deepcopy(infos)
        self.terminations = {agent: False for agent in self.agents}
        self.history = {
            'paths': [self.starts]
        }

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
            if not self.terminations[agent] and self.locations[i] == self.goals[i]:
                self.terminations[agent] = True
                rewards[agent] = self.REWARDS['goal']
        terminations = deepcopy(self.terminations)

        self.step_cnt += 1
        if self.step_cnt >= self.MAX_NUM_STEP:
            truncations = {agent: True for agent in self.agents}
        else:
            truncations = {agent: False for agent in self.agents}

        if succ_locations == self.goals or self.step_cnt >= self.MAX_NUM_STEP:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def _render(self):
        if self.render_mode == 'human':
            from marp.animator import Animation
            paths = []
            for step in self.history['paths']:
                paths.append(step)
            self.animator = Animation(
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

    def _save(self, filename, speed=1):
        self.animator.save(f"figs/{filename}", speed)

    def _get_state(self):
        # summarize the information state
        state = {
            'locations': deepcopy(self.locations),
            'infos': deepcopy(self.info_n),
            'goals': deepcopy(self.goals),
        }
        return state

    def _get_observation(self, agent=None):
        # only the current observation
        pass

    def _transit(self, state, actions):
        if np.all(self._is_goal_state(state)):
            state, True

        locations = state['locations']
        infos = state['infos']

        succ_locations = []
        for i, agent in enumerate(self.agents):
            _a = actions[agent]
            if not infos[agent]['action_mask'][_a]:
                _a = 'stop'
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
            'goals': state['goals'],
        }

        return succ_state, collision_free

    def _is_goal_state(self, state):
        ret = []
        for i, loc in enumerate(state['locations']):
            ret.append(loc == state['goals'][i])
        return ret

    def _observation_space(self, agent):
        return self.observation_spaces[agent]

    def _action_space(self, agent):
        return self.action_spaces[agent]
