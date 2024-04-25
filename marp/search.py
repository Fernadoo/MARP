import itertools
import random
from collections import namedtuple
from queue import PriorityQueue

import numpy as np


class SingleAgentSearchWrapper():
    """docstring for SingleAgentSearchWrapper
    induce a single agent search problem for one single agent
    """

    def __init__(self, ma_env, agent):
        self.ma_env = ma_env
        self.agent = agent
        self.i = self.ma_env.agents.index(agent)
        self.A = tuple(range(self.ma_env.action_space(agent).n))  # assume discrete env
        self.init_state = self.ma_env.get_state()

    def get_state(self):
        return self.ma_env.get_state()

    def transit(self, state, action):
        fake_actions = {
            agent: 0 if agent != self.agent else action
            for agent in self.ma_env.agents
        }
        return self.ma_env.transit(state, fake_actions)[0], 1

    def is_goal_state(self, state):
        return self.ma_env.is_goal_state(state)[self.i]

    def heuristic(self, state):
        """A domain dependent heuristic"""
        loc = state['locations'][self.i]
        goal = state['goals'][self.i]
        return np.sqrt(np.sum(np.square(np.array(loc) - np.array(goal))))


class MultiAgentSearchWrapper():
    """docstring for MultiAgentSearchWrapper"""

    def __init__(self, ma_env):
        self.ma_env = ma_env
        self.agents = ma_env.agents
        joint_actions = (range(self.ma_env.action_space(agent).n) for agent in ma_env.agents)
        self.A = tuple((
            dict(zip(ma_env.agents, actions)) for actions in itertools.product(*joint_actions)
        ))
        self.init_state = (self.ma_env.get_state(), True)
        # self.ma_env.world.goals = [(3, 1), (1, 5), (2, 6)]

    def get_state(self):
        return self.ma_env.get_state(), True

    def transit(self, state, actions):
        raw_state, is_legal = state
        goal_reached = self.ma_env.is_goal_state(raw_state)
        cost = 0
        for reached in goal_reached:
            if not reached:
                cost += 1
        return self.ma_env.transit(raw_state, actions), cost

    def is_goal_state(self, state):
        raw_state, is_legal = state
        return np.all(self.ma_env.is_goal_state(raw_state))

    def heuristic(self, state):
        """A domain dependent heuristic"""
        raw_state, is_legal = state
        if not is_legal:
            return 99999
        loc = raw_state['locations']
        goal = raw_state['goals']
        return np.sum(np.sqrt(np.sum(np.square(np.array(loc) - np.array(goal)), axis=1)))


def astar(sa_env):
    """
    Given a single agent search environment, return the optimal plan.

    :param sa_env: a single agent environment the transition function provided.
    :type sa_env: SingleSearchEnv
    :raise RuntimeError: If no plan can be found.
    :return: The optimal plan.
    :rtype: list[int]
    """
    Node = namedtuple('ANode',
                      ['fValue', 'gValue', 'PrevAction', 'State'])

    def get_successors(node):
        f, g, prev_action, curr_state = node
        successors = []
        for a in sa_env.A:
            succ_state, cost = sa_env.transit(curr_state, a)
            heu = sa_env.heuristic(succ_state)
            if heu > 9999:
                continue
            tie_break_noise = random.uniform(0, 1e-2)
            succ_node = Node(heu + g + cost, g + cost + tie_break_noise, a, succ_state)
            successors.append(succ_node)
        return successors

    plan = []
    visited = []
    parent_dict = dict()
    q = PriorityQueue()
    init = sa_env.init_state
    q.put(Node(sa_env.heuristic(init), 0, None, init))

    while not q.empty():
        curr_node = q.get()

        if sa_env.is_goal_state(curr_node.State):
            # backtrack to get the plan
            curr = curr_node
            while curr.State != init:
                plan.insert(0, curr.PrevAction)
                curr = parent_dict[str(curr)]
            return plan

        if str(curr_node.State) in visited:
            continue

        successors = get_successors(curr_node)
        for succ_node in successors:
            q.put(succ_node)
            parent_dict[str(succ_node)] = curr_node
        visited.append(str(curr_node.State))
    raise RuntimeError("No astar plan found!")
