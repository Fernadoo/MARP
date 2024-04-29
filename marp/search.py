import itertools
import random
from collections import namedtuple
from queue import PriorityQueue

import numpy as np


class SingleAgentSearchWrapper():
    """
    Given a MARP environment, wrap it as inducing a single-agent search problem.

    :param ma_env: a multi-agent environment with the transition function provided.
    :type ma_env: MARP
    :param agent: the agent that the problem is induced for
    :type agent: str
    """

    def __init__(self, ma_env, agent):
        self.ma_env = ma_env
        self.agent = agent
        self.i = self.ma_env.agents.index(agent)
        self.A = tuple(range(self.ma_env.action_space(agent).n))  # assume discrete env
        self.init_state = self.ma_env.get_state()

    def get_state(self):
        """
        Obtain the underlying system state.

        :return: the underlying state
        :rtype: dict
        """
        return self.ma_env.get_state()

    def transit(self, state, action):
        """
        Given a state and an action, return the successor state and the cost.

        :param state: the state to enquire
        :type state: dict
        :param action: the action to enquire
        :type action: Action
        :return: (the successor state, the cost)
        :rtype: (dict, float)
        """
        fake_actions = {
            agent: 0 if agent != self.agent else action
            for agent in self.ma_env.agents
        }
        return self.ma_env.transit(state, fake_actions)[0], 1

    def is_goal_state(self, state):
        """
        Check whether it is a goal state

        :param state: the state to enquire
        :type state: dict
        :return: whether it is a goal state
        :rtype: bool
        """
        return self.ma_env.is_goal_state(state)[self.i]

    def heuristic(self, state):
        """
        A domain dependent heuristic

        :param state: the state to enquire
        :type state: dict
        :return: the Euclidean distance from the current state to the goal state
        :rtype: float
        """
        loc = state['locations'][self.i]
        goal = state['goals'][self.i]
        return np.sqrt(np.sum(np.square(np.array(loc) - np.array(goal))))


class MultiAgentSearchWrapper():
    """
    Given a MARP environment, wrap it as a multi-agent joint search problem.

    :param ma_env: a multi-agent environment with the transition function provided.
    :type ma_env: MARP
    """

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
        """
        Obtain the joint state.

        :return: (system state, collision-free or not)
        :rtype: (dict, bool)
        """
        return self.ma_env.get_state(), True

    def transit(self, state, actions):
        """
        Given a joint state and a joint action, return the successor joint state and the aggreated cost.

        :param state: the joint state to enquire
        :type state: dict
        :param action: the joint action to enquire
        :type action: dict[str, Actions]
        :return: (the successor state, the cost)
        :rtype: (dict, cost)
        """
        raw_state, is_legal = state
        goal_reached = self.ma_env.is_goal_state(raw_state)
        cost = 0
        for reached in goal_reached:
            if not reached:
                cost += 1
        return self.ma_env.transit(raw_state, actions), cost

    def is_goal_state(self, state):
        """
        Check whether it is a goal state

        :param state: the joint state to enquire
        :type state: dict
        :return: whether it is a goal state
        :rtype: bool
        """
        raw_state, is_legal = state
        return np.all(self.ma_env.is_goal_state(raw_state))

    def heuristic(self, state):
        """
        A domain dependent heuristic:
        return 99999 if any collision, otherwise the sum of all agents' costs.

        :param state: the joint state to enquire
        :type state: dict
        :return: the sum of the Euclidean distance of all agents' locations to their goals
        :rtype: float
        """
        raw_state, is_legal = state
        if not is_legal:
            return 99999
        loc = raw_state['locations']
        goal = raw_state['goals']
        return np.sum(np.sqrt(np.sum(np.square(np.array(loc) - np.array(goal)), axis=1)))


def astar(env):
    """
    Given a search environment, return the optimal plan.

    :param env: a search environment with the transition function provided.
    :type env: SearchEnv
    :raise RuntimeError: If no plan can be found.
    :return: The optimal plan.
    :rtype: list[Action]
    """
    Node = namedtuple('ANode',
                      ['fValue', 'gValue', 'PrevAction', 'State'])

    def get_successors(node):
        f, g, prev_action, curr_state = node
        successors = []
        for a in env.A:
            succ_state, cost = env.transit(curr_state, a)
            heu = env.heuristic(succ_state)
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
    init = env.init_state
    q.put(Node(env.heuristic(init), 0, None, init))

    while not q.empty():
        curr_node = q.get()

        if env.is_goal_state(curr_node.State):
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
