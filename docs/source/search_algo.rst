.. automodule:: marp.search

Search
======

As we mentioned in the :doc:`quick_start`, in addition to `RL-related` interfaces,
we have also provide with `search-related` interfaces.

The search problem underneath the environment is to find collision-free paths for each agent,
a path for an agent is a sequence of actions from her initial location to her designated goal location.
To formulate a search problem for each agent,
we again follow the :ref:`formulation` principle by encoding wrappers.


Path-Finding for Individual Agents
----------------------------------

Suppose a MARP environment has been intialized, one can induce a single-agent search problem by ingoring others.

.. code-block:: python

    env.reset()
    from marp.search import SingleAgentSearchWrapper, astar
    
    plans = {}
    for agent in env.agents:
        plans[agent] = astar(SingleAgentSearchWrapper(env, agent))
    
    while env.agents:
        actions = {}
        for agent in env.agents:
            path = plans[agent]
            if len(path):
                actions[agent] = path.pop(0)
            else:
                actions[agent] = 0
        observations, rewards, terminations, truncations, infos = env.step(actions)

The above code has done the following things:

1. Formulate a single-agent search problem for each agent by the wrapper :py:class:`SingleAgentSearchWrapper`
2. Call A star search algorithm :py:func:`astar` to compute a plan (path) for each agent.
3. Execute the plans.
   
One will probably get similar visual results as follows. Note that there might be collisions (agents turning red). 

.. figure:: ../../figs/sapf.gif
	:scale: 50%
	:align: center


Joint Path-Finding
------------------

To deal with collisions explicitly, one can instead formulate a search problem over joint states and joint actions, i.e., view all agents as one single joint agent.

.. code-block:: python

    env.reset()
    from marp.search import MultiAgentSearchWrapper, astar

    joint_plan = astar(MultiAgentSearchWrapper(env))

    while env.agents:
        actions = joint_plan.pop(0)
        env.step(actions)
    env.render()

The above code has down the following things:

1. Formulate a joint search problem by the wrapper :py:class:`MultiAgentSearchWrapper`, where a state with any collision is specified as a dead end.
2. Again, call A star search :py:func:`astar`, but this time the algorithm will return a sequence of joint-actions
3. Execute the joint-plan.

One will probably get similar visual results as follows. Now there will not be collisions anymore.

.. figure:: ../../figs/sapf.gif
	:scale: 50%
	:align: center


Detailed Usage
--------------

Functions
^^^^^^^^^

.. autofunction:: astar


Classes
^^^^^^^

.. autoclass:: SingleAgentSearchWrapper
	:members:

.. autoclass:: MultiAgentSearchWrapper
	:members:
