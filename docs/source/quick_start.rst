
Quick Start
===========

Initializing environments in MARP is very similar to
doing that in `PettingZoo <https://pettingzoo.farama.org/>`_ and `Gym <https://gymnasium.farama.org/>`_.


.. code-block:: python

   from marp.ma_env import MARP
   env = MARP(N=3, layout='small', orthogonal_actions=True, one_shot=True, render_mode='human')

This creates a multi-agent environment where each agent takes actions simultaneously.
In other words, every time the environment takes as input an action profile (i.e., an joint-action)
and proceeds to the next step. We provide similar interfaces as PettingZoo

.. code-block:: python

   from marp.ma_env import MARP

   env = MARP(N=3, layout='small', orthogonal_actions=True, one_shot=True, render_mode='human')
   observations, infos = env.reset()
   
   while env.agents:
       actions = {
           agent: env.action_space(agent).sample(infos[agent]['action_mask'])
           for agent in env.agents
       }
       observations, rewards, terminations, truncations, infos = env.step(actions)


In addition to the conventional ``step()`` interface that is commonly used in the RL community,
we also provide interfaces that help obtain the explicit transition between (global or system) states.

.. code-block:: python

   from marp.ma_env import MARP

   env = MARP(N=3, layout='small', orthogonal_actions=True, one_shot=True, render_mode='human')
   env.reset()

   curr_state = env.get_state()
   actions = {
           agent: env.action_space(agent).sample(infos[agent]['action_mask'])
           for agent in env.agents
   }
   succ_state = env.transit(curr_state, actions)

Compared to the ``step()`` interface, the ``transit()`` interface explicitly takes in a state,
which can be aquired by ``get_state()`` in advance, and an action profile, and returns a successor state.
Note that, calls to this function will not change the internal state of the environment,
therefore, can be used to implement search algorithms that plan ahead.

