Basic Usage
===========


.. currentmodule:: marp.ma_env

.. autoclass:: MARP
	:members: reset, step, render, get_state, transit, is_goal_state, save

.. _formulation:


Formulation as Wrappers
-----------------------

By implementing a base environment, we provide standard interfaces and the minimal set of infomations needed.
It only provides a multi-agent simulation environment
but does not restrict the exact problem that one may want to solve.
We hereby claim one possible principle **formulation as wrappers**.
That is, a downstream problem can be simulated and investigated
by implementing an appropriate light-weight wrapper out of the basic :py:class:`MARP` environment.
For example, if one wants to simulate and solve a centralized multi-agent search problem,
then she can have a customized formulation wrapper as follows

.. code-block:: python

	class CustomizedFormulationWrapper():

	    def __init__(self, ma_env, options=None):
	        self.ma_env = ma_env
	        self.options = options

	    def get_state(self):
	        """
	        State enquiry
	        """

	    def transit(self, state, action):
	        """
	        Returns the successor state and the associated cost
	        """

	    def is_goal_state(self, state):
	        """
	        Check whether it is a goal state
	        """

	    def heuristic(self, state):
	        """
	        A domain dependent heuristic
	            """
