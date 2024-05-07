Warehouse
=========

**Warehouse** is a real-world application of the general **Multi-Agent Contingent Planning**, where each agent is associated with a battery, and therefore will stop if she runs out of battery.
There is also certain chance an agent will not go as planned,
e.g., accidentally stops or the package gets dropped.

.. |img1| image:: ../../figs/warehouse.gif
	:width: 300px
	:align: middle

.. |img2| image:: ../../figs/warehouse_ctg.gif
	:width: 300px
	:align: middle

.. table::
   :align: center

   +---------+---------+
   |  |img1| |  |img2| |
   +---------+---------+

A more reasonable case is to deploy charging stations so that
agents can autonomously recharge themselves
when they are short of batteries.
Again, contingencies may happen.
The worst case is that even if the agent is planning to go to
one of the charging stations, she accidentally gets stuck and
runs out all the rest of power.



.. |img3| image:: ../../figs/warehouse_recharge.gif
	:width: 300px
	:align: middle

.. |img4| image:: ../../figs/warehouse_recharge_ctg.gif
	:width: 300px
	:align: middle

.. table::
   :align: center

   +---------+---------+
   |  |img3| |  |img4| |
   +---------+---------+


