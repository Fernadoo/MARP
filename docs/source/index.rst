.. MARP documentation master file, created by
   sphinx-quickstart on Tue Apr 23 15:14:04 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to MARP's documentation!
================================

**MARP** (multi-agent route planning) is a Python testbed to facilitate research
in both multi-agent search and multi-agent learning.
More specifically,
we provide `PettingZoo-style <https://pettingzoo.farama.org/>`_ multi-agent environments
for multi-agent RL,
as well as model-based APIs for investigating efficient multi-agent search algorithms.



.. note::

   This project is still under active development.


Contents
--------

.. toctree::
   :maxdepth: 1
   :caption: User guide

   quick_start


.. toctree::
   :maxdepth: 1
   :caption: Algorithms
   
   search_algo
   rl_algo




API
---

.. autosummary::
   :toctree: generated
   :caption: API 

   marp.ma_env
   marp.marp_orth
   marp.marp_spin
   marp.search
   marp.utils