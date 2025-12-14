# Problem Description
We consider a simplified routing environment with two nodes connected by two parallel links (paths). Each link has a fixed capacity (e.g., 5 units). A sequence of 100 connection requests arrives over time. For each incoming request, the DQN agent must decide which link to use for routing.

Each request occupies one unit of capacity on the chosen link for a specified holding time, after which it departs and releases the occupied capacity. If both links are full when a new request arrives, the request is blocked. The objective of the DQN agent is to learn a routing policy that minimizes blocking probability or maximizes link utilization over time.

# You will see...
- How to define this routing problem as a custom environment
- How to use a pre-coded DRL agent for the custom environment (training and testing)

# Files
- ```net_env.py```: A custom environment example for a simple routing problem
- ```myrun.py```: A main runner (verify the env, configure DQN, train DQN, and test DQN)
- ```requests.txt```: request holding time

# Dependencies
- Python 3.12
- See ```requirements.txt``` for packages

# Reference
- [Custom env](https://gymnasium.farama.org/introduction/create_custom_env/#)
- [SB3 DQN](https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html#)