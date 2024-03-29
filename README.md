# Tactical decision making for Autonomous Truck in SUMO using Reinforcement Learning
A gym package for tactical decision making of autonomous truck in highway environment with SUMO.

The repository includes four versions of the framework:
1. baseline_architecture: Baseline architecture where the truck is controlled by a Reinforcement Learning (RL) agent.
2. new_architecture: New architecture which integrates RL with low level longitudinal and lateral controllers.
3. new_architecture_tcop: New architecture by incorporating total cost of operation (TCOP) of the truck into the RL reward function. 
4. new_architecture_tcop_crl: New architecture integrated with Curriculum Learning to train the RL agent with TCOP based reward function.

For more details of the environment and the architectures, please refer the paper [Improved Tactical Decision Making and Control Architecture for
Autonomous Truck in SUMO Using Reinforcement Learning](https://ieeexplore.ieee.org/document/10386803)
