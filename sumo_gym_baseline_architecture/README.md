# Baseline Architecture
A gym package for tactical decision making of autonomous truck in highway environment using SUMO.
In the baseline architecture, the truck is controlled by a Reinforcement Learning (RL) agent.

## Observation Space
For ego vehicle
1. Longitudinal speed
2. Lane change state (1: changing to left, -1: changing to right, o: no lane change)
3. Lane number
4. State of left indicator
5. State of right indicator
6. Distance to leading vehicle

For each vehicle in the sensor range of ego vehicle:
1. Relative longitudinal distance from ego vehicle
2. Relative lateral distance from ego vehicle
3. Lane number
4. Relative longitudinal speed with ego vehicle
5. Lane change state (1: changing to left, -1: changing to right, o: no lane change)
6. State of left indicator
7. State of right indicator

## Action Space
Action space is discrete. Each action consists of [longitudinal action, lateral action]

Longitudinal actions are to set accelaration of 0, 1, -1 or -4 m/s2.
Lateral actions are to stay on line, change to left or change to right.
 
## Reward function
1. Positive reward propotional to speed (current speed/max speed)
2. Positive reward for completion of an episode (100/number of timesteps)
3. Negative reward (-10) for collision, near collision or driving outside road
4. Negative reward (-1) for lane change

## Register gym package
pip install -e sumo_gym_env
