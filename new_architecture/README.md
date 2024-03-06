# New Architecture
A gym package for tactical decision making of autonomous truck in highway environment using SUMO. The new architecture integrates RL with low level longitudinal and lateral controllers for the truck.

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
Action space is discrete.     

1. Set short time gap with target vehicle 
2. Set medium time gap with target vehicle
3. Set long time gap with target vehicle
4. Increase desired speed
5. Decrease desired speed
6. Change lane to left
7. Change lane to right
8. Maintain current desired speed and time gap
 
## Reward function
1. Positive reward propotional to speed (current speed/max speed)
2. Positive reward for completion of an episode (100/number of timesteps)
3. Negative reward (-10) for collision, near collision or driving outside road
4. Negative reward (-1) for lane change

## Register gym package
pip install -e sumo_gym_env
