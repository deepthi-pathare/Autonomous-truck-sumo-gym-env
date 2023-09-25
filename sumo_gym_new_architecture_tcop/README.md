# New Architecture with TCOP
A gym package for tactical decision making of autonomous truck in highway environment using SUMO. In this version of the new architecture, total cost of operation (TCOP) of the truck is incorporated into the reward function.

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
1. Negative reward for lane change (0.1)
2. Negative reward for energy cost (0.5 euro per kwh)
3. Negative reward for driver cost (50 euro per hour)
4. Negative reward for collision, near collision or driving outside road (1000)
5. When it reaches exit, positive reward for reacing exit (100)

## Register gym package
pip install -e sumo_gym_env
