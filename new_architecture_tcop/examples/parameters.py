"""
Parameters of the highway driving environment.

The meaning of the different parameters are described below.
"""

import numpy as np

# Simulation parameters
sim_params = {}
sim_params['sim_step_length'] = 0.1  # Length of one SUMO simulation step (seconds)
sim_params['long_control_duration'] = 1  # Duration of low level longitudinal control (seconds)
sim_params['max_steps'] = 500   # Maximum number of RL steps per episode
sim_params['init_steps'] = int(4 * (sim_params['long_control_duration']/sim_params['sim_step_length'])) # Initial SUMO simulation steps before episode starts. At least one to get sumo subscriptions right.
sim_params['nb_vehicles'] = 15   # Number of inserted vehicles.
sim_params['remove_sumo_warnings'] = True
sim_params['safety_check'] = False   # Should be False. Otherwise, SUMO checks if agent's decisions are safe.
sim_params['sensor_range'] = 200.
sim_params['sensor_nb_vehicles'] = 15   # Maximum number of vehicles the sensor can represent.
sim_params['target_veh_short_gap'] = 1   # Tight distance with target(leading) vehicle for ACC
sim_params['target_veh_medium_gap'] = 2   # Normal distance with target(leading) vehicle for ACC
sim_params['target_veh_long_gap'] = 3   # Long distance with target(leading) vehicle for ACC
sim_params['cruise_acceleration'] = 1   # Acceleration value for cruise control
sim_params['cruise_deceleration'] = -1   # Deceleration value for cruise control

# Reward parameters
sim_params['collision_penalty'] = 1000
sim_params['near_collision_penalty'] = 1000
sim_params['outside_road_penalty'] = 1000
sim_params['lane_change_penalty'] = 0.1
sim_params['completion_reward'] = 2.78

# Vehicle type parameters
# Vehicle 0 is the ego vehicle
vehicles = []
vehicles.append({})
vehicles[0]['id'] = 'truck'
vehicles[0]['vClass'] = 'trailer'
vehicles[0]['length'] = 16.0   # default 16.5
vehicles[0]['width'] = 2.55   # default 2.55
vehicles[0]['maxSpeed'] = 25.0
vehicles[0]['speedFactor'] = 1.
vehicles[0]['speedDev'] = 0
vehicles[0]['carFollowModel'] = 'Krauss'
vehicles[0]['minGap'] = 2.5   # default 2.5. Minimum longitudinal gap. A closer distance will trigger a collision.
vehicles[0]['accel'] = 1.1   # default 1.1.
vehicles[0]['decel'] = 4.0   # default 4.0.
vehicles[0]['emergencyDecel'] = 9.0   # default 4.0
vehicles[0]['sigma'] = 0.0   # default 0.5. Driver imperfection (0 = perfect driver)
vehicles[0]['tau'] = 1.0   # default 1.0. Time headway to leading vehicle.
vehicles[0]['color'] = '1,0,0'
vehicles[0]['laneChangModel'] = 'LC2013'
vehicles[0]['lcStrategic'] = 0
vehicles[0]['lcCooperative'] = 0   # default 1.0. 0 - no cooperation
vehicles[0]['lcSpeedGain'] = 1.0   # default 1.0. Eagerness for tactical lane changes.
vehicles[0]['lcKeepRight'] = 0   # default 1.0. 0 - no incentive to move to the rightmost lane
vehicles[0]['lcOvertakeRight'] = 0   # default 0. Obsolete since overtaking on the right is allowed.
vehicles[0]['lcOpposite'] = 1.0   # default 1.0. Obsolete for freeway.
vehicles[0]['lcLookaheadLeft'] = 2.0   # default 2.0. Probably obsolete.
vehicles[0]['lcSpeedGainRight'] = 1.0   # default 0.1. 1.0 - symmetric desire to change left/right
vehicles[0]['lcAssertive'] = 1.0   # default 1.0. 1.0 - no effect
vehicles[0]['lcMaxSpeedLatFactor'] = 1.0   # default 1.0. Obsolete.
vehicles[0]['lcSigma'] = 0.0   # default 0.0. Lateral imperfection.

# Vehicle 1 is the type of the surrounding vehicles
vehicles.append({})
vehicles[1]['id'] = 'car'
vehicles[1]['vClass'] = 'passenger'
vehicles[1]['length'] = 4.8   # default 5.0. 4.8 used in previous paper.
vehicles[1]['width'] = 1.8   # default 1.8.
vehicles[1]['maxSpeed'] = 100.0   # Obsolete, since will be randomly set later
vehicles[1]['speedFactor'] = 1.   # Factor times the speed limit. Obsolete, since the speed is set.
vehicles[1]['speedDev'] = 0   # Randomness in speed factor. Obsolete, since speed is set.
vehicles[1]['carFollowModel'] = 'Krauss'
vehicles[1]['minGap'] = 2.5   # default 2.5. Minimum longitudinal gap.
vehicles[1]['accel'] = 2.6   # default 2.6
vehicles[1]['decel'] = 4.5   # default 4.6
vehicles[1]['emergencyDecel'] = 9.0   # default 9.0
vehicles[1]['sigma'] = 0.0   # default 0.5. Driver imperfection.
vehicles[1]['tau'] = 1.0   # default 1.0. Time headway to leading vehicle.
vehicles[1]['laneChangModel'] = 'LC2013'
vehicles[1]['lcStrategic'] = 0
vehicles[1]['lcCooperative'] = 0   # default 1.0. 0 - no cooperation
vehicles[1]['lcSpeedGain'] = 1.0   # default 1.0. Eagerness for tactical lane changes.
vehicles[1]['lcKeepRight'] = 0   # default 1.0. 0 - no incentive to move to the rightmost lane
vehicles[1]['lcOvertakeRight'] = 0   # default 0. Obsolete since overtaking on the right is allowed.
vehicles[1]['lcOpposite'] = 1.0   # default 1.0. Obsolete for freeway.
vehicles[1]['lcLookaheadLeft'] = 2.0   # default 2.0. Probably obsolete.
vehicles[1]['lcSpeedGainRight'] = 1.0   # default 0.1. 1.0 - symmetric desire to change left/right
vehicles[1]['lcAssertive'] = 1.0   # default 1.0. 1.0 - no effect
# vehicles[1]['lcMaxSpeedLatStanding']   # default maxSpeedLat
vehicles[1]['lcMaxSpeedLatFactor'] = 1.0   # default 1.0. Obsolete.
vehicles[1]['lcSigma'] = 0.0   # default 0.0. Lateral imperfection.

# Road parameters
road_params = {}
road_params['name'] = 'highway'
road_params['nb_lanes'] = 3
road_params['lane_width'] = 3.2   # default 3.2
road_params['max_road_speed'] = 100.   # Set very high, the actual max speed is set by the vehicle type parameters.
road_params['min_road_speed'] = 1.
road_params['lane_change_duration'] = 4   # Number of time steps for a lane change
road_params['speed_range'] = np.array([15, 35])   # Speed range of surrounging vehicles.
road_params['min_start_dist'] = 30   # Minimum vehicle separation when the surrounding vehicles are added.
road_params['overtake_right'] = 'true'   # Allow overtaking on the right side.
road_params['nodes'] = np.array([[0., 0.], [400., 0.], [1000., 0.], [3000., 0.], [5000., 0.]])    # Road nodes
road_params['edges'] = ['add', 'start', 'highway', 'exit']
road_params['vehicles'] = vehicles
road_params['collision_action'] = 'warn'   # 'none', 'warn' (if none, sumo totally ignores collisions)
road_params['oncoming_traffic'] = False   # Only used for allowing test case with oncoming traffic

# Terminal output
road_params['emergency_decel_warn_threshold'] = 10   # A high value disables the warnings
road_params['no_display_step'] = 'true'

# Gui settings
road_params['view_position'] = np.array([750, 0])
road_params['zoom'] = 3500
road_params['view_delay'] = 200
road_params['info_pos'] = [0, 35]
road_params['action_info_pos'] = [0, -30]
