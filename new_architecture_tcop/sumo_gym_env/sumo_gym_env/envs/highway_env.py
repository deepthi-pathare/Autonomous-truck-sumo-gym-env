import os
import sys
import numpy as np
import copy
import warnings
import gym
import math
from gym import spaces

#warnings.simplefilter('always', UserWarning)

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary
import traci

from sumo_gym_env.common.road import Road
from sumo_gym_env.common.longitudinal_controller import Longitudinal_Controller
from sumo_gym_env.common.lateral_controller import Lateral_Controller

# Action values
ACTION_COUNT = 8
SET_TARGET_VEH_SHORT_GAP = 0
SET_TARGET_VEH_MEDIUM_GAP = 1
SET_TARGET_VEH_LONG_GAP = 2
INCREASE_DESIRED_SPEED = 3
DECREASE_DESIRED_SPEED = 4
CHANGE_LANE_LEFT = 5
CHANGE_LANE_RIGHT = 6
MAINTAIN_SPEED_AND_GAP = 7

# Sumo subscription constants
POSITION = 66
LONG_SPEED = 64
LAT_SPEED = 50
LONG_ACC = 114
LANE_INDEX = 82
INDICATOR = 91 # Named as signal states in sumo
ROAD_ID = 80 # Edge name

class Highway(gym.Env):
    """
    This class creates a gym-like highway driving environment.

    The parameters of the environment are defined in parameters_simulation.py.
    The environment is built in a gym-like structure, with the methods 'reset' and 'step'

    Args:
        sim_params: Parameters that describe the simulation setup, the action space, and the reward function
        road_params: Parameters that describe the road geometry, rules, and properties of different vehicles
        use_gui (bool): Run simulation w/wo GUI
        start_time (str): Optional label
    """
    def __init__(self, sim_params, road_params, use_gui=True):
        super(Highway, self).__init__()
        self.step_ = 0
        self.sim_step_length = sim_params['sim_step_length']
        self.long_control_duration = sim_params['long_control_duration']
        self.max_steps = sim_params['max_steps']
        self.init_steps = sim_params['init_steps']
        self.nb_vehicles = sim_params['nb_vehicles']
        self.vehicles = None
        self.safety_check = sim_params['safety_check']

        self.road = Road(road_params)
        self.road.create_road()

        self.nb_lanes = self.road.road_params['nb_lanes']
        self.lane_width = self.road.road_params['lane_width']
        self.lane_change_duration = self.road.road_params['lane_change_duration']
        self.positions = np.zeros([self.nb_vehicles, 2])
        self.speeds = np.zeros([self.nb_vehicles, 2])
        self.accs = np.zeros([self.nb_vehicles, 1])
        self.lanes = np.zeros([self.nb_vehicles])
        self.indicators = np.zeros([self.nb_vehicles])
        self.edge_name = np.empty([self.nb_vehicles], dtype='object')
        self.init_ego_position = 0.
        self.ego_id = 'veh' + str(0).zfill(int(np.ceil(np.log10(self.nb_vehicles))))   # Add leading zeros to number

        self.max_speed = self.road.road_params['speed_range'][1]
        self.min_speed = self.road.road_params['speed_range'][0]
        self.max_allowed_ego_speed = min(self.road.road_params['vehicles'][0]['maxSpeed'], road_params['max_road_speed'])
        self.min_allowed_ego_speed = road_params['min_road_speed']
        self.sensor_range = sim_params['sensor_range']
        self.sensor_nb_vehicles = sim_params['sensor_nb_vehicles']
        self.target_veh_short_gap = sim_params['target_veh_short_gap']
        self.target_veh_medium_gap = sim_params['target_veh_medium_gap']
        self.target_veh_long_gap = sim_params['target_veh_long_gap']
        self.cruise_acceleration = sim_params['cruise_acceleration']
        self.cruise_deceleration = sim_params['cruise_deceleration']

        # Rewards/Penalties
        self.collision_penalty = sim_params['collision_penalty']
        self.near_collision_penalty = sim_params['near_collision_penalty']
        self.outside_road_penalty = sim_params['outside_road_penalty']
        self.lane_change_penalty = sim_params['lane_change_penalty']
        self.completion_reward = sim_params['completion_reward']

        self.nb_ego_states = 6
        self.nb_states_per_vehicle = 7

        # Actionspace and observation space for gym
        state_size = self.nb_ego_states + (self.sensor_nb_vehicles * self.nb_states_per_vehicle)
        l = np.full(shape=(state_size,), fill_value=(-math.inf))
        h = np.full(shape=(state_size,), fill_value=(math.inf))
        self.observation_space = spaces.Box(low = l, 
                                            high = h,
                                            dtype = np.float64)
    
        
        # Define an action space ranging from 0 to ACTION_COUNT-1
        self.action_space = spaces.Discrete(ACTION_COUNT)

        # Evaluation metrices
        self.nb_collisions = 0
        self.nb_near_collisions = 0
        self.nb_outside_road = 0
        self.nb_max_step = 0
        self.nb_max_distance = 0

        # Low level controllers
        self.long_controller = Longitudinal_Controller(self.ego_id, self.max_allowed_ego_speed, self.min_allowed_ego_speed,
                                                       self.target_veh_long_gap, self.sim_step_length, self.long_control_duration)
        self.lat_controller = Lateral_Controller(self.ego_id, self.nb_lanes, self.lane_width, self.sim_step_length)

        self.ego_speed = 0
        self.ego_energy_cost = 0
        self.ego_driver_cost = 0
        self.ego_tcop = 0

        # launch sumo
        self.use_gui = use_gui
        if self.use_gui:
            sumo_binary = checkBinary('sumo-gui')
        else:
            sumo_binary = checkBinary('sumo')
        
        # this is the normal way of using traci. sumo is started as a
        # subprocess and then the python script connects and runs
        if sim_params['remove_sumo_warnings']:
            traci.start([sumo_binary, "-c", self.road.road_path + self.road.name + ".sumocfg", "--start", "--step-length", str(self.sim_step_length), "--no-warnings"])
        else:
            traci.start([sumo_binary, "-c", self.road.road_path + self.road.name + ".sumocfg", "--start", "--step-length", str(self.sim_step_length)])

    def reset(self, sumo_ctrl=False):
        """
        Resets the highway driving environment to a new random initial state.

        The ego vehicle starts in a random lane. A number of surrounding vehicles are added to random positions.
        Vehicles in front of the ego vehicle are initalized with a lower speed than the ego vehicle, and vehicles behind
         the ego vehicle are initalized with a faster speed. If two vehicles vehicles are initalized too close
         to each other, one of them is moved.

        Args:
            sumo_ctrl (bool): For testing purposes, setting this True lets SUMO control the ego vehicle.

        Returns:
            observation (ndarray): The observation of the traffic situation, according to the sensor model.
        """
        # Remove all vehicles
        for veh in traci.vehicle.getIDList():
            traci.vehicle.unsubscribe(veh)
            traci.vehicle.remove(veh)
        traci.simulationStep()

        # Add vehicles
        for i in range(self.nb_vehicles):
            veh_id = 'veh' + str(i).zfill(int(np.ceil(np.log10(self.nb_vehicles))))   # Add leading zeros to number
            lane = i % self.nb_lanes
            traci.vehicle.add(veh_id, 'route0', typeID='truck' if i == 0 else 'car', depart=None, departLane=lane,
                              departPos='base', departSpeed=self.road.road_params['vehicles'][0]['maxSpeed'],
                              arrivalLane='current', arrivalPos='max', arrivalSpeed='current', fromTaz='', toTaz='',
                              line='', personCapacity=0, personNumber=0)
            if (i + 1) % self.nb_lanes == 0:  # When all lanes are filled
                traci.simulationStep()  # Deploy vehicles
                # Move all vehicles some meters forward so that next vehicles to be added do not get intersected with them
                # (Refer constraints for adding vehicles: https://sumo.dlr.de/docs/Simulation/VehicleInsertion.html )
                for veh in traci.vehicle.getIDList():
                    traci.vehicle.moveTo(veh, traci.vehicle.getLaneID(veh), traci.vehicle.getLanePosition(veh) + 50.)
        traci.simulationStep()
        assert (len(traci.vehicle.getIDList()) == self.nb_vehicles)
        self.vehicles = traci.vehicle.getIDList()

        # Randomly distribute vehicles on 'start' edge
        start_lane_name = self.road.road_params['edges'][1]
        start_lane_length = traci.lane.getLength(start_lane_name + '_0')
        x_pos = np.random.uniform(0., start_lane_length, self.nb_vehicles)
        x_pos[0] = start_lane_length/2  # Put ego vehicle on middle of the edge
        lane = np.random.randint(0, self.nb_lanes, self.nb_vehicles)  # Randomly disctribute vehicles on lanes

        # Set initial speed of vehicles (This also becomes their max speed, see line 177)
        init_speed = np.zeros(self.nb_vehicles)
        leaders = x_pos > x_pos[0]
        followers = x_pos < x_pos[0]
        init_speed[0] = self.max_allowed_ego_speed    # Start ego vehicle with maximum speed
        # Uniformly distribute speed between min and ego speed for leading vehicles
        init_speed[leaders] = np.random.uniform(self.road.road_params['speed_range'][0],   
                                                init_speed[0], np.sum(leaders))
        # Uniformly distribute speed between ego speed and max for following vehicles
        init_speed[followers] = np.random.uniform(init_speed[0], self.road.road_params['speed_range'][1],
                                                  np.sum(followers))

        # Position the vehicles and set their speeds as above using traci
        for i, veh in enumerate(self.vehicles):
            if i == 0 and not sumo_ctrl:
                traci.vehicle.moveTo(veh, start_lane_name + '_' + str(lane[i]), x_pos[i])
                traci.vehicle.setSpeed(veh, init_speed[i])   # Set current speed
                continue
            traci.vehicle.moveTo(veh, start_lane_name + '_' + str(lane[i]), x_pos[i])
            # SpeedMode 1 means that the speed will be reduced to a safe speed, but maximum acceleration is not
            # considered when using setSpeed
            traci.vehicle.setSpeedMode(veh, 1)
            traci.vehicle.setSpeed(veh, init_speed[i])   # Set current speed
            traci.vehicle.setMaxSpeed(veh, init_speed[i])   # Set speed of "cruise controller"

        # Turn off all internal lane changes and all safety checks for ego vehicle
        if not sumo_ctrl:
            if not self.safety_check:
                traci.vehicle.setSpeedMode(self.ego_id, 0)
                traci.vehicle.setLaneChangeMode(self.ego_id, 0)
        else:
            traci.vehicle.setSpeed(self.ego_id, -1)

        traci.vehicle.setAcceleration(self.ego_id, 0, self.init_steps * self.sim_step_length)

        # Variable subscriptions from sumo
        for veh in self.vehicles:
            traci.vehicle.subscribe(veh, [POSITION, LONG_SPEED, LAT_SPEED, LONG_ACC, LANE_INDEX, INDICATOR, ROAD_ID])

        # Initial simulation steps
        for i in range(self.init_steps):
            traci.simulationStep()

        # Return speed control to sumo, starting from initial random speed
        for i, veh in enumerate(self.vehicles[1:]):
            traci.vehicle.setSpeed(veh, -1)

        if self.use_gui:
            traci.gui.trackVehicle('View #0', self.ego_id)

        self.step_ = 0
        self.ego_speed = 0
        self.ego_energy_cost = 0
        self.ego_driver_cost = 0
        self.ego_tcop = 0

        self.init_ego_position = traci.vehicle.getPosition(self.ego_id)[0]

        # Get position and speed of vehicles after initial simulation steps
        for i, veh in enumerate(self.vehicles):
            out = traci.vehicle.getSubscriptionResults(veh)
            self.positions[i, :] = np.array(out[POSITION]) + np.array([0, self.lane_width * self.nb_lanes -
                                                                       self.lane_width/2])
            self.speeds[i, 0] = out[LONG_SPEED]
            self.speeds[i, 1] = out[LAT_SPEED]
            self.accs[i] = out[LONG_ACC]
            self.lanes[i] = out[LANE_INDEX]
            self.indicators[i] = out[INDICATOR]
            self.edge_name[i] = out[ROAD_ID]

            # Set color to vehicles based on speed
            if self.use_gui:
                if i == 0:
                    traci.vehicle.setColor(veh, (0, 200, 0))
                else:
                    speed_factor = (self.speeds[i, 0] - self.min_speed)/(self.max_speed - self.min_speed)
                    speed_factor = np.max([speed_factor, 0])
                    speed_factor = np.min([speed_factor, 1])
                    traci.vehicle.setColor(veh, (255, int(255*(1-speed_factor)), 0))

        # Create observation space
        state = [self.positions, self.speeds, self.lanes, self.indicators, self.edge_name, False]
        observation = self.sensor_model(state)
		
        self.ego_energy_cost += (self.long_controller.get_initial_kinetic_energy(self.speeds[0,0]) * 0.5)

        if self.use_gui:
            self.print_info_in_gui(info='Start')

        return observation

    def step(self, action, action_info=None, sumo_ctrl=False):
        """
        Transition the environment to the next state with the specified action.

        Args:
            action (int): Specified action, which is then translated to a longitudinal and lateral action.
            action_info (dict): Only used to display information in the GUI.
            sumo_ctrl (bool): For testing purposes, setting this True lets SUMO control the ego vehicle.

        Returns:
            tuple, containing:
                observation (ndarray): Observation of the environment, given by the sensor model.
                reward (float): Reward of the current time step.
                done (bool): True if terminal state is reached, otherwise False
                info (list): List of information on what caused the terminal condition.

        """
        self.step_ += 1
        outside_road = False
        done = False      
        info = []

        if action == SET_TARGET_VEH_SHORT_GAP:
                self.long_controller.change_desired_timegap(self.target_veh_short_gap)
        elif action == SET_TARGET_VEH_MEDIUM_GAP:
                self.long_controller.change_desired_timegap(self.target_veh_medium_gap)
        elif action == SET_TARGET_VEH_LONG_GAP:
                self.long_controller.change_desired_timegap(self.target_veh_long_gap)
        elif action == INCREASE_DESIRED_SPEED:
                self.long_controller.change_desired_speed(self.cruise_acceleration)
        elif action == DECREASE_DESIRED_SPEED:
                self.long_controller.change_desired_speed(self.cruise_deceleration)
        elif action == CHANGE_LANE_LEFT:
                if(self.lanes[0] == 0):
                    done = True
                    outside_road = True
                    info.append('Outside of road')
                else:
                    self.lat_controller.change_to_left_lane()
        elif action == CHANGE_LANE_RIGHT:
                if(self.lanes[0] == (self.nb_lanes - 1)):
                    done = True
                    outside_road = True
                    info.append('Outside of road')
                else:
                    self.lat_controller.change_to_right_lane()
        elif action == MAINTAIN_SPEED_AND_GAP:
                self.long_controller.maintain_speed_and_gap()                
        else:
                print('Undefined action, this should never happen')

        # Number of digits in vehicle name. Can't just enumerate index because vehicles can be removed in the event of
        # simultaneous change to center lane.
        nb_digits = int(np.floor(np.log10(self.nb_vehicles))) + 1
        for veh in self.vehicles:
            i = int(veh[-nb_digits:])   # See comment above
            out = traci.vehicle.getSubscriptionResults(veh)
            # Skip if the vehicle has left from simulation
            if out == {}:
                continue
            self.positions[i, :] = np.array(out[POSITION]) + \
                np.array([0, self.lane_width * self.nb_lanes - self.lane_width/2])
            self.speeds[i, 0] = out[LONG_SPEED]
            self.speeds[i, 1] = out[LAT_SPEED]
            self.accs[i] = out[LONG_ACC]
            self.lanes[i] = out[LANE_INDEX]
            self.indicators[i] = out[INDICATOR]
            self.edge_name[i] = out[ROAD_ID]

            # Set color to vehicles based on speed
            if self.use_gui and not i == 0:
                if i == 0:
                    traci.vehicle.setColor(veh, (0, 200, 0))
                else:
                    speed_factor = (self.speeds[i, 0] - self.min_speed)/(self.max_speed - self.min_speed)
                    speed_factor = np.max([speed_factor, 0])
                    speed_factor = np.min([speed_factor, 1])
                    traci.vehicle.setColor(veh, (255, int(255*(1-speed_factor)), 0))

        # Check for collision
        collision = traci.simulation.getCollidingVehiclesNumber() > 0
        ego_collision = False
        ego_near_collision = False
        if collision:
            colliding_ids = traci.simulation.getCollidingVehiclesIDList()
            colliding_positions = [traci.vehicle.getPosition(veh) for veh in colliding_ids]

            # If "collision" because violating minGap distance. Don't consider this as a collision,
            # but a near collision and give negative reward.
            if self.ego_id in colliding_ids:
                long_dist = colliding_positions[1][0] - colliding_positions[0][0]
                if colliding_ids[0] == 'veh00':
                    other_veh_idx = 1
                else:
                    other_veh_idx = 0
                if other_veh_idx == 0:
                    long_dist = -long_dist
                if long_dist > 0:   # Only consider collisions when the other vehicle is in front of the ego vehicle
                    front_veh_length = traci.vehicle.getLength(colliding_ids[other_veh_idx])
                    if long_dist - front_veh_length > 0:
                        collision = False
                        ego_near_collision = True
        # Second if statement because a situation that is considered a collision by SUMO can be reclassified to a
        # near collision
        if collision:
            info.append(colliding_ids)
            info.append(colliding_positions)
            colliding_speeds = [traci.vehicle.getSpeed(veh) for veh in colliding_ids] 
            info.append(colliding_speeds)
            if self.step_ == 0:
                warnings.warn('Collision during reset phase. This should not happen.')
                #print(info)
            else:
                if self.ego_id in colliding_ids:
                    ego_collision = True
                    done = True
                else:
                    warnings.warn('Collision not involving ego vehicle. This should normally not happen.')
                    #print(self.step_, info)
                # assert self.ego_id in info[0]   # If not, there has been a collision between two other vehicles

        
        if self.edge_name[0] == "exit":
            done = True
            info.append('Reached exit edge')
        elif self.step_ == self.max_steps:
            done = True
            info.append('Max steps')

        if outside_road:
            self.nb_outside_road += 1
        elif ego_collision:
            self.nb_collisions += 1
        elif ego_near_collision:
            self.nb_near_collisions += 1
        elif self.step_ == self.max_steps:
            self.nb_max_step += 1
        elif self.edge_name[0] == "exit":
            self.nb_max_distance += 1
        else:
            pass

        state = copy.deepcopy([self.positions, self.speeds, self.lanes, self.indicators, self.edge_name, done])
        observation = self.sensor_model(state)
        reward = self.reward_model(state, action, ego_collision, ego_near_collision, outside_road)

        if self.use_gui:
            self.print_info_in_gui(reward=reward, action=action, info=info, action_info=action_info)

        info_dict = {'info':info}
        return observation, reward, done, info_dict
    
    def reward_model(self, new_state, action, ego_collision=False, ego_near_collision=False, outside_road=False):
        """
        Reward model of the highway environment.

        Args:
            new_state (list) : New state of the vehicles
            action (int): Action by the agent
            ego_collision (bool): True if ego vehicle collides.
            ego_near_collision (bool): True if ego vehicle is close to a collision.
            outside_road (bool): True if ego vehicle drives off the road.

        Returns:
            reward (float): Reward for the current environment step.
        """        
        reward = 0

        if action == CHANGE_LANE_LEFT or action == CHANGE_LANE_RIGHT:
            reward -= self.lane_change_penalty # Lane change penalty
            electricity_consumed = self.lat_controller.energy_consumed # kwh
        else:
            electricity_consumed = self.long_controller.energy_consumed # kwh

        reward -= (electricity_consumed * 0.5) # electricity cost - 0.5 euro/kwh

        time_consumed = self.long_control_duration / 3600 # hours
        reward -= (time_consumed * 50) # driver cost - 50 euro / hr
        
        # Collision/Near collision/Outside road penalty
        if outside_road:
            reward -= self.outside_road_penalty
        elif ego_near_collision:
            reward -= self.near_collision_penalty
        elif ego_collision:
            reward -= self.collision_penalty 
        # Reward when episode is completed successfully (ego vehicle reached exit edge)
        elif new_state[4][0] == "exit":   
            reward += (self.completion_reward)

        # compute evaluation metrices
        self.ego_speed += new_state[1][0, 0] # This is to compute the average speed.
        self.ego_energy_cost += (electricity_consumed * 0.5)
        self.ego_driver_cost += (time_consumed * 50)
        self.ego_tcop += reward
        return reward

    def sensor_model(self, state):
        """
        Sensor model of the ego vehicle.

        Creates an observation vector from the current state of the environment. All observations are normalized.
        Only surrounding vehicles within the sensor range are included.

        Args:
            state (list): Current state of the environment.

        Returns:
            observation( (ndarray): Current observation of the highway environment.
        """
        vehicles_in_range = np.abs(state[0][1:, 0] - state[0][0, 0]) <= self.sensor_range
        if np.sum(vehicles_in_range) > self.sensor_nb_vehicles:
            warnings.warn('More vehicles within range than sensor can represent')

        # Create a vector with state values(positionx,positiony lat speed, long speed) of each vehicle
        observation = np.zeros(self.nb_ego_states + self.nb_states_per_vehicle * self.sensor_nb_vehicles)
        observation[0] = state[1][0, 0]   # Longitudinal speed
        observation[1] = np.sign(state[1][0, 1])   # Lane change state
        observation[2] = state[2][0]  # Lane number
        observation[3] = 1 if int(state[3][0]) & 2 else 0 # Left indicator 
        observation[4] = 1 if int(state[3][0]) & 1 else 0 # Right indicator 

        leader = traci.vehicle.getLeader(self.ego_id) # Returns leading vehicle id and distance
        if leader == None:
            leading_veh_dist = 1e6
        else:
            leading_veh_dist = leader[1]

        observation[5] = leading_veh_dist # Distance to leading vehicle

        s = self.nb_ego_states
        idx = 0
        for i, in_range in enumerate(vehicles_in_range):
            if not in_range:
                continue
            observation[s + idx * self.nb_states_per_vehicle] = (state[0][i + 1, 0] - state[0][0, 0]) # Longitudinal distance to ego vehicle
            observation[s + 1 + idx * self.nb_states_per_vehicle] = (state[0][i + 1, 1] - state[0][0, 1]) # Lateral distance to ego vehicle
            observation[s + 2  + idx * self.nb_states_per_vehicle] = (state[1][i + 1, 0] - state[1][0, 0]) # Relative longitudinal speed
            observation[s + 3  + idx * self.nb_states_per_vehicle] = np.sign(state[1][i + 1, 1]) # Lane change state
            observation[s + 4  + idx * self.nb_states_per_vehicle] = state[2][i + 1]  # Lane number
            observation[s + 5  + idx * self.nb_states_per_vehicle] = 1 if int(state[3][i + 1]) & 2 else 0 # Left indicator 
            observation[s + 6  + idx * self.nb_states_per_vehicle] = 1 if int(state[3][i + 1]) & 1 else 0 # Right indicator 

            idx += 1
            if idx >= self.sensor_nb_vehicles:
                break
        
        # If number of vehicles in range is less than sensor_nb_vehicles, fill with dummy values
        for i in range(idx, self.sensor_nb_vehicles):
            observation[s + idx * self.nb_states_per_vehicle] = 0
            observation[s + 1 + idx * self.nb_states_per_vehicle] = 0
            observation[s + 2 + idx * self.nb_states_per_vehicle] = 0
            observation[s + 3 + idx * self.nb_states_per_vehicle] = 0
            observation[s + 4 + idx * self.nb_states_per_vehicle] = 0
            observation[s + 5 + idx * self.nb_states_per_vehicle] = 0
            observation[s + 6 + idx * self.nb_states_per_vehicle] = 0
            idx += 1

        return observation

    def print_info_in_gui(self, reward=None, action=None, info=None, action_info=None):
        """
        Prints information in the GUI.
        """
        polygons = traci.polygon.getIDList()
        for polygon in polygons:
            traci.polygon.remove(polygon)
        dy = 10
        traci.polygon.add('Position: {0:.1f}, {1:.1f}'.format(self.positions[0, 0] - self.init_ego_position,
                                                              self.positions[0, 1]),
                          [self.positions[0] + self.road.road_params['info_pos'], self.positions[0] +
                           self.road.road_params['info_pos'] + [1, 0]], [0, 0, 0, 0])
        traci.polygon.add('Speed: {0:.1f}, {1:.1f}'.format(*self.speeds[0, :]),
                          [self.positions[0] + self.road.road_params['info_pos'], self.positions[0] +
                           self.road.road_params['info_pos'] + [1, -dy]], [0, 0, 0, 0])
        traci.polygon.add('Action previous step: ' + str(action),
                          [self.positions[0] + self.road.road_params['info_pos'], self.positions[0] +
                           self.road.road_params['info_pos'] + [1, -2*dy]], [0, 0, 0, 0])
        traci.polygon.add('Reward: ' + str(reward),
                          [self.positions[0] + self.road.road_params['info_pos'], self.positions[0] +
                           self.road.road_params['info_pos'] + [1, -3 * dy]], [0, 0, 0, 0])
        traci.polygon.add(str(info),
                          [self.positions[0] + self.road.road_params['info_pos'], self.positions[0] +
                           self.road.road_params['info_pos'] + [1, -4*dy]], [0, 0, 0, 0])
        traci.polygon.add('Step: ' + str(self.step_),
                          [self.positions[0] + self.road.road_params['info_pos'], self.positions[0] +
                           self.road.road_params['info_pos'] + [1, -5 * dy]], [0, 0, 0, 0])
        if action_info is not None:
            if 'q_values' in action_info:
                traci.polygon.add('  | '.join(['{:6.1f}'.format(element) for element in action_info['q_values']]),
                                  [self.positions[0] + self.road.road_params['action_info_pos'], self.positions[0] +
                                   self.road.road_params['action_info_pos'] + [1, 0]], [0, 0, 0, 0])
            if 'q_values_all_nets' in action_info:
                for i, row in enumerate(action_info['q_values_all_nets']):
                    traci.polygon.add('  | '.join(['{:6.1f}'.format(element) for element in row]),
                                      [self.positions[0] + self.road.road_params['action_info_pos'], self.positions[0] +
                                       self.road.road_params['action_info_pos'] + [1, -i*dy]], [0, 0, 0, 0])
            if 'mean' in action_info:
                traci.polygon.add('  | '.join(['{:6.1f}'.format(element) for element in action_info['mean']]),
                                  [self.positions[0] + self.road.road_params['action_info_pos'], self.positions[0] +
                                   self.road.road_params['action_info_pos'] + [1, -10.5*dy]], [0, 0, 0, 0])
            if 'coefficient_of_variation' in action_info:
                traci.polygon.add('  | '.join(['{:5.3f}'.format(element) for element in
                                               action_info['coefficient_of_variation']]),
                                  [self.positions[0] + self.road.road_params['action_info_pos'], self.positions[0] +
                                   self.road.road_params['action_info_pos'] + [1, -11.5*dy]], [0, 0, 0, 0])

    def close(self):        
        super(Highway, self).close()
        traci.close()
