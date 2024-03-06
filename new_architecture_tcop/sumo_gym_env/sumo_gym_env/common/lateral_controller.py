import traci
import math
from sumo_gym_env.common.energy_calc import calculate_energy_consumed

EGO_LATERAL_SPEED = 0.8 # m/s

class Lateral_Controller:
    def __init__(self, ego_id, nb_lanes, lane_width, sim_step_length):
        self.ego_id = ego_id
        self.nb_lanes = nb_lanes
        self.lane_width = lane_width
        self.sim_step_length = sim_step_length
        self.energy_consumed = 0

    def change_to_left_lane(self):
        """
            Change ego vehicle to left lane
        """             
        # 1e15 is the durarion: the lane will be chosen for the given amount of time (in s).
        # If vehicle is in leftmost lane, this method has no effect.
        traci.vehicle.changeLaneRelative(self.ego_id, -1, 1e15)
        self.update_simulation()

    def change_to_right_lane(self):
        """
            Change ego vehicle to right lane
        """             
        traci.vehicle.changeLaneRelative(self.ego_id, 1, 1e15)
        self.update_simulation()

    def update_simulation(self):
        """
            Update simulation
        """             
        lc_duration = self.lane_width/EGO_LATERAL_SPEED
        lc_steps = math.ceil(lc_duration/self.sim_step_length)
        for i in range(0, lc_steps):                    
            traci.simulationStep()
        
        self.energy_consumed = calculate_energy_consumed(0, EGO_LATERAL_SPEED, traci.vehicle.getSlope(self.ego_id), lc_duration)
