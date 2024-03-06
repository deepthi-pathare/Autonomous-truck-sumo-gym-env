import traci
import numpy as np
import math
from sumo_gym_env.common import energy_calc

IDM_EXP = 4

class Longitudinal_Controller:
    def __init__(self, ego_id, max_allowed_speed, min_allowed_speed,
                 desired_timegap, sim_step_length, long_control_duration):
        self.ego_id = ego_id
        self.max_allowed_speed = max_allowed_speed
        self.min_allowed_speed = min_allowed_speed
        self.sim_step_length = sim_step_length
        self.long_control_duration = long_control_duration
        self.long_control_step_count = math.ceil(long_control_duration/sim_step_length)
        self.desired_speed = max_allowed_speed
        self.desired_time_gap = desired_timegap
        self.energy_consumed = 0

    def change_desired_speed(self, change_value):
        """
            Change the desired speed for the ego vehicle
        Args:
            change_value (float): Speed change value in m/s
        """     
        self.desired_speed = self.desired_speed + change_value   
        if self.desired_speed > self.max_allowed_speed:
            self.desired_speed = self.max_allowed_speed
        elif self.desired_speed < self.min_allowed_speed:
            self.desired_speed = self.min_allowed_speed

        self.execute_longitudinal_control()

    def change_desired_timegap(self, time_gap):
        """
            Set a time gap between ego vehicle and target(leading) vehicle
        Args:
            time_gap (float): Desired time gap to be set between ego vehicle and target vehicle
        """
        self.desired_time_gap = time_gap
        self.execute_longitudinal_control()

    def execute_longitudinal_control(self):
        """
            Execute longitudinal control of ego vehicle based on the currently set desired speed and time gap
        """
        self.energy_consumed = 0
        for i in range(0, self.long_control_step_count):
            current_ego_speed = traci.vehicle.getSpeed(self.ego_id)
            new_ego_accel = self.get_follow_speed_by_idm()
            new_ego_speed = current_ego_speed + (new_ego_accel * self.sim_step_length)
            if new_ego_speed < 0:
                print("Speed < zero. current_sp={}, new_acc={}, new_sp={}".format(current_ego_speed, new_ego_accel, new_ego_speed))
                new_ego_speed = 0
            self.energy_consumed += energy_calc.calculate_energy_consumed(new_ego_accel, current_ego_speed,
                                                          traci.vehicle.getSlope(self.ego_id), self.sim_step_length)
            traci.vehicle.setSpeed(self.ego_id, new_ego_speed)
            traci.simulationStep()

    def get_follow_speed_by_idm(self): 
        """
            Get speed to set a safe distance between ego vehicle 
            and target(leading) vehicle based on Intelligent Driver Model(IDM)
        """     
        ego_speed = traci.vehicle.getSpeed(self.ego_id)
        ego_xpos = traci.vehicle.getPosition(self.ego_id)[0]
        ego_accel = traci.vehicle.getAccel(self.ego_id) # maximum acceleration
        ego_decel = traci.vehicle.getDecel(self.ego_id) # maximum deceleration

        leader = traci.vehicle.getLeader(self.ego_id) # Returns leading vehicle id and distance
        if leader != None:
            leader_id = leader[0]
            lead_xpos = traci.vehicle.getPosition(leader_id)[0]

        if (leader == None) or (lead_xpos < ego_xpos):
            new_ego_accel = ego_accel * (1 - ((ego_speed/self.desired_speed) ** IDM_EXP))
        else:
            lead_speed = traci.vehicle.getSpeed(leader_id)
            lead_len = traci.vehicle.getLength(leader_id)
            s = lead_xpos - ego_xpos - lead_len
            delta_v = ego_speed - lead_speed
            min_gap = traci.vehicle.getMinGap(self.ego_id)

            s_star = min_gap + (ego_speed * self.desired_time_gap) + ((ego_speed * delta_v)/(2*np.sqrt(ego_accel * ego_decel)))
            new_ego_accel = ego_accel * (1 - ((ego_speed/self.desired_speed) ** IDM_EXP) - ((s_star/s) ** 2))
        
        if(new_ego_accel < -1 * ego_decel):
            new_ego_accel = -1 * ego_decel
        elif(new_ego_accel > ego_accel):
            new_ego_accel = ego_accel
        return new_ego_accel
    
    def maintain_speed_and_gap(self):
        """
            Maintain speed and time gap for the ego vehicle
        """       
        # Do not change the desired speed or time gap, just simulate steps based on current values
        self.execute_longitudinal_control()

    def get_initial_kinetic_energy(self, init_speed):
        return energy_calc.calculate_init_kinetic_energy(init_speed)