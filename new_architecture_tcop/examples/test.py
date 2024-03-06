import gym
import parameters as ps
import sumo_gym_env

print('start')
env = gym.make('sumo_highway_env-v0', sim_params=ps.sim_params, road_params=ps.road_params, use_gui=False)
print('loaded env')
print('Number of actions {}'.format(env.action_space.n))
env.reset()
env.step(5)
env.close()
print('end')