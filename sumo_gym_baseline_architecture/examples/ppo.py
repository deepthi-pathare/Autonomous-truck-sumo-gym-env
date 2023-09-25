import sys
import datetime 
import gym
import sumo_gym_env
import parameters as ps
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env 

# Set log path and name
start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = "../logs/"+start_time+"/"
model_name = 'ppo'

timesteps = 1e6

def train():

    # Create the environment
    env = gym.make('sumo_highway_env-v0', sim_params=ps.sim_params, road_params=ps.road_params, use_gui=False)
    check_env(env)
    env.reset()

    # Save a checkpoint every 1000 steps
    checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path=log_dir,
    name_prefix="model",
    save_replay_buffer=False,
    save_vecnormalize=False,
    )

    # Create the model
    model = PPO('MlpPolicy', env,
                verbose=1,
                tensorboard_log=log_dir)
                

    # Train the model
    model.learn(total_timesteps=timesteps, callback = checkpoint_callback)
    model.save(log_dir + model_name)
    del model
    env.close()

if __name__ == '__main__':
    print('Training started')
    train()
    print('Training completed')