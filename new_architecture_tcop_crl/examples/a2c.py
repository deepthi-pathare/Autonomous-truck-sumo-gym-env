import sys
import datetime 
import gym
import sumo_gym_env
import parameters as ps
from stable_baselines3 import A2C
from stable_baselines3.common.env_checker import check_env 
from stable_baselines3.common.logger import configure
from custom_callbacks import TensorboardCallback, EntropyDecayCallback
import pandas as pd

# Set log path and name
start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = "../logs/"+start_time+"/"
model_name = 'a2c'

validation_steps = 5
validation_episodes_per_step = 100

def validate(model, c):
    episode_count = validation_episodes_per_step

    env = gym.make('sumo_highway_env-v0', sim_params=ps.sim_params, road_params=ps.road_params, use_gui=False, curriculum = c)

    travelled_dist = []
    executed_steps = []
    tcop = []
    energy_cost = []
    driver_cost = []
    avg_speed = []

    for i in range(episode_count):
        done = False
        obs = env.reset()
        while not done:
            # Predict
            action, _states = model.predict(obs, deterministic=True)
            # Get reward
            obs, reward, done, info = env.step(action)
        travelled_dist.append(env.positions[0, 0] - env.init_ego_position)
        executed_steps.append(env.step_)
        tcop.append(env.ego_tcop)
        energy_cost.append(env.ego_energy_cost)
        driver_cost.append(env.ego_driver_cost)
        avg_speed.append(env.ego_speed/env.step_)

    failed_episodes = env.nb_outside_road + env.nb_collisions
    fail_rate = failed_episodes/episode_count
    succuess_rate = (env.nb_max_distance)/episode_count  
    max_step_rate = (env.nb_max_step)/episode_count  
    avg_dist = sum(travelled_dist) / len(travelled_dist)
    avg_steps = sum(executed_steps) / len(executed_steps)
    tcop = sum(tcop) / len(tcop)
    energy_cost = sum(energy_cost) / len(energy_cost)
    driver_cost = sum(driver_cost) / len(driver_cost)
    avg_speed = sum(avg_speed) / len(avg_speed)
    val = [succuess_rate, fail_rate, max_step_rate, avg_dist, avg_steps, tcop, energy_cost, driver_cost, avg_speed]
    env.close()

    return val

def train_and_eval():
    c = 0
    print("LEARNING CURRICULUM " + str(c))

    # Create the environment
    env = gym.make('sumo_highway_env-v0', sim_params=ps.sim_params, road_params=ps.road_params, use_gui=False, curriculum = c)
    obs = env.reset()

    # Create the model
    model = A2C('MlpPolicy', env, ent_coef=0.01,
                verbose=1)

    # Set up logger
    new_logger = configure(log_dir + str(c), ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    # Adaptive entropy
    ent_coef_initial = 0.01
    ent_coef_final = 0.0001  
    decay_steps = 700000
    entropy_decay_callback = EntropyDecayCallback(ent_coef_initial, ent_coef_final, decay_steps)

    # Train the model
    model.learn(total_timesteps=int(7e5), callback=[TensorboardCallback(), entropy_decay_callback])
    model.save(log_dir + model_name + '_curriculum_' + str(c))
    env.close()

    # Validate the model
    df = pd.DataFrame(columns=['success_rate', 'failure_rate', 'max_step_rate', 'avg_distance', 'avg_steps', 'tcop', 'energy_cost', 'driver_cost', 'avg_speed'])
    for i in range(validation_steps):
        val = validate(model, c)
        df.loc[i] = val
    df = df.append(df.mean(), ignore_index=True )
    df.to_csv(log_dir + model_name + '_curriculum_' + str(c) + '.csv')
    print(df)

    for c in range(1, 3):
        print("LEARNING CURRICULUM " + str(c))
                
        # Set new curriculum
        env = gym.make('sumo_highway_env-v0', sim_params=ps.sim_params, road_params=ps.road_params, use_gui=False, curriculum = c)
        env.reset()
        model.set_env(env)

        # Set up logger
        new_logger = configure(log_dir + str(c), ["stdout", "csv", "tensorboard"])
        model.set_logger(new_logger)

        # Adaptive entropy
        ent_coef_initial = 0.01
        ent_coef_final = 0.0001  
        decay_steps = 500000
        entropy_decay_callback = EntropyDecayCallback(ent_coef_initial, ent_coef_final, decay_steps)

        # Train the model
        model.learn(total_timesteps=int(5e5), callback=[TensorboardCallback(), entropy_decay_callback])
        model.save(log_dir + model_name + '_curriculum_' + str(c))

        env.close()

        # Validate the model
        df = pd.DataFrame(columns=['success_rate', 'failure_rate', 'max_step_rate', 'avg_distance', 'avg_steps', 'tcop', 'energy_cost', 'driver_cost', 'avg_speed'])
        for i in range(validation_steps):
            val = validate(model, c)
            df.loc[i] = val
        df = df.append(df.mean(), ignore_index=True )
        df.to_csv(log_dir + model_name + '_curriculum_' + str(c) + '.csv')
        print(df)

    del model

if __name__ == '__main__':
    print('Training started')
    train_and_eval()
    print('Training completed')