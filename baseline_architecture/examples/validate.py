import sys
import gym
import sumo_gym_env
import parameters as ps
from stable_baselines3 import DQN, PPO, A2C
import pandas as pd

def validate(log_dir, model_file, model_name):

    episode_count = 100

    env = gym.make('sumo_highway_env-v0', sim_params=ps.sim_params, road_params=ps.road_params, use_gui=False)

    if model_name == 'dqn':
        model = DQN.load(log_dir + "/" + model_file, env=env)
    elif model_name == 'ppo':
        model = PPO.load(log_dir + "/" + model_file, env=env)
    elif model_name == 'a2c':
        model = A2C.load(log_dir + "/" + model_file, env=env)
    else:
        print('Unknown model')
        env.close()
        exit()

    travelled_dist = []
    executed_steps = []
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
        avg_speed.append(env.ego_speed/env.step_)

    failed_episodes = env.nb_outside_road + env.nb_collisions
    fail_rate = failed_episodes/episode_count
    succuess_rate = (env.nb_max_distance)/episode_count  
    max_step_rate = (env.nb_max_step)/episode_count  
    avg_dist = sum(travelled_dist) / len(travelled_dist)
    avg_steps = sum(executed_steps) / len(executed_steps)
    avg_speed = sum(avg_speed) / len(avg_speed)
    val = [succuess_rate, fail_rate, max_step_rate, avg_dist, avg_steps, avg_speed]
    env.close()

    return val


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Please specify log directory, model file and model name(dqn or ppo)")
        exit()

    log_dir = sys.argv[1]
    model_file = sys.argv[2]
    model_name = sys.argv[3]

    df = pd.DataFrame(columns=['success_rate', 'failure_rate', 'max_step_rate', 'avg_distance', 'avg_steps', 'avg_speed'])
    for i in range(5):
        val = validate(log_dir, model_file, model_name)
        df.loc[i] = val

    print('Validation completed')
    print(df)
    print('Avergae of 5 runs')
    print(df.mean(axis=0))