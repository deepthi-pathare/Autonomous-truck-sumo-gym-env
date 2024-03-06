from gym.envs.registration import register

register(
    id='sumo_highway_env-v0',
    entry_point='sumo_gym_env.envs:Highway',
)