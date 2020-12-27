from gym.envs.registration import register

register(
    id='Hexapod-v16',
    entry_point='Hexapod_16.envs:Hexapod_v16',
)