from gym.envs.registration import register

register(
    id='Hexapod-v12',
    entry_point='Hexapod_12.envs:Hexapod_v12',
)