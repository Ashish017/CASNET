from gym.envs.registration import register

register(
    id='Hexapod-v32',
    entry_point='Hexapod_32.envs:Hexapod_v32',
)