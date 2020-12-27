from gym.envs.registration import register

register(
    id='Hexapod-v42',
    entry_point='Hexapod_42.envs:Hexapod_v42',
)