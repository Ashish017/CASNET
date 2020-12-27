from gym.envs.registration import register

register(
    id='Hexapod-v21',
    entry_point='Hexapod_21.envs:Hexapod_v21',
)