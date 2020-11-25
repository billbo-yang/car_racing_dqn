from gym.envs.registration import register

register(
    id='car_racing_v1-v0',
    entry_point='gym_car_racing_v1.envs:CarRacing',
)