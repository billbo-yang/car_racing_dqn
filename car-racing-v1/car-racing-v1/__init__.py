from gym.envs.registration import register

register(
    id='car-racing-v1',
    entry_point='car-racing-v1.envs:CarRacing',
)
register(
    id='car_dynamics-v1',
    entry_point='car-racing-v1.envs:CarDynamics',
)