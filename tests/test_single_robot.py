import json
from itmobotics_gym.envs.single_robot_pybullet_env import SingleRobotPyBulletEnv

with open('tests/test_single_robot_env.json') as json_file:
    env_config = json.load(json_file)

env = SingleRobotPyBulletEnv(env_config)