import json
import time
import unittest
from tqdm import tqdm

import numpy as np
from itmobotics_gym.envs.single_robot_pybullet_env import SingleRobotPyBulletEnv


with open('tests/test_single_robot_env.json', encoding='UTF-8') as json_file:
    env_config = json.load(json_file)

env = SingleRobotPyBulletEnv(env_config)
env.seed = int(time.time())
time.sleep(1.0)
env.reset()
time.sleep(1.0)

random_action = np.random.uniform(-1, 1, size=6)*0.0
for i in tqdm(range(1000)):
    full_state = env.observation_state_as_vec()
    if i%10==0:
        random_action = np.random.uniform(-1, 1, size=6)*0.0
    env._sim.sim_step()
    env._take_action_vector(random_action)
    env.step(random_action)
