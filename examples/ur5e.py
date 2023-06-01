import json
import time
from tqdm import tqdm
import cv2


import numpy as np
from itmobotics_gym.envs.single_robot_pybullet_env import SingleRobotPyBulletEnv


with open('single_robot_env.json', encoding='UTF-8') as json_file:
    env_config = json.load(json_file)

env = SingleRobotPyBulletEnv(env_config)
env.seed = int(time.time())
time.sleep(1.0)
env.reset()
time.sleep(1.0)

random_action = np.random.uniform(-1, 1, size=6)
print(random_action.shape)
for i in tqdm(range(1000)):
    obs = env.observation_state_as_dict()
    obs_img = obs['camera']
    
    random_action = np.random.uniform(-1, 1, size=6)

    env._sim.sim_step()
    env._take_action_vector(random_action)
    cv2.imshow('out', obs_img)
    # cv2.imshow('out', env.render())
    cv2.waitKey(1)
    env.step(random_action)
