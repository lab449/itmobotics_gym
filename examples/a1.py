import json
import time
import unittest
from tqdm import tqdm

import pybullet as p 
import numpy as np
from itmobotics_gym.envs.quadruped import Quadruped


with open('examples/a1_moment_env.json', encoding='UTF-8') as json_file:
    env_config = json.load(json_file)

env = Quadruped(env_config)
env.seed = int(time.time())
time.sleep(1.0)
env.reset()
time.sleep(1.0)

random_action = env.action_space.sample()*0.2
print(random_action)
# log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, 'assets/a1.mp4')
for i in tqdm(range(2000)):
    # full_state = env.observation_state_as_vec()
    if i%20==0:
        random_action = env.action_space.sample()*0.2
    env._sim.sim_step()
    env._take_action_vector(random_action)
    env.step(random_action)
# p.stopStateLogging(log_id)