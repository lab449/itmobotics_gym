import json
import time
from tqdm import tqdm
import cv2

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

import numpy as np
from itmobotics_gym.envs.single_robot_peginhole_env import SinglePegInHole

with open('single_robot_peginhole_gui.json', encoding='UTF-8') as json_file:
    env_config = json.load(json_file)

env = DummyVecEnv([lambda: SinglePegInHole(env_config)])
env.seed = int(time.time())
model = PPO.load('model.zip', print_system_info=True)

time.sleep(1.0)
obs = env.reset()
time.sleep(1.0)

print(env.action_space)

for _ in tqdm(range(10)):
    while True:
        action = np.array(np.array(model.predict(obs)[0]))
        obs, reward, done, info = env.step(action)
        # cv2.imshow('out', env.render())
        # cv2.waitKey(1)
        if done:
            break

env.close()
