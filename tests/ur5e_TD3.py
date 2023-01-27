import json
import time
import unittest
from tqdm import tqdm

import numpy as np
from itmobotics_gym.envs.single_robot_peginhole_env import SinglePegInHole

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

with open("tests/test_single_robot_peginhole.json", encoding="UTF-8") as json_file:
    env_config = json.load(json_file)

    env = SinglePegInHole(env_config)
    time.sleep(5.0)
    check_env(env)

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = DDPG('MlpPolicy', env, verbose=1, action_noise=action_noise, tensorboard_log="./ddpg_peginhole_tensorboard/")
    number_of_epoch = 10
    for i in tqdm(range(number_of_epoch)):
        model.learn(total_timesteps=100000, tb_log_name="ddpg")
        model.save("ddpg_peginhole" + str(i))
    del model # remove to demonstrate saving and loading
    