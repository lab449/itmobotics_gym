import json
import time
import unittest
from tqdm import tqdm

import numpy as np
from itmobotics_gym.envs.a1_go_forward_env import A1GoForward

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

with open('examples/a1_go_forward_task_env.json', encoding='UTF-8') as json_file:
    env_config = json.load(json_file)

env = A1GoForward(env_config)
time.sleep(5.0)
check_env(env)

n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = DDPG('MlpPolicy',
             env,
             verbose=1,
             action_noise=action_noise,
             tensorboard_log="./ddpg_a1goforward_tensorboard/"
)
number_of_epoch = 2
for i in tqdm(range(number_of_epoch)):
    model.learn(total_timesteps=1000, tb_log_name='ddpg')
    model.save('ddpg_peginhole' + str(i))
    # del model # remove to demonstrate saving and loading

env_config['simulation']['gui'] = True
env = A1GoForward(env_config)
for i in range(100):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        print(reward)
        env.render()
        # time.sleep(0.1)
