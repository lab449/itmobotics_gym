import json
import time
import unittest

import numpy as np
from itmobotics_gym.envs.single_robot_peginhole_env import SinglePegInHole

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise


class testSingleRobotPegInHoleEnv(unittest.TestCase):
    def setUp(self):
        with open('tests/test_single_robot_peginhole.json') as json_file:
            self.__env_config = json.load(json_file)

        self.__env = SinglePegInHole(self.__env_config)
        time.sleep(5.0)
        check_env(self.__env)
    
    @unittest.skip
    def test_sim(self):
        number_of_epoch = 10
        for i in range(number_of_epoch):
            done = False
            while not done:
                action = self.__env.action_space.sample()*0.01
                obs, reward, done, info = self.__env.step(action)
            print(info)
            self.__env.reset()
    
    @unittest.skip
    def test_trained_model(self):
        model = DDPG.load('ddpg_peginhole1', env=self.__env)
        for i in range(1000):
            obs = self.__env.reset()
            done = False
            while not done:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, info = self.__env.step(action)
                print(reward)
                self.__env.render()
                time.sleep(0.1)
            
    def test_learn(self):
        n_actions = self.__env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

        model = DDPG('MlpPolicy', self.__env, verbose=1, action_noise=action_noise, tensorboard_log="./ddpg_peginhole_tensorboard/")
        number_of_epoch = 10
        for i in range(number_of_epoch):
            model.learn(total_timesteps=100000, tb_log_name="ddpg")
            model.save("ddpg_peginhole" + str(i))
        del model # remove to demonstrate saving and loading
        
            
                
def main():
    unittest.main(exit=False)

if __name__ == "__main__":
    main()