import json
import time
import unittest

import numpy as np
from itmobotics_gym.envs.single_robot_peginhole_env import SinglePegInHole

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import DDPG


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
    
    def test_learn(self):
        model = DDPG('MlpPolicy', self.__env, verbose=1, tensorboard_log="./ddpg_peginhole_tensorboard/")
        number_of_epoch = 10
        for i in range(number_of_epoch):
            model.learn(total_timesteps=1000000, tb_log_name="ddpg")
            model.save("ddpg_peginhole" + str(i))
        del model # remove to demonstrate saving and loading
        
            
                
def main():
    unittest.main(exit=False)

if __name__ == "__main__":
    main()