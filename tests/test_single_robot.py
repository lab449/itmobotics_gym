import json
import time
import unittest
from tqdm import tqdm
import cv2

import numpy as np
from itmobotics_gym.envs.single_robot_pybullet_env import SingleRobotPyBulletEnv

class testSingleRobotPyBulletEnv(unittest.TestCase):
    def setUp(self):
        with open('tests/test_single_robot_env.json') as json_file:
            self.__env_config = json.load(json_file)

        self.__env = SingleRobotPyBulletEnv(self.__env_config)
        self.__env.seed = int(time.time())
        time.sleep(1.0)
        self.__env.reset()
        time.sleep(1.0)

    def test_env(self):
        random_action = np.random.uniform(-1, 1, size=6)*0.1
        for i in tqdm(range(1000)):
            full_state = self.__env.observation_state_as_tuple()
            # print(self.__env.robot.ee_state('peg_link'))
            if i%10==0:
                random_action = np.random.uniform(-1, 1, size=6)*1.0
            self.__env._sim.sim_step()
            self.__env._take_action_vector(random_action)
            self.__env.step(random_action)
            img = full_state[0]
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imshow('out', img_rgb)
            cv2.waitKey(1)

def main():
    unittest.main(exit=False)

if __name__ == '__main__':
    main()
