import json
import time
import unittest
from tqdm import tqdm

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
        random_action = np.random.uniform(-1, 1, size=6)*0.0
        for i in tqdm(range(1000)):
            full_state = self.__env.observation_state_as_vec()
            # print(full_state)
            # print(self.__env._robot.ee_state('peg_link'))
            self.assertEqual(len(full_state), 18)
            if i%10==0:
                random_action = np.random.uniform(-1, 1, size=6)*0.0
            self.__env._sim.sim_step()
            self.__env._take_action_vector(random_action)
            self.__env.step(random_action)

def main():
    unittest.main(exit=False)

if __name__ == '__main__':
    main()
