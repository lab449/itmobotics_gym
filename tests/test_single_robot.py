import json
import time
import unittest

import numpy as np
from itmobotics_gym.envs.single_robot_pybullet_env import SingleRobotPyBulletEnv

class testSingleRobotPyBulletEnv(unittest.TestCase):
    def setUp(self):
        with open('tests/test_single_robot_env.json') as json_file:
            self.__env_config = json.load(json_file)

        self.__env = SingleRobotPyBulletEnv(self.__env_config)
        self.__env.seed = int(time.time())
    
    def test_state(self):
        for i in range(1000):
            full_state = self.__env.get_observation_state_as_vec()
            # print(full_state)
            self.assertEqual(len(full_state), 18)
            if i%10==0:
                self.__env._take_action_vector(np.random.uniform(-1, 1, size=6)*0.1)
            self.__env._sim.sim_step()
        
        
                
def main():
    unittest.main(exit=False)

if __name__ == "__main__":
    main()