import json
import time
import unittest
from itmobotics_gym.envs.single_robot_pybullet_env import SingleRobotPyBulletEnv

class testSingleRobotPyBulletEnv(unittest.TestCase):
    def setUp(self):
        with open('tests/test_single_robot_env.json') as json_file:
            self.__env_config = json.load(json_file)

        self.__env = SingleRobotPyBulletEnv(self.__env_config)
        self.__env.seed = int(time.time())
    
    def test_state(self):
        full_state = self.__env.get_observation_state_as_vec()
        time.sleep(5.0)
        self.assertEqual(len(full_state), 18)
        print(full_state)
                
def main():
    unittest.main(exit=False)

if __name__ == "__main__":
    main()