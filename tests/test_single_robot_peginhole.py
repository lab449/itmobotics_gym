import json
import time
import unittest
from itmobotics_gym.envs.single_robot_peginhole_env import SinglePegInHole

class testSingleRobotPegInHoleEnv(unittest.TestCase):
    def setUp(self):
        with open('tests/test_single_robot_peginhole.json') as json_file:
            self.__env_config = json.load(json_file)

        self.__env = SinglePegInHole(self.__env_config)
        time.sleep(2.0)
    
    def test_env(self):
        self.__env.reset()
        self.__env._sim.sim_step()
        
        time.sleep(5.0)
        obs = self.__env.get_observation_state_as_vec()
        print(obs)
                
def main():
    unittest.main(exit=False)

if __name__ == "__main__":
    main()