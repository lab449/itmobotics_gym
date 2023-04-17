import gym
from gym.spaces import Discrete


import numpy as np
from .single_robot_pybullet_env import DefaultValidatingDraft7Validator, SingleRobotPyBulletEnv

from spatialmath import SE3, SO3
from spatialmath import base as sb

import json
import jsonschema
import pkg_resources

class SinglePegInHole(SingleRobotPyBulletEnv):

    def __init__(self, config: dict):
        super().__init__(config)

        with open(pkg_resources.resource_filename(__name__,'single_peginhole_task_config_schema.json')) as json_file:
            task_schema = json.load(json_file)
            DefaultValidatingDraft7Validator(task_schema).validate(self._env_config)

    
    def step(self, action: np.ndarray):

        done = False

        # Downgrade reward value depends on time
        reward = -0.001

        self._take_action_vector(action)
        self._sim.sim_step()

        obs = self.observation_state_as_dict()

        peg_in_hole_state = self._sim.link_state(
            self._env_config['task']['peg']['model_name'],
            self._env_config['task']['peg']['target_link'],
            self._env_config['task']['hole']['model_name'],
            self._env_config['task']['hole']['target_link']
        )
        peg_hole_distance = np.linalg.norm(peg_in_hole_state.tf.t)

        ft_config = self._env_config['task']['termination']['force_torque']
        current_force_torque = self._robot.ee_state(ft_config['target_link']).force_torque
        
        if np.any(np.abs(current_force_torque) > np.asarray(ft_config['limits'])):
            done = True
        if peg_hole_distance < self._env_config['task']['termination']['complete_pose_tolerance']:
            reward += 1000.0
            done = True
        if self._sim.sim_time > self._env_config['task']['termination']['max_time']:
            done = True

        reward -= 1000*np.linalg.norm(current_force_torque)/np.linalg.norm(self._state_references['cart_force_torque'][1][:3])
        reward -= 1000*peg_hole_distance/np.linalg.norm(self._state_references['cart_tf'][1][:3])            
        
        info = {'peg_hole_distance': peg_hole_distance, 'force_torque': current_force_torque, 'sim_time': self._sim.sim_time}
        return obs, reward, done, info



