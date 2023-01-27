import gym
from gym.spaces import Discrete


import numpy as np
from .quadruped import DefaultValidatingDraft7Validator, Quadruped

from spatialmath import SE3, SO3
from spatialmath import base as sb

import json
import jsonschema

class A1GoForward(Quadruped):

    def __init__(self, config: dict):
        super().__init__(config)

        # with open('src/itmobotics_gym/envs/a1_go_forward_task_config_schema.json', encoding='UTF-8')\
        #     as json_file:
        #     task_schema = json.load(json_file)
            # DefaultValidatingDraft7Validator(task_schema).validate(self._env_config)

    def step(self, action: np.ndarray):

        done = False

        # Downgrade reward value depends on time
        reward = -0.001

        self._take_action_vector(action)
        self._sim.sim_step()

        obs = self.observation_state_as_vec()

        peg_in_hole_state = self._sim.link_state(
            model_name = '',
            link = 'world',
            reference_model_name = self._env_config['task']['robot']['model_name'],
            reference_link =  self._env_config['task']['robot']['target_link']
        )

        traveled_distance = peg_in_hole_state.tf.t[0]

        reward += 10*traveled_distance
        if self._sim.sim_time > self._env_config['task']['termination']['max_time']:
            done = True

        info = {'traveled_distance': traveled_distance, 'sim_time': self._sim.sim_time}
        return obs, reward, done, info

