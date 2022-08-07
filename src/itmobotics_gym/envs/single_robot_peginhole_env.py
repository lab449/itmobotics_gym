import gym
from gym.spaces import Discrete


import numpy as np
from .single_robot_pybullet_env import DefaultValidatingDraft7Validator, SingleRobotPyBulletEnv

from spatialmath import SE3, SO3
from spatialmath import base as sb

import json
import jsonschema

class SinglePegInHole(SingleRobotPyBulletEnv):

    def __init__(self, config: dict):
        super().__init__(config)

        with open('src/itmobotics_gym/envs/single_peginhole_task_config_schema.json') as json_file:
            task_schema = json.load(json_file)
            DefaultValidatingDraft7Validator(task_schema).validate(self._env_config)

    

    
    # def step
    