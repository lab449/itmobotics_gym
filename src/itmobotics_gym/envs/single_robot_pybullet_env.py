from abc import abstractmethod
import os
import sys
import time
from unicodedata import name

import gym
from gym.utils import seeding
from gym.spaces import Discrete

import numpy as np

import pybullet as p
import itmobotics_sim as isim
from itmobotics_sim.utils.robot import EEState, JointState, Motion
from itmobotics_sim.pybullet_env.pybullet_world import PyBulletWorld, GUI_MODE
from itmobotics_sim.pybullet_env.pybullet_robot import PyBulletRobot
from itmobotics_sim.utils.math import vec2SE3, SE32vec
import itmobotics_sim.utils.controllers as ctrl

from spatialmath import SE3, SO3, Twist3
from spatialmath import base as sb

import json
from jsonschema import Draft7Validator, validators

def extend_with_default(validator_class):
    validate_properties = validator_class.VALIDATORS["properties"]

    def set_defaults(validator, properties, instance, schema):
        for property, subschema in properties.items():
            if "default" in subschema:
                instance.setdefault(property, subschema["default"])

        for error in validate_properties(
            validator, properties, instance, schema,
        ):
            yield error

    return validators.extend(
        validator_class, {"properties" : set_defaults},
    )
DefaultValidatingDraft7Validator = extend_with_default(Draft7Validator)


class SingleRobotPyBulletEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    action_controller_builder = {
        'joint_positions': ctrl.JointPositionsController,
        'joint_velocities': ctrl.JointVelocitiesController,
        'joint_torques': ctrl.JointTorquesController,
        'ee_twist': ctrl.EEVelocityToJointVelocityController
    }


    def __init__(self, env_config: dict):
        super(SingleRobotPyBulletEnv, self).__init__()
        self._np_random, self._seed = seeding.np_random(int(time.time()))
        self._env_config = env_config
        with open('src/itmobotics_gym/envs/single_env_config_schema.json') as json_file:
            env_schema = json.load(json_file)
            DefaultValidatingDraft7Validator(env_schema).validate(self._env_config)

        self._sim = PyBulletWorld(
            self._env_config['world']['urdf_filename'], 
            gui_mode = GUI_MODE.SIMPLE_GUI, 
            time_step = self._env_config['simulation']['time_step']
        )
        self._robot = PyBulletRobot(
            self._env_config['robot']['urdf_filename'], 
            vec2SE3(np.array(self._env_config['robot']['init_state']))
        )
        self._sim.add_robot(self._robot, name='robot')

        self._action_robot_controller = SingleRobotPyBulletEnv.action_controller_builder[self._env_config['robot']['action_space']['type']](self._robot)
        self.action_space = gym.spaces.box.Box(
            low=np.array(self._env_config['robot']['action_space']['range_min'], dtype=np.float32),
            high=np.array(self._env_config['robot']['action_space']['range_max'], dtype=np.float32)
        )
        
        self._state_references = {
            'joint_positions': (
                -3.1457*np.ones(self._robot.num_joints),
                3.1457*np.ones(self._robot.num_joints)
            ),
            'joint_torques': (
                -1e3*np.ones(self._robot.num_joints),
                1e3*np.ones(self._robot.num_joints)
            ),
            'joint_velocities': (
                -1e1*np.ones(self._robot.num_joints),
                1e1*np.ones(self._robot.num_joints)
            ),
            'ee_tf': (
                np.concatenate([-1e1*np.ones(3), -6.28*np.ones(3)]),
                np.concatenate([-1e1*np.ones(3), -6.28*np.ones(3)])
            ),
            'ee_twist': (
                -5e1*np.ones(6),                           
                5e1*np.ones(6)
            ),
            'ee_force_torque': (
                np.concatenate([-1e2*np.ones(3), 1e-2*np.ones(3)]),
                np.concatenate([-1e2*np.ones(3), 1e-2*np.ones(3)])
            ),
            'cart_tf': (
                np.concatenate([-1e1*np.ones(3), -6.28*np.ones(3)]),
                np.concatenate([-1e1*np.ones(3), -6.28*np.ones(3)])
            ),
            'cart_twist': (
                -5e1*np.ones(6),                           
                5e1*np.ones(6)
            ),
            'cart_force_torque': (
                np.concatenate([-1e2*np.ones(3), 1e-2*np.ones(3)]),
                np.concatenate([-1e2*np.ones(3), 1e-2*np.ones(3)])
            )
        }
        observation_space_range_min = []
        observation_space_range_max = []
        for state in self._env_config['robot']['observation_space']['type_list']:
            assert state['type'] in self._state_references, "Unknown type for observable state: {:s}, expecten one of this: {:s}".format(
                state['type'], str(list(self._state_references.keys()))
            )
            observation_space_range_min.append(self._state_references[state['type']][0])
            observation_space_range_max.append(self._state_references[state['type']][1])

        self.observation_space = gym.spaces.box.Box(
            low=np.array(np.array(observation_space_range_min), dtype=np.float32),
            high=np.array(np.array(observation_space_range_max), dtype=np.float32))
        
        self.reset()
    
    def get_observation_state_as_vec(self) -> np.ndarray:
        full_state_vector = np.array([])
        try:
            for state_type in self._env_config['robot']['observation_space']['type_list']:
                part_of_state = None
                if 'joint' in state_type['type']:
                    part_of_state = getattr(self._robot.joint_state, state_type['type'])
                elif 'ee' in state_type['type']:
                    robot_ee_state = self._robot.ee_state(state_type['target_link'])
                    tf_in_to_reference = self._sim.link_state(
                        state_type['reference_model'], state_type['reference_link'], "", "world"
                    ).inv()
                    part_of_state = getattr(robot_ee_state, state_type['type'].replace('ee_', ''))
                    robot_ee_state.transform(tf_in_to_reference)
                    if 'tf' in state_type['type']:
                        part_of_state = SE32vec(part_of_state)
                elif 'cart' in state_type['type']:
                    tf_in_to_reference = self._sim.link_state(state_type['target_model'], state_type['target_link'], state_type['reference_model'], state_type['reference_link'])
                    
                full_state_vector = np.concatenate((full_state_vector, part_of_state))
        except AttributeError:
            raise AttributeError('Unknown observation state type with name: {:s}'.format(state_type['type']))
        return full_state_vector
    
    def _sample_random_tf(self, init_state: np.ndarray, random_variation: np.ndarray) -> SO3:
        pose_variation = (2*self._np_random.random(3) - 1.0)*random_variation[:3]
        random_pose = SE3(*( (init_state[:3] + pose_variation).tolist()) )

        var_theta = random_variation[3]
        theta_orient_variation = (2*self._np_random.random() - 1.0) * var_theta
        vec_orient_variation = (2*self._np_random.random(3) - 1.0)
        
        orient_variation = SE3(SO3(sb.angvec2r(theta=theta_orient_variation, v=vec_orient_variation), check = False))
        only_rotation_init_state = init_state
        only_rotation_init_state[:3] = 0.0
        random_orient = orient_variation @  Twist3(only_rotation_init_state).SE3()        
        random_tf = random_pose @ random_orient
        return random_tf
    
    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, seed: int):
        print("Set Seed = %d"%seed)
        self._np_random, self._seed = seeding.np_random(seed)

    def reset(self):
        self._sim.reset()
        self.__sample_random_objects()
        
    def render(self, mode: str = 'human', close: bool = False):
        pass

    def step(self, action: np.ndarray):
        pass
        # done = False
        # assert self.action_space.contains(action), "Invalid Action"

    def __sample_random_objects(self):
        for object_name in self._env_config['world']['world_objects']:
            object_config = self._env_config['world']['world_objects'][object_name]
            random_tf = self._sample_random_tf(
                np.array(object_config['init_state']),
                np.array(object_config['random_state_variation'])
            )
            self._sim.add_object(
                object_name,
                object_config['urdf_filename'],
                base_transform = random_tf,
                fixed = object_config['fixed'],
                save = object_config['save'],
                scale_size = object_config['scale_size']
            )