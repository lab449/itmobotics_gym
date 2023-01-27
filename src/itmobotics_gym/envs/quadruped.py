from abc import abstractmethod
from json import tool
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
    validate_properties = validator_class.VALIDATORS['properties']

    def set_defaults(validator, properties, instance, schema):
        for property, subschema in properties.items():
            if 'default' in subschema:
                instance.setdefault(property, subschema['default'])

        for error in validate_properties(
            validator, properties, instance, schema,
        ):
            yield error

    return validators.extend(
        validator_class, {'properties': set_defaults},
    )


DefaultValidatingDraft7Validator = extend_with_default(Draft7Validator)

class Quadruped(gym.Env):
    metadata = {'render.modes': ['human']}
    action_controller_builder = {
        'joint_positions': ctrl.JointPositionsController,
        'joint_velocities': ctrl.JointVelocitiesController,
        'joint_torques': ctrl.JointTorquesController,
    }

    def __init__(self, env_config: dict):
        super(Quadruped, self).__init__()
        self._env_config = env_config

        with open('src/itmobotics_gym/envs/quadruped.json',
                  encoding='UTF-8') as json_file:
            env_schema = json.load(json_file)
            DefaultValidatingDraft7Validator(
                env_schema).validate(self._env_config)

        gui_mode = GUI_MODE.SIMPLE_GUI if env_config['simulation']['gui'] else GUI_MODE.DIRECT

        self._sim = PyBulletWorld(
            gui_mode,
            time_step=self._env_config['simulation']['time_step'],
            time_scale=self._env_config['simulation']['sim_time_scale']
        )

        self._robot = self._sim.add_robot(
            self._env_config['robot']['urdf_filename'],
            vec2SE3(np.array(self._env_config['robot']['mount_tf'])),
            self._env_config['robot']['name']
        )

        controller_params = {'kp': 0.01*np.ones(12), 'kd': 1*np.array([1.0, 1.0, 1.0]*4)}
        self._robot.joint_controller_params = controller_params

        if 'tool' in self._env_config['robot']:
            tool_config = self._env_config['robot']['tool']
            if isinstance(tool_config, dict):
                self._robot.connect_tool(
                    tool_config['name'],
                    tool_config['urdf_filename'],
                    tool_config['root_link'],
                    tf=vec2SE3(tool_config['mount_tf']),
                    save=False
                )

        if 'random_seed' in self._env_config['simulation']:
            self._np_random, self._seed = seeding.np_random(
                self._env_config['simulation']['random_seed'])
        else:
            self._np_random, self._seed = seeding.np_random(int(time.time()))

        self._controller_type = self._env_config['robot']['action_space']['type']

        assert self._controller_type in Quadruped.action_controller_builder, \
            f'Unknows controller type {self._controller_type}'
        self._action_robot_controller = \
            Quadruped.action_controller_builder[self._controller_type](self._robot)
        self.action_space = gym.spaces.box.Box(
            low=np.array(self._env_config['robot']['action_space']['range_min'], dtype=np.float32),
            high=np.array(self._env_config['robot']['action_space']['range_max'], dtype=np.float32)
        )

        self._target_motion = Motion(
            ee_link=None,
            num_joints=self._robot.num_joints
        )

        self._state_references = {
            'joint_positions': self._robot.joint_limits.limit_positions,
            'joint_velocities': self._robot.joint_limits.limit_velocities,
            'joint_torques': self._robot.joint_limits.limit_torques
        }

        observation_space_range_min = []
        observation_space_range_max = []
        for state in self._env_config['robot']['observation_space']['type_list']:
            assert state['type'] in self._state_references, \
                f'Unknown type for observable state: {state["type"]}, '\
                f'expecten one of this: {list(self._state_references.keys())}'
            observation_space_range_min.append(self._state_references[state['type']][0])
            observation_space_range_max.append(self._state_references[state['type']][1])

        self.observation_space = gym.spaces.box.Box(
            low=np.array(observation_space_range_min, dtype=np.float32).flatten(),
            high=np.array(observation_space_range_max, dtype=np.float32).flatten()
        )
        self.reset()

    def _take_action_vector(self, action: np.ndarray):
        action = np.asarray(action, np.float32)
        assert self.action_space.contains(action), \
            'Given action state is out of range of the limits'
        # if 'ee' in self._controller_type:
        #     if self._controller_type == 'ee_tf':
        #         action = vec2SE3(action)
        #     setattr(self._target_motion.ee_state, self._controller_type.replace('ee_', ''),action)
        # else:
        setattr(self._target_motion.joint_state, self._controller_type, action)

        self._action_robot_controller.send_control_to_robot(self._target_motion)

    def _sample_random_tf(self, init_tf: np.ndarray, random_variation: np.ndarray) -> SO3:
        pose_variation = (2*self._np_random.random(3) - 1.0)*random_variation[:3]
        random_pose = SE3(*( (init_tf[:3] + pose_variation).tolist()) )

        var_theta = random_variation[3]
        theta_orient_variation = (2*self._np_random.random() - 1.0) * var_theta
        vec_orient_variation = (2*self._np_random.random(3) - 1.0)

        orient_variation = SE3(SO3(sb.angvec2r(theta=theta_orient_variation,\
            v=vec_orient_variation), check = False))
        only_rotation_init_tf = init_tf
        only_rotation_init_tf[:3] = 0.0
        random_orient = orient_variation @  Twist3(only_rotation_init_tf).SE3()        
        random_tf = random_pose @ random_orient
        return random_tf

    def _sample_random_objects(self):
        for object_name in self._env_config['world']['world_objects']:
            object_config = self._env_config['world']['world_objects'][object_name]
            random_tf = self._sample_random_tf(
                np.array(object_config['init_tf']),
                np.array(object_config['random_tf_variation'])
            )
            self._sim.add_object(
                object_name,
                object_config['urdf_filename'],
                base_transform = random_tf,
                fixed = object_config['fixed'],
                save = object_config['save'],
                scale_size = object_config['scale_size']
            )

    def _sample_random_tool(self):
        if 'tool' in self._env_config['robot']:
            tool_config = self._env_config['robot']['tool']
            if isinstance(tool_config, list):
                new_tool_config = self._np_random.choice(tool_config)
                self._robot.connect_tool(
                    new_tool_config['name'],
                    new_tool_config['urdf_filename'],
                    new_tool_config['root_link'],
                    tf=vec2SE3(new_tool_config['mount_tf'])
                )

    def _sample_random_robot_state(self):
        if 'joint_positions' == self._env_config['robot']['init_state']['type']:
            new_init_state = self._env_config['robot']['init_state']['value']
            assert len(new_init_state) == self._robot.num_joints, \
                'Invalid size of robots init_state value. The robot has a different number of joints'
            partialy_fill_random_joint_state = np.zeros(self._robot.num_joints)
            config_random_variation = np.array(self._env_config['robot']\
                ['init_state']['random_variation'])
            partialy_fill_random_joint_state[:config_random_variation.shape[0]] = config_random_variation
            random_joint_state = (2*self._np_random.random(self._robot.num_joints) - 1.0)\
                *partialy_fill_random_joint_state

            new_js = JointState.from_position(self._env_config['robot']['init_state']['value']\
                + random_joint_state)
            self._robot.reset_joint_state(new_js)

        elif 'ee_tf' == self._env_config['robot']['init_state']['type']:
            new_tf = self._sample_random_tf(
                np.array(self._env_config['robot']['init_state']['value']),
                np.array(self._env_config['robot']['init_state']['random_variation'])
            )
            new_eestate = EEState.from_tf(new_tf, \
                self._env_config['robot']['init_state']['target_link'])
            self._robot.reset_ee_state(new_eestate)

    def observation_state_as_vec(self) -> np.ndarray:
        full_state_vector = np.array([])
        try:
            for state_type in self._env_config['robot']['observation_space']['type_list']:
                part_of_state = None
                if 'joint' in state_type['type']:
                    part_of_state = getattr(self._robot.joint_state, state_type['type'])
                elif 'cart' in state_type['type']:
                    link_state = self._sim.link_state(
                        state_type['target_model'],
                        state_type['target_link'],
                        state_type['reference_model'],
                        state_type['reference_link']
                    )
                    part_of_state = getattr(link_state, state_type['type'].replace('cart_', ''))
                else:
                    raise RuntimeError('Unknown observation state type with name')
                if 'tf' in state_type['type']:
                    part_of_state = SE32vec(part_of_state)
                full_state_vector = np.concatenate([full_state_vector, part_of_state])
        except AttributeError:
            raise AttributeError(f'Unknown observation state type with name: {state_type["type"]}')

        full_state_vector = np.asarray(full_state_vector, dtype=np.float32)
        # assert self.observation_space.contains(full_state_vector), \
                # f'Given observation state:\n {full_state_vector}\n'\
                # f'is out of range of the limits:\n {self.observation_space}\n'
        return full_state_vector

    def reset(self) -> np.ndarray:
        self._sim.reset()
        self._sample_random_objects()
        self._sample_random_tool()
        self._sample_random_robot_state()
        obs = self.observation_state_as_vec()
        # print(obs)
        return obs

    @abstractmethod
    def render(self, mode: str = 'human', close: bool = False):
        pass

    @abstractmethod
    def step(self, action: np.ndarray):
        pass

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, seed: int):
        print(f'Set Seed = {seed}')
        self._np_random, self._seed = seeding.np_random(seed)
    