import os
import sys

import gym
from gym.utils import seeding
from gym.spaces import Discrete


import numpy as np

import pybullet as p
import itmobotics_sim as isim
from itmobotics_sim.utils.robot import EEState, JointState, Motion
from itmobotics_sim.pybullet_env.pybullet_world import PyBulletWorld, GUI_MODE
from itmobotics_sim.pybullet_env.pybullet_robot import PyBulletRobot
import itmobotics_sim.utils.controllers as ctrl

from spatialmath import SE3, SO3
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
        'cart_twist': ctrl.CartVelocityToJointVelocityController
    }


    def __init__(self, env_config: dict):
        super(SingleRobotPyBulletEnv, self).__init__()
        self.__env_config = env_config
        with open('src/itmobotics_gym/envs/single_env_config_schema.json') as json_file:
            env_schema = json.load(json_file)
            DefaultValidatingDraft7Validator(env_schema).validate(self.__env_config)

        self.__sim = PyBulletWorld(
            self.__env_config['world']['urdf_filename'], 
            gui_mode = GUI_MODE.SIMPLE_GUI, 
            time_step = self.__env_config['simulation']['time_step']
        )
        for obj in self.__env_config['world']['world_objects']:
            self.__sim.add_object(obj['name'], obj['urdf_filename'], fixed=obj['fixed'], save = obj['save'])

        self.__robot = PyBulletRobot(
            self.__env_config['robot']['urdf_filename'],
            SE3(*self.__env_config['robot']['init_pose']) @ SE3(SO3(sb.q2r(self.__env_config['robot']['init_rotation']),
            check=False))
        )
        self.__sim.add_robot(self.__robot)

        self.__action_robot_controller = SingleRobotPyBulletEnv.action_controller_builder[self.__env_config['robot']['action_space']['type']](self.__robot)
        self.action_space = gym.spaces.box.Box(
            low=np.array(self.__env_config['robot']['action_space']['range_min'], dtype=np.float32),
            high=np.array(self.__env_config['robot']['action_space']['range_max'], dtype=np.float32)
        )
        
        self.__state_references = {
            'joint_positions': (
                -3.1457*np.ones(self.__robot.num_joints),
                3.1457*np.ones(self.__robot.num_joints)
            ),
            'joint_torques': (
                -1e3*np.ones(self.__robot.num_joints),
                1e3*np.ones(self.__robot.num_joints)
            ),
            'joint_velocities': (
                -1e1*np.ones(self.__robot.num_joints),
                1e1*np.ones(self.__robot.num_joints)
            ),
            'cart_tf': (
                np.concatenate([-1e1*np.ones(3), -6.28*np.ones(3)]),
                np.concatenate([-1e1*np.ones(3), -6.28*np.ones(3)])
            ),
            'cart_twist': (
                -1e2*np.ones(6),                           
                1e2*np.ones(6)
            ),
            'cart_force_torque': (
                np.concatenate([-1e2*np.ones(3), 1e-2*np.ones(3)]),
                np.concatenate([-1e2*np.ones(3), 1e-2*np.ones(3)])
            )
        }
        observation_space_range_min = []
        observation_space_range_max = []
        for state in self.__env_config['robot']['observation_space']['type_list']:
            assert state['type'] in self.__state_references, "Unknown type for observable state: {:s}, expecten one of this: {:s}".format(state['type'], list(self.__state_references.keys()))
            observation_space_range_min.append(self.__state_references[state['type'][0]])
            observation_space_range_max.append(self.__state_references[state['type'][1]])

        self.observation_space = gym.spaces.box.Box(
            low=np.array(self.__env_config['robot']['observation_space']['range_min'], dtype=np.float32),
            high=np.array(self.__env_config['robot']['observation_space']['range_min'], dtype=np.float32))

    def seed(self,seed=None):
        print("Seed = %d"%seed)
        self.np_random,seed=seeding.np_random(seed)

    def step(self, action):
        done = False
        assert self.action_space.contains(action), "Invalid Action"
    
    def reset(self):
        pass

    def render(self, mode='human', close=False):
        pass 
        

    # def reset(self):
    #     self.__sim.reset()
    #     self.__robot.reset()
    #     self.__robot.connect_tool('peg' ,'urdf/peg_round.urdf', root_link='ee_tool', tf=SE3(0.0, 0.0, 0.1), save=True)

    #     hole_random_pose = self.__init_hole_pose_range_min + np.random.uniform(0.2, 0.0, 0.625)*(self.__init_hole_pose_range_max - self.__init_hole_pose_range_min)

    #     self.__sim.add_object('hole_round', 'urdf/hole_round.urdf', base_transform = SE3(0.0, 1.0, 3.0), fixed = True, save = False)

    #     hole_state = SE3(*list(hole_random_pose)) @ SE3.Eul(hole_random_pose, order ='xyz')
    #     # self.__robot.reset_ee_state(robot_ee_state)

    #     hole_state_tf = self.__sim.link_tf(object_name='hole_round', link='hole_link')
    #     observation_state =  hole_state_tf.inv() @ robot_ee_state
    #     observation_as_vec = np.concatenate( (observation_state.t, observation_state.eul(), np.zeros(6)), axis=1)
    #     return observation_as_vec
    
    # def step
    