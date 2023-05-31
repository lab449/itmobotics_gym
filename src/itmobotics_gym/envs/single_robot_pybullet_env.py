from abc import abstractmethod
from json import tool
import os
import sys
import time
from unicodedata import name
import pkg_resources

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
from itmobotics_gym.data import get_data_path
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
        self._env_config = env_config
        with open(pkg_resources.resource_filename(__name__,'single_env_config_schema.json')) as json_file:
            env_schema = json.load(json_file)
            DefaultValidatingDraft7Validator(env_schema).validate(self._env_config)

        gui_mode = GUI_MODE.SIMPLE_GUI if env_config['simulation']['gui'] else GUI_MODE.DIRECT
        self.__render_config = None
        if 'render' in env_config['simulation']:
            self.__render_config = env_config['simulation']['render']

        self._sim = PyBulletWorld(
            gui_mode,
            time_step = self._env_config['simulation']['time_step'],
            time_scale = self._env_config['simulation']['sim_time_scale']
        )

        self._sim.add_additional_search_path(get_data_path())

        self._robot = self._sim.add_robot(self._env_config['robot']['urdf_filename'],
                                         vec2SE3(np.array(self._env_config['robot']['mount_tf'])),
                                         self._env_config['robot']['name'])

        tool_config = self._env_config['robot']['tool']
        if isinstance(tool_config, dict):
            self._robot.connect_tool(
                tool_config['name'],
                tool_config['urdf_filename'],
                tool_config['root_link'],
                tf=vec2SE3(tool_config['mount_tf']),
                save=True
            )

        if 'random_seed' in self._env_config['simulation']:
            self._np_random, seed = seeding.np_random(self._env_config['simulation']['random_seed'])
        else:
            self._np_random, seed = seeding.np_random(int(time.time()))

        # Definition of the action space vector
        self._controller_type = self._env_config['robot']['action_space']['type']
        assert self._controller_type in SingleRobotPyBulletEnv.action_controller_builder, "Unknows controller type {:s}".format(self._controller_type)
        self._action_robot_controller = SingleRobotPyBulletEnv.action_controller_builder[self._controller_type](self._robot)
        self.action_space = gym.spaces.box.Box(
            low=np.array(self._env_config['robot']['action_space']['range_min'], dtype=np.float32),
            high=np.array(self._env_config['robot']['action_space']['range_max'], dtype=np.float32)
        )
        if not 'target_link' in self._env_config['robot']['action_space']:
            self._env_config['robot']['action_space']['target_link'] = 'world'
        self._target_motion = Motion(ee_link = self._env_config['robot']['action_space']['target_link'], num_joints = self._robot.num_joints)
        
        self._state_references = {
            'joint_positions': self._robot.joint_limits.limit_positions,
            'joint_velocities': self._robot.joint_limits.limit_velocities,
            'joint_torques': self._robot.joint_limits.limit_torques,
            'cart_tf': (
                np.concatenate([-1e1*np.ones(3), -6.28*np.ones(3)]),
                np.concatenate([1e1*np.ones(3), 6.28*np.ones(3)])
            ),
            'cart_twist': (
                -5e2*np.ones(6),                           
                5e2*np.ones(6)
            ),
            'cart_force_torque': (
                np.concatenate([-1e6*np.ones(3), -1e4*np.ones(3)]),
                np.concatenate([1e6*np.ones(3), 1e4*np.ones(3)])
            )
        }

        # Definition of the observation space vector
        self.observation_space = []
        for state in self._env_config['robot']['observation_space']['type_list']:
            if 'camera' in state['type']:
                print("CONNECT CAMERA")
                self._robot.connect_camera(state['name'], resolution = state['resolution'], link=state['target_link'])
                self.observation_space.append(
                    gym.spaces.box.Box(
                        low=0, high=255,
                        shape=(state['resolution'][0], state['resolution'][1], 3), 
                        dtype=np.uint8
                    )
                )
            else:
                assert state['type'] in self._state_references, "Unknown type for observable state: {:s}, expecten one of this: {:s}".format(
                    state['type'], str(list(self._state_references.keys()))
                )
                self.observation_space.append(
                    gym.spaces.box.Box(
                        np.array(self._state_references[state['type']][0], dtype=np.float32).flatten(),
                        np.array(self._state_references[state['type']][1], dtype=np.float32).flatten()
                    )
                )
            

        self.observation_space = gym.spaces.Tuple( tuple(self.observation_space))
        self.reset()
    
    def observation_state_as_tuple(self) -> np.ndarray:
        full_state = []
        try:
            for state in self._env_config['robot']['observation_space']['type_list']:
                part_of_state = None
                if 'joint' in state['type']:
                    part_of_state = getattr(self._robot.joint_state, state['type'])
                    full_state.append(np.asarray(part_of_state, dtype=np.float32))
                elif 'cart' in state['type']:
                    link_state = self._sim.link_state(
                        state['target_model'],
                        state['target_link'],
                        state['reference_model'],
                        state['reference_link']
                    )
                    part_of_state = getattr(link_state, state['type'].replace('cart_', ''))
                    if 'tf' in state['type']:
                        part_of_state = SE32vec(part_of_state)
                    full_state.append(np.asarray(part_of_state, dtype=np.float32))
                elif 'camera' in state['type']:
                    part_of_state, _ = self._robot.get_image(state['name'])
                    full_state.append(np.asarray(part_of_state, dtype=np.uint8))
                else:
                    raise RuntimeError('Unknown observation state type with name')
        except AttributeError:
            raise AttributeError('Unknown observation state type with name: {:s}'.format(state['type']))
        assert self.observation_space.contains(tuple(full_state)), "Given observation state:\n {:s}\n is out of range of the limits:\n {:s}\n".format(str(full_state), str(self.observation_space))
        return tuple(full_state)

    def _sample_random_tf(self, init_tf: np.ndarray, random_variation: np.ndarray) -> SO3:
        pose_variation = (2*self._np_random.random(3) - 1.0)*random_variation[:3]
        random_pose = SE3(*( (init_tf[:3] + pose_variation).tolist()) )

        var_theta = random_variation[3]
        theta_orient_variation = (2*self._np_random.random() - 1.0) * var_theta
        vec_orient_variation = (2*self._np_random.random(3) - 1.0)

        orient_variation = SE3(SO3(sb.angvec2r(theta=theta_orient_variation, v=vec_orient_variation), check = False))
        only_rotation_init_tf = init_tf
        only_rotation_init_tf[:3] = 0.0
        random_orient = orient_variation @  Twist3(only_rotation_init_tf).SE3()        
        random_tf = random_pose @ random_orient
        return random_tf
    
    # @property
    # def seed(self):
    #     return self._seed

    # @seed.setter
    # def seed(self, seed: int):
    #     print("Set Seed = %d"%seed)
    #     self._np_random, self._seed = seeding.np_random(seed)

    def reset(self) -> np.ndarray:
        self._sim.reset()
        self._sample_random_objects()
        # self._sample_random_tool()
        self._sample_random_robot_state()
        obs = self.observation_state_as_tuple()
        # print(obs)
        return obs
    
        # pixelWidth = 640
        # pixelHeight = 640
        # color = np.zeros((pixelWidth,pixelHeight,3))
        # if not self.__render_config is None:
        #     pixelWidth = self.__render_config['resolution'][1]
        #     pixelHeight = self.__render_config['resolution'][0]
        #     nearPlane = 0.001
        #     farPlane = 5.0
        #     viewMatrix = vec2SE3(self.__render_config['view_cam_tf']).A
        #     print(viewMatrix)
        #     viewMatrix = viewMatrix.flatten()
        #     aspect = pixelHeight / pixelWidth
        #     projectionMatrix = p.computeProjectionMatrixFOV(self.__render_config['fov'], aspect, nearPlane, farPlane)
                
        #     color, _, _ = p.getCameraImage(self.__render_config['resolution'][1],self.__render_config['resolution'][0], viewMatrix, projectionMatrix)[2:5]
        #     color = np.reshape(color, (pixelHeight, pixelWidth, 4))[..., :3]

    @abstractmethod
    def render(self, mode: str = 'human', close: bool = False):
        pixelWidth = 640
        pixelHeight = 640
        color = np.zeros((pixelWidth,pixelHeight,3))
        if not self.__render_config is None:
            pixelWidth = self.__render_config['resolution'][1]
            pixelHeight = self.__render_config['resolution'][0]
            nearPlane = 0.001
            farPlane = 100
            camera_position = [1.0, 0.4, 1.2]
            up_vector = [0, 0, -1]
            target = [0, 0, 0]
            viewMatrix = p.computeViewMatrix(camera_position, target, up_vector)
            # print(viewMatrix)
            aspect = pixelHeight / pixelWidth
            projectionMatrix = p.computeProjectionMatrixFOV(self.__render_config['fov'], aspect, nearPlane, farPlane)
                
            color, _, _ = p.getCameraImage(
                pixelWidth,
                pixelHeight,
                viewMatrix,
                projectionMatrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL,
                flags=p.ER_NO_SEGMENTATION_MASK
            )[2:5]
            color = np.reshape(color, (pixelHeight, pixelWidth, 4))[..., :3]
        return color

    @abstractmethod
    def step(self, action: np.ndarray):
        pass
        # done = False
        # 
    
    def _take_action_vector(self, action: np.ndarray):
        action = np.asarray(action, np.float32)
        assert self.action_space.contains(action), "Given action state is out of range of the limits"
        if 'ee' in self._controller_type:
            if self._controller_type == "ee_tf":
                action = vec2SE3(action)
            setattr(self._target_motion.ee_state , self._controller_type.replace("ee_", ""), action)
        else:
            setattr(self._target_motion.joint_state , self._controller_type, action)
        
        self._action_robot_controller.send_control_to_robot(self._target_motion)

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
            assert len(new_init_state) == self._robot.num_joints, "Invalid size of robots init_state value. The robot has a different number of joints"
            partialy_fill_random_joint_state = np.zeros(self._robot.num_joints)
            config_random_variation = np.array(self._env_config['robot']['init_state']['random_variation'])
            partialy_fill_random_joint_state[:config_random_variation.shape[0]] = config_random_variation
            random_joint_state = (2*self._np_random.random(self._robot.num_joints) - 1.0)*partialy_fill_random_joint_state
            
            new_js = JointState.from_position(self._env_config['robot']['init_state']['value'] +  random_joint_state)
            self._robot.reset_joint_state(new_js)

        elif 'ee_tf' == self._env_config['robot']['init_state']['type']:
            new_tf = self._sample_random_tf(
                np.array(self._env_config['robot']['init_state']['value']),
                np.array(self._env_config['robot']['init_state']['random_variation'])
            )
            new_eestate = EEState.from_tf(new_tf, self._env_config['robot']['init_state']['target_link'])
            self._robot.reset_ee_state(new_eestate)
