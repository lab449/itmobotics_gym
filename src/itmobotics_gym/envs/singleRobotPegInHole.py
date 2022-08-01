import gym
import functools
from gym.spaces import Discrete
from pettingzoo import ParallelEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils import parallel_to_aec


import numpy as np
import itmobotics_sim as isim

import pybullet as p
from itmobotics_sim.utils.robot import EEState, JointState, Motion
from itmobotics_sim.pybullet_env.pybullet_world import PyBulletWorld, GUI_MODE
from itmobotics_sim.pybullet_env.pybullet_robot import PyBulletRobot
from itmobotics_sim.utils.controllers import CartVelocityToJointVelocityController


from spatialmath import SE3
from spatialmath import base as sb

import json
import jsonschema

class SinglePegInHole(gym.Env):
    metadata = {'render.modes': ['human']}  
    pegs = {'peg_round.urdf': ['hole_round.urdf', 'hole_sqr.urdf']}

    def __init__(self, config: dict):
        super().__init__()
        
        with open('single_peg_in_hole_env_config') as json_file:
            env_schema = json.load(json_file)
            jsonschema.validate(instance=self._connection_config, schema=env_schema)

        # There's a action vector space for vx, vy, vz, wx, wy, wz (linear and angular velocities around axis x,y,z)
        self.action_space = gym.spaces.box.Box(
            low=np.array(env_schema['']), dtype=np.float32),
            high=np.array([1.0,  1.0, 1.0, 2*np.pi, 2*np.pi, 2*np.pi], dtype=np.float32))

        # There's a observation vector space for px, py, pz, r, p, y, fx, fy, fz, tx, ty, tz 
        # (linear translation, Euler PRY angles of peg depens in hole frame, also forces and torques)
        self.observation_space = gym.spaces.box.Box(
            low=np.array([-1.5, -1.5, 0.0, -np.pi, -np.pi, -np.pi, -50, -50, -50, -20, -20, -20], dtype=np.float32),
            high=np.array([1.5,  1.5, 1.5, np.pi, np.pi, np.pi, 50, 50, 50, 20, 20, 20], dtype=np.float32))
        
        self.__hole_pose_range = (
            np.array([0.15, -0.4, 0.8, -np.pi/4, -np.pi/4, -np.pi/4]),
            np.array([0.6, 0.4, 1.5, np.pi/4, np.pi/4, np.pi/4])
        )

        self.__sim = PyBulletWorld('plane.urdf', gui_mode = GUI_MODE.SIMPLE_GUI, time_step = 0.01)
        self.__sim.add_object('table', 'urdf/table.urdf', fixed=True, save = True)
        self.__robot = PyBulletRobot('urdf/ur5e_pybullet.urdf', SE3(0,0.0,0.625))
        self.__sim.add_robot(self.__robot)
        
        self.__robot.apply_force_sensor('ee_tool')
        self.__controller_speed = CartVelocityToJointVelocityController(self.__robot)

    def reset(self):
        self.__sim.reset()
        self.__robot.reset()
        self.__robot.connect_tool('peg' ,'urdf/peg_round.urdf', root_link='ee_tool', tf=SE3(0.0, 0.0, 0.1), save=True)

        hole_random_pose = self.__init_hole_pose_range_min + np.random.uniform(0.2, 0.0, 0.625)*(self.__init_hole_pose_range_max - self.__init_hole_pose_range_min)

        self.__sim.add_object('hole_round', 'urdf/hole_round.urdf', base_transform = SE3(0.0, 1.0, 3.0), fixed = True, save = False)

        hole_state = SE3(*list(hole_random_pose)) @ SE3.Eul(hole_random_pose, order ='xyz')
        # self.__robot.reset_ee_state(robot_ee_state)

        hole_state_tf = self.__sim.link_tf(object_name='hole_round', link='hole_link')
        observation_state =  hole_state_tf.inv() @ robot_ee_state
        observation_as_vec = np.concatenate( (observation_state.t, observation_state.eul(), np.zeros(6)), axis=1)
        return observation_as_vec
    
    # def step
    