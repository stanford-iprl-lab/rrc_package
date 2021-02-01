"""Custom Gym environment for the Real Robot Challenge Phase 1 (Simulation)."""
import numpy as np
import gym
import pybullet
import os.path as osp
import logging
import pdb

import signal
from contextlib import contextmanager

from scipy.spatial.transform import Rotation
from collections import deque
from gym import wrappers
from gym import ObservationWrapper
from gym.spaces import Dict

try:
    import robot_interfaces
    import robot_fingers
    from robot_interfaces.py_trifinger_types import Action
except ImportError:
    robot_interfaces = robot_fingers = None
    from trifinger_simulation.action import Action

import trifinger_simulation
import trifinger_simulation.visual_objects
from trifinger_simulation import trifingerpro_limits
from trifinger_simulation.tasks import move_cube

import rrc_iprl_package.pybullet_utils as pbutils
from rrc_iprl_package.control.custom_pinocchio_utils import CustomPinocchioUtils
from rrc_iprl_package.control.control_policy import ImpedanceControllerPolicy, TrajMode
from rrc_iprl_package.envs import env_wrappers
from rrc_iprl_package.envs import cube_env
from rrc_iprl_package.envs.cube_env import ActionType
from rrc_iprl_package.envs.env_utils import configurable
from rrc_iprl_package.control.controller_utils import PolicyMode
from rrc_iprl_package.control.control_policy import HierarchicalControllerPolicy
from dm_control.utils import rewards as dmr


MAX_DIST = move_cube._max_cube_com_distance_to_center
DIST_THRESH = 0.05
_CUBOID_WIDTH = max(move_cube._CUBOID_SIZE)
_CUBOID_HEIGHT = min(move_cube._CUBOID_SIZE)

ORI_THRESH = np.pi / 8
REW_BONUS = 10
REW_PENALTY = -10
POS_SCALE = np.array([0.128, 0.134, 0.203, 0.128, 0.134, 0.203, 0.128, 0.134,
                      0.203])


class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


@configurable(pickleable=True)
class PushCubeEnv(gym.Env):
    observation_names = ["robot_position",
            "robot_velocity",
            "robot_tip_positions",
            "object_position",
            "object_orientation",
            "goal_object_position",
            "goal_object_orientation",
            "action"]

    def __init__(
        self,
        initializer=None,
        cube_goal_pose=None,
        action_type=ActionType.POSITION,
        frameskip=1,
        num_steps=None,
        visualization=False,
        alpha=0.01,
        save_npz=None,
        target_dist=0.156,
        pos_coef=0.5,
        ori_coef=0.5,
        fingertip_coef=0.,
        step_coef=0.,
        ac_norm_pen=0.1,
        rew_fn='sigmoid',
        min_ftip_height=0.01, 
        max_velocity=0.17
        ):
        """Initialize.

        Args:
            initializer: Initializer class for providing initial cube pose and
                goal pose. If no initializer is provided, we will initialize in a way
                which is be helpful for learning.
            action_type (ActionType): Specify which type of actions to use.
                See :class:`ActionType` for details.
            frameskip (int):  Number of actual control steps to be performed in
                one call of step().
            visualization (bool): If true, the pyBullet GUI is run for
                visualization.
        """
        # Basic initialization
        # ====================

        self.initializer = initializer
        if initializer:
            self.goal = initializer.get_goal()
        else:
            self.goal = move_cube.Pose.from_dict(cube_goal_pose)
        self.info = {'difficulty': initializer.difficulty}
        self.visualization = visualization

        if frameskip < 1:
            raise ValueError("frameskip cannot be less than 1.")
        self.frameskip = frameskip
        self.episode_length = num_steps * frameskip if num_steps else move_cube.episode_length

        # will be initialized in reset()
        self.platform = None
        self.save_npz = save_npz
        self.action_log = []
        self._prev_action = np.zeros(9)
        self.action_type = action_type

        # Create the action and observation spaces
        # ========================================

        robot_torque_space = gym.spaces.Box(
            low=trifingerpro_limits.robot_torque.low,
            high=trifingerpro_limits.robot_torque.high,
        )
        robot_position_space = gym.spaces.Box(
            low=trifingerpro_limits.robot_position.low,
            high=trifingerpro_limits.robot_position.high,
        )
        robot_velocity_space = gym.spaces.Box(
            low=trifingerpro_limits.robot_velocity.low,
            high=trifingerpro_limits.robot_velocity.high,
        )

        object_state_space = gym.spaces.Dict(
            {
                "position": gym.spaces.Box(
                    low=trifingerpro_limits.object_position.low,
                    high=trifingerpro_limits.object_position.high,
                ),
                "orientation": gym.spaces.Box(
                    low=trifingerpro_limits.object_orientation.low,
                    high=trifingerpro_limits.object_orientation.high,
                ),
            }
        )

        # verify that the given goal pose is contained in the cube state space
        goal_pose = self.goal.to_dict()
        if not object_state_space.contains(goal_pose):
            raise ValueError("Invalid goal pose.")

        if self.action_type == ActionType.TORQUE:
            self.action_space = robot_torque_space
            self._initial_action = trifingerpro_limits.robot_torque.default
        elif self.action_type == ActionType.POSITION:
            self.action_space = robot_position_space
            self._initial_action = trifingerpro_limits.robot_position.default
        elif self.action_type == ActionType.TORQUE_AND_POSITION:
            self.action_space = gym.spaces.Dict(
                {
                    "torque": robot_torque_space,
                    "position": robot_position_space,
                }
            )
            self._initial_action = {
                "torque": trifingerpro_limits.robot_torque.default,
                "position": trifingerpro_limits.robot_position.default,
            }
        else:
            raise ValueError("Invalid action_type")

        p_low = np.concatenate([object_state_space.spaces['position'].low for _ in range(3)])
        p_high = np.concatenate([object_state_space.spaces['position'].high for _ in range(3)])

        obs_spaces = {
                "robot_position": robot_position_space,
                "robot_velocity": robot_velocity_space,
                "robot_torque": robot_torque_space,
                "robot_tip_positions": gym.spaces.Box(low=p_low, high=p_high),
                "robot_tip_forces": gym.spaces.Box(low=np.zeros(3), high=np.ones(3)),
                "action": self.action_space,
                "goal_object_position": object_state_space.spaces['position'],
                "goal_object_orientation": object_state_space.spaces['orientation'],
                "object_position": object_state_space.spaces['position'],
                "object_orientation": object_state_space.spaces['orientation'],
            }

        self.observation_space = gym.spaces.Dict({k:obs_spaces[k]
                                                  for k in self.observation_names})
        self._min_ftip_height = min_ftip_height
        self._max_velocity = max_velocity
        self.rew_fn = rew_fn
        self.obj_pos = deque(maxlen=20)
        self.obj_ori = deque(maxlen=20)
        self.alpha = alpha
        self.filtered_position = self.filtered_orientation = None
        self._target_dist = target_dist
        self._pos_coef = pos_coef
        self._ori_coef = ori_coef
        self._fingertip_coef = fingertip_coef
        self._step_coef = step_coef
        self._ac_norm_pen = ac_norm_pen

    def write_action_log(self, observation, action, reward, **log_kwargs):
        log = dict(
            observation=observation, action=action, t=self.step_count,
            reward=reward)
        for k, v in log_kwargs.items():
            log[k] = v
        self.action_log.append(log)

    def save_action_log(self, save_npz=None):
        save_npz = save_npz or self.save_npz
        if save_npz and self.action_log:
            np.savez(save_npz, initial_pose=self.initial_pose.to_dict(),
                     goal_pose=self.goal.to_dict(), action_log=self.action_log)
            del self.action_log
        self.action_log = []

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        move_cube.random = self.np_random
        return [seed]

    def _gym_action_to_robot_action(self, gym_action):
        # construct robot action depending on action type
        if self.action_type == ActionType.TORQUE:
            robot_action = Action(torque=gym_action, position=np.repeat(np.nan, 9))
        elif self.action_type == ActionType.POSITION:
            robot_action = Action(position=gym_action, torque=np.zeros(9))
        elif self.action_type == ActionType.TORQUE_AND_POSITION:
            robot_action = Action(
                torque=gym_action["torque"], position=gym_action["position"]
            )
        else:
            raise ValueError("Invalid action_type")

        return robot_action

    def _reset_platform_frontend(self, **platform_kwargs):
        """Reset the platform frontend."""
        logging.debug("Resetting simulation with robot_fingers (frontend-only)")

        # full reset is not possible is platform instantiated
        if self.platform is not None:
            logging.debug("Virtually resetting after %d resets", self.num_resets+1)
            self.num_resets += 1
        else:
            self.platform = robot_fingers.TriFingerPlatformFrontend()
            platform = trifinger_simulation.TriFingerPlatform(
                visualization=False,
                initial_object_pose=move_cube.sample_goal(difficulty=-1)
            )
            self.kinematics = platform.simfinger.kinematics

    def _reset_direct_simulation(self, **platform_kwargs):
        """Reset direct simulation.

        With this the env can be used without backend.
        """
        logging.debug("Resetting simulation with trifinger_simulation (pybullet-sim backend)")

        # reset simulation
        del self.platform

        # initialize simulation
        initial_object_pose = move_cube.sample_goal(difficulty=-1)
        self.platform = trifinger_simulation.TriFingerPlatform(
            visualization=self.visualization,
            initial_object_pose=initial_object_pose,
            object_mass=platform_kwargs.get('object_mass'),
            joint_friction=platform_kwargs.get('joint_friction'),
        )
        self.kinematics = self.platform.simfinger.kinematics

        # visualize the goal
        if self.visualization:
            self.goal_marker = trifinger_simulation.visual_objects.CuboidMarker(
                size=move_cube._CUBOID_SIZE,
                position=self.goal.position,
                orientation=self.goal.orientation,
                pybullet_client_id=self.platform.simfinger._pybullet_client_id,
            )
            pbutils.reset_camera()

    def reset(self, **platform_kwargs):
        # reset simulation
        if robot_fingers:
            self._reset_platform_frontend(**platform_kwargs)
        else:
            self._reset_direct_simulation(**platform_kwargs)

        if self.initializer:
            self.goal = self.initializer.get_goal()

        self.info = {"difficulty": self.initializer.difficulty}
        self.step_count = 0
        observation, _, _, _ = self.step(self._initial_action)
        return observation

    def get_camera_pose(self, camera_observation):
        if osp.exists('/output'):
            cam_pose = camera_observation.filtered_object_pose
            if np.linalg.norm(cam_pose.orientation) == 0.:
                return camera_observation.object_pose
            else:
                return cam_pose
        else:
            return camera_observation.object_pose

    def _create_observation(self, t, action):
        robot_observation = self.platform.get_robot_observation(t)
        camera_observation = self.platform.get_camera_observation(t)
        cam_pose = self.get_camera_pose(camera_observation)

        # use exponential smoothing filter camera observation pose 
        # filter orientation
        if self.filtered_orientation is None:
            self.filtered_orientation = cam_pose.orientation
        self.filtered_orientation = ((1-self.alpha)*self.filtered_orientation + 
                          self.alpha*cam_pose.orientation)

        # filter position
        if self.filtered_position is None:
            self.filtered_position = cam_pose.position
        self.filtered_position = ((1-self.alpha)*self.filtered_position +
                          self.alpha*cam_pose.position)

        try:
            robot_tip_positions = self.kinematics.forward_kinematics(
                robot_observation.position
            )
            robot_tip_positions = np.array(robot_tip_positions)
        except:
            robot_tip_positions = np.zeros(9)

        # verify that the given goal pose is contained in the cube state space
        goal_pose = self.goal.to_dict()

        self._obs_dict = observation = {
            "robot_position": robot_observation.position,
            "robot_velocity": robot_observation.velocity,
            "robot_torque": robot_observation.torque,
            "robot_tip_positions": robot_tip_positions,
            "robot_tip_forces": robot_observation.tip_force,
            "object_position": self.filtered_position,
            "object_orientation": self.filtered_orientation,
            "goal_object_position": np.asarray(goal_pose["position"]),
            "goal_object_orientation": np.asarray(goal_pose["orientation"]),
            "action": action
        }
        if not osp.exists('/output'):
            self.filtered_position = self.filtered_orientation = None

        return {k: observation[k] for k in self.observation_names}

    def compute_position_error(self, goal_pose, object_pose):
        pos_error = np.linalg.norm(object_pose.position - goal_pose.position)
        return pos_error

    def compute_corner_error(self, goal_pose, actual_pose):
        # copy goal and actual poses to new Pose objects
        goal_pose = move_cube.Pose.from_dict(goal_pose.to_dict())
        actual_pose = move_cube.Pose.from_dict(actual_pose.to_dict())
        # reset goal and actual pose positions to center
        # goal_pose.position = np.array([0., 0., .01])
        # actual_pose.position = np.array([0., 0., .01])
        goal_corners = move_cube.get_cube_corner_positions(goal_pose)
        actual_corners = move_cube.get_cube_corner_positions(actual_pose)
        orientation_errors = np.linalg.norm(goal_corners - actual_corners, axis=1)
        return orientation_errors

    def compute_orientation_error(self, goal_pose, actual_pose):
        goal_rot = Rotation.from_quat(goal_pose.orientation)
        actual_rot = Rotation.from_quat(actual_pose.orientation)

        y_axis = [0, 1, 0]
        goal_direction_vector = goal_rot.apply(y_axis)
        actual_direction_vector = actual_rot.apply(y_axis)

        orientation_error = np.arccos(
            goal_direction_vector.dot(actual_direction_vector)
        )

        # scale both position and orientation error to be within [0, 1] for
        # their expected ranges
        error = orientation_error / np.pi
        return error

    def compute_fingertip_error(self, observation):
        if 'robot_tip_positions' not in observation:
            ftip_pos = observation['robot_position']
            ftip_pos = self.kinematics.forward_kinematics(
                    observation['robot_position']
                )
        else:
            ftip_pos = observation.get('robot_tip_positions')
        ftip_err = np.linalg.norm(
                ftip_pos.reshape((3,3))
                - observation.get('object_position'), axis=1)
        return ftip_err

    def _compute_reward_sigmoid(self, previous_observation, observation):
        goal_pose = self.goal
        object_pose = move_cube.Pose(position=observation['object_position'],
                                     orientation=observation['object_orientation'])
        position_error = self.compute_position_error(goal_pose, object_pose)
        orientation_error = self.compute_orientation_error(goal_pose, object_pose)
        corner_error = self.compute_corner_error(goal_pose, object_pose).sum()
        #ftip_error = self.compute_fingertip_error(observation).sum()
        reward = dmr.tolerance(position_error, (0., DIST_THRESH/2),
                               margin=DIST_THRESH/2, sigmoid='long_tail')
        if reward > .5:
            reward += dmr.tolerance(orientation_error, (0., ORI_THRESH/2),
                                    margin=ORI_THRESH/2, sigmoid='long_tail')
        else:
            reward += .5*dmr.tolerance(orientation_error, (0., ORI_THRESH/2),
                                    margin=ORI_THRESH/2, sigmoid='long_tail')

        reward += dmr.tolerance(corner_error, (0., DIST_THRESH*3),
                                margin=DIST_THRESH*3, sigmoid='long_tail')
        # reward += dmr.tolerance(ftip_error, (3*_CUBOID_HEIGHT/2, 2*_CUBOID_HEIGHT),
        #                         margin=_CUBOID_HEIGHT/2, sigmoid='long_tail')
        self.info['pos_error'] = position_error
        self.info['ori_error'] = orientation_error
        self.info['corner_error'] = corner_error
        # self.info['ftip_error'] = ftip_error
        return reward

    def _compute_reward(self, previous_observation, observation):
        if self.rew_fn == 'sigmoid':
            return self._compute_reward_sigmoid(previous_observation, observation)

        goal_pose = self.goal
        if previous_observation is None:
            prev_object_pose = None
        else:
            prev_object_pose = move_cube.Pose(position=previous_observation['object_position'],
                                           orientation=previous_observation['object_orientation'])
        object_pose = move_cube.Pose(position=observation['object_position'],
                                     orientation=observation['object_orientation'])

        info = self.info
        pos_error = self.compute_position_error(goal_pose, object_pose)
        ori_error = self.compute_orientation_error(goal_pose, object_pose)
        corner_error = self.compute_corner_error(goal_pose, object_pose).sum()

        # compute previous object pose error
        if previous_observation:
            prev_pos_error = self.compute_position_error(goal_pose, prev_object_pose)
            prev_ori_error = self.compute_orientation_error(goal_pose, prev_object_pose)
            prev_corner_error = self.compute_corner_error(goal_pose, prev_object_pose).sum()
            step_rew = (prev_pos_error - pos_error) + (prev_ori_error - ori_error)
        else:
            step_rew = 0

        # compute position and orientation joint reward based on chosen rew_fn 
        if self.rew_fn == 'lin':
            rew = self._pos_coef * (1 - pos_error/self._target_dist)
            if self._ori_coef:
                rew += self._ori_coef * (1 - ori_error)
        elif self.rew_fn == 'exp4':
            rew = self._pos_coef * (1 - (pos_error/self._target_dist)**0.4)
            if self._ori_coef:
                rew += self._ori_coef * (1 - ori_error**0.4)
        elif self.rew_fn == 'exp':
            # pos error penalty/reward
            if pos_error >= self._target_dist:
                rew = -1
            else:
                rew = self._pos_coef * np.exp(
                        -np.min([0, pos_error - DIST_THRESH]) / self._target_dist)
            # ori error reward
            if self._ori_coef:
                rew += self._ori_coef * np.exp(-np.min([0, ori_error - ORI_THRESH]))
            # fingertip error reward
            if self._fingertip_coef:
                ftip_error = self.compute_fingertip_error(observation)
                rew += (self._fingertip_coef *
                        np.exp(-(ftip_error - _CUBOID_HEIGHT) / _CUBOID_HEIGHT)).sum()
        elif self.rew_fn == 'cost':
            rew = self._pos_coef * (-pos_error / self._target_dist)
            if self._ori_coef:
                rew += self._ori_coef * (-ori_error)

        # Add to info dict
        # compute action penalty
        ac_penalty = -np.linalg.norm(self._prev_action) * self._ac_norm_pen
        info['ac_penalty'] = ac_penalty
        if step_rew:
            info['step_rew'] = step_rew
        info['rew'] = rew
        info['pos_error'] = pos_error
        info['ori_error'] = ori_error
        info['corner_error'] = corner_error

        total_rew = rew + ac_penalty + self._step_coef * step_rew
        return total_rew + ((pos_error < DIST_THRESH) + (ori_error < ORI_THRESH))

    def step(self, action):
        if self.platform is None:
            raise RuntimeError("Call `reset()` before starting to step.")

        if not self.action_space.contains(action):
            raise ValueError(
                "Given action is not contained in the action space."
            )

        num_steps = self.frameskip

        # ensure episode length is not exceeded due to frameskip
        step_count_after = self.step_count + num_steps
        if step_count_after > move_cube.episode_length:
            excess = step_count_after - move_cube.episode_length
            num_steps = max(1, num_steps - excess)

        reward = 0.0
        observation = previous_observation = None
        for _ in range(num_steps):
            self.step_count += 1
            if self.step_count > move_cube.episode_length:
                raise RuntimeError("Exceeded number of steps for one episode.")

            # send action to robot
            robot_action = self._gym_action_to_robot_action(action)
            t = self.platform.append_desired_action(robot_action)

            # Use observations of step t + 1 to follow what would be expected
            # in a typical gym environment.  Note that on the real robot, this
            # will not be possible
            if previous_observation is None and observation is not None:
                previous_observation = observation
            observation = self._create_observation(t, self._prev_action)

        reward += self._compute_reward(
            previous_observation=previous_observation,
            observation=observation,
        )

        is_done = self.step_count == move_cube.episode_length
        if self.rew_fn == 'cost':
            is_done = is_done or reward >= 2.5
        else:
            if np.linalg.norm(observation['object_position'][:2]) >= self._target_dist:
                is_done = True

        if is_done and isinstance(self.initializer, env_wrappers.CurriculumInitializer):
            goal_pose = self.goal
            object_pose = move_cube.Pose.from_dict(dict(
                position=observation['object_position'].flatten(),
                orientation=observation['object_orientation'].flatten()))
            self.initializer.update_initializer(object_pose, goal_pose)

        self._prev_action = action
        return observation, reward, is_done, self.info


@configurable(pickleable=True)
class HierarchicalPolicyWrapper(ObservationWrapper):
    def __init__(self, env, policy):
        assert isinstance(env.unwrapped, cube_env.RealRobotCubeEnv), \
                'env expects type CubeEnv or RealRobotCubeEnv'
        self.env = env
        self.reward_range = self.env.reward_range
        # set observation_space and action_space below
        spaces = trifinger_simulation.TriFingerPlatform.spaces
        self._action_space = gym.spaces.Dict({
            'torque': spaces.robot_torque.gym, 'position': spaces.robot_position.gym}) 
        self._last_action = np.zeros(9)
        self.set_policy(policy)
        self._platform = None

    @property
    def impedance_control_mode(self):
        return (self.mode == PolicyMode.IMPEDANCE or
                (self.mode == PolicyMode.RL_PUSH and
                 self.rl_observation_space is None))

    @property
    def action_space(self):
        if self.impedance_control_mode:
            return self._action_space['torque']
        else:
            return self.wrapped_env.action_space

    @property
    def action_type(self):
        if self.impedance_control_mode:
            return ActionType.TORQUE
        else:
            return ActionType.POSITION

    @property
    def mode(self):
        assert self.policy, 'Need to first call self.set_policy() to access mode'
        return self.policy.mode

    @property
    def frameskip(self):
        if self.mode == PolicyMode.RL_PUSH:
            return self.policy.rl_frameskip
        return 4

    @property
    def step_count(self):
        return self.env.step_count

    @step_count.setter
    def step_count(self, v):
        self.env.step_count = v

    def set_policy(self, policy):
        self.policy = policy
        if policy:
            self.rl_observation_names = policy.observation_names
            self.rl_observation_space = policy.rl_observation_space
            obs_dict = {'impedance': self.env.observation_space}
            if self.rl_observation_space is not None:
                obs_dict['rl'] = self.rl_observation_space
            self.observation_space = gym.spaces.Dict(obs_dict)
        # if env is wrapped, unwrap and store wrapped_env separately
        if isinstance(policy.rl_env, gym.Wrapper):
            self.is_wrapped = True
            self.wrapped_env = self.policy.rl_env
        else:
            self.is_wrapped = False

    def observation(self, observation):
        obs_dict = {'impedance': observation}
        if 'rl' in self.observation_space.spaces:
            observation_rl = self.process_observation_rl(observation)
            obs_dict['rl'] = observation_rl
        return obs_dict

    def get_goal_object_ori(self, obs):
        val = obs['desired_goal']['orientation']
        goal_rot = Rotation.from_quat(val)
        actual_rot = Rotation.from_quat(np.array([0,0,0,1]))
        y_axis = [0, 1, 0]
        actual_vector = actual_rot.apply(y_axis)
        goal_vector = goal_rot.apply(y_axis)
        N = np.array([0,0,1])
        proj = goal_vector - goal_vector.dot(N) * N
        proj = proj / np.linalg.norm(proj)
        ori_error = np.arccos(proj.dot(actual_vector))
        xyz = np.zeros(3)
        xyz[2] = ori_error
        val = Rotation.from_euler('xyz', xyz).as_quat()
        return val

    def process_observation_rl(self, obs, return_dict=False):
        t = self.step_count
        obs_dict = {}
        cpu = self.policy.impedance_controller.custom_pinocchio_utils
        for on in self.rl_observation_names:
            if on == 'robot_position':
                val = obs['observation']['position']
            elif on == 'robot_velocity':
                val = obs['observation']['velocity']
            elif on == 'robot_tip_positions':
                val = cpu.forward_kinematics(obs['observation']['position'])
            elif on == 'object_position':
                val = obs['achieved_goal']['position']
            elif on == 'object_orientation':
                actual_rot = Rotation.from_quat(obs['achieved_goal']['orientation'])
                xyz = actual_rot.as_euler('xyz')
                xyz[:2] = 0.
                val = Rotation.from_euler('xyz', xyz).as_quat()
                val = obs['achieved_goal']['orientation']
            elif on == 'goal_object_position':
                val = 0 * np.asarray(obs['desired_goal']['position'])
            elif on == 'goal_object_orientation':
                # disregard x and y axis rotation for goal_orientation
                val = self.get_goal_object_ori(obs)
            elif on == 'relative_goal_object_position':
                val = 0. * np.asarray(obs['desired_goal']['position']) - np.asarray(obs['achieved_goal']['position'])
            elif on == 'relative_goal_object_orientation':
                goal_rot = Rotation.from_quat(self.get_goal_object_ori(obs))
                actual_rot = Rotation.from_quat(obs_dict['object_orientation'])
                if self.policy.rl_env.use_quat:
                    val = (goal_rot*actual_rot.inv()).as_quat()
                else:
                    val = get_theta_z_wf(goal_rot, actual_rot)
            elif on == 'action':
                val = self._last_action
                if isinstance(val, dict):
                    val = val['torque']
            obs_dict[on] = np.asarray(val, dtype='float64').flatten()
        if return_dict:
            return obs_dict

        self._prev_obs = obs_dict
        obs = np.concatenate([obs_dict[k] for k in self.rl_observation_names])
        return obs

    def reset(self, **platform_kwargs):
        self.resetting = True
        if self._platform is None:
            initial_object_pose = move_cube.sample_goal(-1)
            self._platform = trifinger_simulation.TriFingerPlatform(
                visualization=False,
                initial_object_pose=initial_object_pose,
            )
        obs = super(HierarchicalPolicyWrapper, self).reset(**platform_kwargs)
        initial_object_pose = move_cube.Pose.from_dict(obs['impedance']['achieved_goal'])
        # initial_object_pose = move_cube.sample_goal(difficulty=-1) 
        self.policy.reset_policy(obs['impedance'], self._platform)
        self._prev_action = np.zeros(9)
        self.resetting = False
        return obs

    def _step(self, action):
        if self.unwrapped.platform is None:
            raise RuntimeError("Call `reset()` before starting to step.")

        if not self.action_space.contains(action):
            raise ValueError(
                "Given action is not contained in the action space."
            )

        num_steps = self.frameskip

        # ensure episode length is not exceeded due to frameskip
        step_count_after = self.step_count + num_steps
        if step_count_after > self.episode_length:
            excess = step_count_after - self.episode_length
            num_steps = max(1, num_steps - excess)

        reward = 0.0
        for _ in range(num_steps):
            # send action to robot
            robot_action = self._gym_action_to_robot_action(action)
            self.step_count = t = self.unwrapped.platform.append_desired_action(robot_action)

            # Use observations of step t + 1 to follow what would be expected
            # in a typical gym environment.  Note that on the real robot, this
            # will not be possible
            if osp.exists('/output'):
                observation = self.unwrapped._create_observation(t, action)
            else:
                observation = self.unwrapped._create_observation(t+1, action)

            reward += self.unwrapped.compute_reward(
                observation["achieved_goal"],
                observation["desired_goal"],
                self.unwrapped.info,
            )

            if self.step_count >= self.episode_length:
                break

        is_done = self.step_count == self.episode_length
        info = self.env.info
        info['num_steps'] = self.step_count
        return observation, reward, is_done, info

    def _gym_action_to_robot_action(self, gym_action):
        if self.action_type == ActionType.TORQUE:
            robot_action = Action(torque=gym_action, position=np.repeat(np.nan, 9))
        elif self.action_type == ActionType.POSITION:
            robot_action = Action(position=gym_action, torque=np.zeros(9))
        else:
            raise ValueError("Invalid action_type")

        return robot_action

    def scale_action(self, action, wrapped_env):
        obs = self._prev_obs
        poskey, velkey = 'robot_position', 'robot_velocity'
        current_position, current_velocity = obs[poskey], obs[velkey]
        if wrapped_env.relative:
            goal_position = current_position + .8*wrapped_env.scale * action
            pos_low, pos_high = wrapped_env.env.action_space.low, wrapped_env.env.action_space.high
        else:
            pos_low, pos_high = wrapped_env.spaces.robot_position.low, wrapped_env.spaces.robot_position.high
            pos_low = np.max([current_position - wrapped_env.scale, pos_low], axis=0)
            pos_high = np.min([current_position + wrapped_env.scale, pos_high], axis=0)
            goal_position = action
        action = np.clip(goal_position, pos_low, pos_high)
        self._clipped_action = np.abs(action - goal_position)
        return action

    def step(self, action):
        # RealRobotCubeEnv handles gym_action_to_robot_action
        #print(self.mode)
        self._last_action = action
        self.unwrapped.frameskip = self.frameskip

        # if env was originally wrapped, unwrap to see if ActionWrapper was used
        if self.is_wrapped and self.mode == PolicyMode.RL_PUSH:
            wrapped_env = self.wrapped_env
            while wrapped_env.unwrapped != wrapped_env:
                if isinstance(wrapped_env, env_wrappers.ScaledActionWrapper):
                    action = self.scale_action(action, wrapped_env)
                elif isinstance(wrapped_env, wrappers.ClipAction):
                    action = wrapped_env.action(action)
                elif isinstance(wrapped_env, gym.ActionWrapper):
                    action = wrapped_env.action(action, self._prev_obs)
                wrapped_env = wrapped_env.env

        obs, r, d, i = self._step(action)

        if self.is_wrapped and self.mode == PolicyMode.RL_PUSH:
            wrapped_env = self.wrapped_env
            while wrapped_env.unwrapped != wrapped_env:
                if isinstance(wrapped_env, env_wrappers.ReorientWrapper):
                    wrapped_env.goal_env = True
                    r += wrapped_env.is_success(obs) * wrapped_env.rew_bonus
                elif isinstance(wrapped_env, env_wrappers.CubeRewardWrapper):
                    goal_pose, prev_object_pose = self.get_goal_object_pose(self._prev_obs)
                    rl_obs = self.process_observation_rl(obs, return_dict=True)
                    _, object_pose = self.get_goal_object_pose(rl_obs)
                    wrapped_env._prev_action = action
                    r += wrapped_env._compute_reward(goal_pose, object_pose, prev_object_pose)
                wrapped_env = wrapped_env.env
        obs = self.observation(obs)
        return obs, r, d, i

    def get_goal_object_pose(self, observation):
        goal_pose = self.unwrapped.goal
        goal_pose = move_cube.Pose.from_dict(goal_pose)
        if not isinstance(observation, dict):
            observation = self.unflatten_observation(observation)
        pos, ori = observation['object_position'], observation['object_orientation'],
        object_pose = move_cube.Pose(position=pos, orientation=ori)
        return goal_pose, object_pose


@configurable(pickleable=True)
class ResidualPolicyWrapper(ObservationWrapper):
    def __init__(self, env, goal_env=False, rl_torque=True,
                 observation_names=None):
        super(ResidualPolicyWrapper, self).__init__(env)
        self.rl_torque = rl_torque
        self.goal_env = goal_env
        self.impedance_controller = None
        self._platform = None
        self.observation_names = observation_names or PushCubeEnv.observation_names
        self.observation_names.remove('action')
        self.observation_names.extend(['robot_torque', 'desired_torque',
                                       'robot_tip_forces'])
        if not self.rl_torque:
            self.observation_names.extend(['residual_tip_forces', 
                                           'desired_tip_forces'])
        assert env.action_type in [ActionType.TORQUE,
                                   ActionType.TORQUE_AND_POSITION]
        self._prev_action = np.zeros(9)
        if self.rl_torque:
            self.action_space = env.action_space.spaces['torque']
        else:
            self.action_space = gym.spaces.Box(low=-np.ones(9), high=np.ones(9))
        self.make_obs_space()

    def make_obs_space(self):
        robot_torque_space = gym.spaces.Box(
            low=trifingerpro_limits.robot_torque.low,
            high=trifingerpro_limits.robot_torque.high,
        )
        robot_position_space = gym.spaces.Box(
            low=trifingerpro_limits.robot_position.low,
            high=trifingerpro_limits.robot_position.high,
        )
        robot_velocity_space = gym.spaces.Box(
            low=trifingerpro_limits.robot_velocity.low,
            high=trifingerpro_limits.robot_velocity.high,
        )

        object_state_space = gym.spaces.Dict(
            {
                "position": gym.spaces.Box(
                    low=trifingerpro_limits.object_position.low,
                    high=trifingerpro_limits.object_position.high,
                ), "orientation": gym.spaces.Box(
                    low=trifingerpro_limits.object_orientation.low,
                    high=trifingerpro_limits.object_orientation.high,
                ),
            }
        )
        p_low = np.concatenate([object_state_space.spaces['position'].low for _ in range(3)])
        p_high = np.concatenate([object_state_space.spaces['position'].high for _ in range(3)])

        imp_obs_space = self.env.observation_space
        rl_obs_spaces = {
                "robot_position": robot_position_space,
                "robot_velocity": robot_velocity_space,
                "robot_torque": robot_torque_space,
                "robot_tip_positions": gym.spaces.Box(low=p_low, high=p_high),
                "robot_tip_forces": gym.spaces.Box(low=np.zeros(3), high=np.ones(3)),
                "applied_torque": self.action_space,
                "desired_torque": self.action_space,
                "goal_object_position": object_state_space.spaces['position'],
                "goal_object_orientation": object_state_space.spaces['orientation'],
                "object_position": object_state_space.spaces['position'],
                "object_orientation": object_state_space.spaces['orientation'],
            }
        if not self.rl_torque:
                rl_obs_spaces.update({
                    "residual_tip_forces": gym.spaces.Box(low=-np.ones(9), 
                                                          high=np.ones(9)),
                    "desired_tip_forces": gym.spaces.Box(low=-np.ones(9)*.2,
                                                         high=np.ones(9)*.2)})


        rl_obs_space = gym.spaces.Dict(
                {k: rl_obs_spaces[k] for k in self.observation_names})
        self.full_observation_space = gym.spaces.Dict(
                {'impedance': imp_obs_space, 'rl': rl_obs_space})
        self.impedance_controller = None
        if not self.goal_env:
            self.observation_space = rl_obs_space

        if robot_interfaces:
            platform = trifinger_simulation.TriFingerPlatform(
                visualization=self.visualization,
                initial_object_pose=initial_object_pose,
            )
            self.kinematics = platform.simfinger.kinematics
        else:
            self.kinematics = None

    def init_impedance_controller(self):
        init_pose, goal_pose = self.process_obs_init_goal(self._obs_dict['impedance'])
        self.impedance_controller = ImpedanceControllerPolicy(
                self.action_space, init_pose, goal_pose)
        #else:
        #    self.impedance_controller.set_init_goal(init_pose, goal_pose)
        if self._platform is None:
            self._platform = trifinger_simulation.TriFingerPlatform(
                visualization=False,
                initial_object_pose=init_pose,
            )
        self.impedance_controller.reset_policy(self._obs_dict['impedance'],
                                               self._platform)

    def _reset(self, **platform_kwargs):
        obs = super(ResidualPolicyWrapper, self).reset(**platform_kwargs)
        if self.kinematics is None:
            self.kinematics = self.platform.simfinger.kinematics
        self.init_impedance_controller()
        obs = self.grasp_object(obs)
        return obs

    def reset(self, timed=True, **platform_kwargs):
        self._des_torque = np.zeros(9)
        self._des_ft_force = np.zeros(9)
        finished = False
        if not timed:
            obs = self._reset(**platform_kwargs)
        else:
            while not finished:
                try:
                    with time_limit(20):
                        obs = self._reset(**platform_kwargs)
                    finished = True
                except TimeoutException:
                    print('resetting again, timed out')
        return obs

    def step(self, action):
        action = self.action(action)
        if not self.rl_torque:
            res_ft_force, action = action * 0.1, self._des_torque
        else:
            res_ft_force = None
        obs, r, d, i = super(ResidualPolicyWrapper, self).step(action)
        if not d:
            self._des_torque = self.impedance_controller.predict(
                    self._obs_dict['impedance'], 
                    residual_ft_force=res_ft_force)
            self._des_torque = np.clip(
                    self._des_torque, self.env.action_space.low,
                    self.env.action_space.high)
            if (not self.rl_torque and self.impedance_controller.l_wf_traj is not None
                    and self.impedance_controller.traj_waypoint_counter 
                    < self.impedance_controller.l_wf_traj.shape[0]):
                self._des_ft_force = self.impedance_controller.l_wf_traj[
                        self.impedance_controller.traj_waypoint_counter]
        return obs, r, d, i

    def grasp_object(self, obs):
        frameskip = self.unwrapped.frameskip
        self.unwrapped.frameskip = 1
        while (self.impedance_controller.mode not in 
               [TrajMode.ROTATE_X, TrajMode.ROTATE_Z, TrajMode.REPOSITION]):
            obs, _, _, _ = self.step(np.zeros(9))
        self.unwrapped.frameskip = frameskip
        return obs

    def observation(self, observation):
        obs_dict = {'impedance': observation}
        rl_obs = self.process_observation_rl(observation)
        self._obs_dict = {'impedance': observation, 'rl': rl_obs}
        if self.goal_env:
            return observation
        else:
            return rl_obs

    def process_observation_rl(self, observation):
        t = self.step_count
        robot_observation = self.platform.get_robot_observation(t)
        camera_observation = self.platform.get_camera_observation(t)
        object_observation = camera_observation.object_pose
        if self.kinematics is None:
            robot_tip_positions = self.platform.forward_kinematics(
                robot_observation.position)
        else:
            robot_tip_positions = self.kinematics.forward_kinematics(
                    robot_observation.position)
        robot_tip_positions = np.array(robot_tip_positions)
        goal_pose = self.goal
        if not isinstance(goal_pose, dict):
            goal_pose = goal_pose.to_dict()

        observation = {
            "robot_position": robot_observation.position,
            "robot_velocity": robot_observation.velocity,
            "robot_torque": robot_observation.torque,
            "robot_tip_positions": robot_tip_positions,
            "robot_tip_forces": robot_observation.tip_force,
            "object_position": object_observation.position,
            "object_orientation": object_observation.orientation,
            "goal_object_position": np.asarray(goal_pose["position"]),
            "goal_object_orientation": np.asarray(goal_pose["orientation"]),
            "desired_torque": self._des_torque,
        }
        if not self.rl_torque:
            observation.update({
                "residual_tip_forces": self._prev_action,
                "desired_tip_forces": self._des_ft_force})

        observation = {k: observation[k] for k in self.observation_names}
        return observation

    def process_obs_init_goal(self, observation):
        if self.goal_env:
            init_pose, goal_pose = observation['achieved_goal'], observation['desired_goal']
        else:
            init_pose = {'position': self._obs_dict['rl']['object_position'],
                         'orientation': self._obs_dict['rl']['object_orientation']}
            goal_pose = {'position': self._obs_dict['rl']['goal_object_position'],
                         'orientation': self._obs_dict['rl']['goal_object_orientation']}
        init_pose = move_cube.Pose.from_dict(init_pose)
        goal_pose = move_cube.Pose.from_dict(goal_pose)
        return init_pose, goal_pose

    def action(self, res_torque=None):
        obs = self._obs_dict['impedance'].copy()
        if self.rl_torque:
            self._prev_action = action = np.clip(
                    res_torque + self._des_torque, 
                    self.action_space.low, self.action_space.high)
        else:
            self._prev_action = res_torque
            action = np.clip(res_torque, self.action_space.low, 
                             self.action_space.high)
        if self.env.action_type == ActionType.TORQUE_AND_POSITION:
            return {'torque': action, 'position': np.repeat(np.nan, 9)}
        return action


def get_theta_z_wf(goal_rot, actual_rot):
    y_axis = [0, 1, 0]

    actual_direction_vector = actual_rot.apply(y_axis)

    goal_direction_vector = goal_rot.apply(y_axis)
    N = np.array([0, 0, 1]) # normal vector of ground plane
    proj = goal_direction_vector - goal_direction_vector.dot(N) * N
    goal_direction_vector = proj / np.linalg.norm(proj) # normalize projection

    orientation_error = np.arccos(
	goal_direction_vector.dot(actual_direction_vector)
    )

    return orientation_error
