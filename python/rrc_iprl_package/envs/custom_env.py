"""Custom Gym environment for the Real Robot Challenge Phase 1 (Simulation)."""
import numpy as np
import gym
import pybullet
import os.path as osp
import logging

from scipy.spatial.transform import Rotation
from gym import wrappers
from gym import ObservationWrapper
from gym.spaces import Dict
from rrc_iprl_package.control.control_policy import ImpedanceControllerPolicy, TrajMode

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
from rrc_iprl_package.envs import cube_env
from rrc_iprl_package.envs.cube_env import ActionType
from rrc_iprl_package.envs.env_wrappers import configurable
from rrc_iprl_package.control.controller_utils import PolicyMode
from rrc_iprl_package.control.control_policy import HierarchicalControllerPolicy, ImpedanceControllerPolicy


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
        save_npz=None
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
            self.goal = initializer.get_goal().to_dict()
        else:
            self.goal = cube_goal_pose
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
        self._prev_action = None
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
        goal_pose = self.goal
        if not isinstance(goal_pose, dict):
            goal_pose = goal_pose.to_dict()
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
        self.alpha = alpha
        self.filtered_position = self.filtered_orientation = None

    def write_action_log(self, observation, action, reward):
        if self.save_npz:
            self.action_log.append(dict(
                observation=observation, action=action, t=self.step_count,
                reward=reward))

    def save_action_log(self):
        if self.save_npz and self.action_log:
            np.savez(self.save_npz, initial_pose=self.initial_pose.to_dict(),
                     goal_pose=self.goal, action_log=self.action_log)
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

    def _reset_platform_frontend(self):
        """Reset the platform frontend."""
        logging.debug("Resetting simulation with robot_fingers (frontend-only)")

        # full reset is not possible is platform instantiated
        if self.platform is not None:
            logging.debug("Virtually resetting after %d resets", self.num_resets+1)
            self.num_resets += 1
        else:
            self.platform = robot_fingers.TriFingerPlatformFrontend()

    def _reset_direct_simulation(self):
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
        )

        # visualize the goal
        if self.visualization:
            goal_pose = self.goal
            if not isinstance(goal_pose, dict):
                goal_pose = self.goal.to_dict()
            self.goal_marker = trifinger_simulation.visual_objects.CuboidMarker(
                size=move_cube._CUBOID_SIZE,
                position=goal_pose["position"]+np.array([0,0,.01]),
                orientation=goal_pose["orientation"],
                pybullet_client_id=self.platform.simfinger._pybullet_client_id,
            )
            pbutils.reset_camera()

    def reset(self):
        # reset simulation
        if robot_fingers:
            self._reset_platform_frontend()
        else:
            self._reset_direct_simulation()

        if self.initializer:
            self.goal = self.initializer.get_goal().to_dict()

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
            robot_tip_positions = self.platform.forward_kinematics(
                robot_observation.position
            )
            robot_tip_positions = np.array(robot_tip_positions)
        except:
            robot_tip_positions = np.zeros(9)

        # verify that the given goal pose is contained in the cube state space
        goal_pose = self.goal
        if not isinstance(goal_pose, dict):
            goal_pose = goal_pose.to_dict()

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

    @staticmethod
    def _compute_reward(previous_observation, observation):

        # calculate first reward term
        current_distance_from_block = np.linalg.norm(
            observation["robot_tip_positions"] - observation["object_position"]
        )
        previous_distance_from_block = np.linalg.norm(
            previous_observation["robot_tip_positions"]
            - previous_observation["object_position"]
        )

        reward_term_1 = (
            previous_distance_from_block - current_distance_from_block
        )

        # calculate second reward term
        current_dist_to_goal = np.linalg.norm(
            observation["goal_object_position"]
            - observation["object_position"]
        )
        previous_dist_to_goal = np.linalg.norm(
            previous_observation["goal_object_position"]
            - previous_observation["object_position"]
        )
        reward_term_2 = previous_dist_to_goal - current_dist_to_goal

        reward = 500 * reward_term_1 + 250 * reward_term_2
        return reward

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
            previous_observation = self._create_observation(t, self._prev_action)
            observation = self._create_observation(t + 1, action)

            reward += self._compute_reward(
                previous_observation=previous_observation,
                observation=observation,
            )

        is_done = self.step_count == move_cube.episode_length
        if is_done and isinstance(self.initializer, CurriculumInitializer):
            goal_pose = self.goal
            if not isinstance(goal_pose, move_cube.Pose):
                goal_pose = move_cube.Pose.from_dict(goal_pose)
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
        self.env = self.wrapped_env = env
        # if env is wrapped, unwrap and store wrapped_env separately
        if isinstance(env, gym.Wrapper):
            self.is_wrapped = True
            self.wrapped_env = env
            self.env = env.unwrapped
        else:
            self.is_wrapped = False

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
            return self._action_space['position']

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

    def observation(self, observation):
        obs_dict = {'impedance': observation}
        if 'rl' in self.observation_space.spaces:
            observation_rl = self.process_observation_rl(observation)
            obs_dict['rl'] = observation_rl
        return obs_dict

    def process_observation_rl(self, obs):
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
                val = obs['achieved_goal']['orientation']
            elif on == 'goal_position':
                val = 0 * obs['desired_goal']['position']
            elif on == 'goal_orientation':
                # disregard x and y axis rotation for goal_orientation
                val = obs['desired_goal']['orientation']
                goal_rot = Rotation.from_quat(val)
                actual_rot = Rotation.from_quat(np.zeros([0,0,0,1]))
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
            elif on == 'action':
                val = self._last_action
                if isinstance(val, dict):
                    val = val['torque']
            obs_dict[on] = np.asarray(val, dtype='float64').flatten()

        obs = np.concatenate([obs_dict[k] for k in self.rl_observation_names])
        return obs

    def reset(self):
        if self._platform is None:
            initial_object_pose = move_cube.sample_goal(-1)
            self._platform = trifinger_simulation.TriFingerPlatform(
                visualization=False,
                initial_object_pose=initial_object_pose,
            )
        self.policy.impedance_controller.init_pinocchio_utils(self._platform)
        obs = super(HierarchicalPolicyWrapper, self).reset()
        initial_object_pose = move_cube.Pose.from_dict(obs['impedance']['achieved_goal'])
        # initial_object_pose = move_cube.sample_goal(difficulty=-1) 
        self.policy.reset_policy(obs['impedance'], self._platform)
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
        self.unwrapped.write_action_log(self.observation(observation), action,
                                        reward)
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

    def step(self, action):
        # RealRobotCubeEnv handles gym_action_to_robot_action
        #print(self.mode)
        self._last_action = action
        self.unwrapped.frameskip = self.frameskip

        # if env was originally wrapped, unwrap to see if ActionWrapper was used
        if self.is_wrapped and self.mode == PolicyMode.RL_PUSH:
            wrapped_env = self.wrapped_env
            while wrapped_env.unwrapped != wrapped_env:
                if isinstance(wrapped_env, gym.ActionWrapper):
                    action = wrapped_env.action(action)
                wrapped_env = wrapped_env.env

        obs, r, d, i = self._step(action)
        obs = self.observation(obs)
        return obs, r, d, i


class ResidualPolicyWrapper(ObservationWrapper):
    def __init__(self, env, goal_env=False, rl_torque=True, rl_tip_pos=False,
                 rl_cp_params=False,
                 observation_names=PushCubeEnv.observation_names):
        super(ResidualPolicyWrapper, self).__init__(env)
        self.rl_torque = rl_torque
        self.rl_tip_pos = rl_tip_pos
        self.rl_cp_params = rl_cp_params
        self.goal_env = goal_env
        self.impedance_controller = None
        self._platform = None
        self.observation_names = PushCubeEnv.observation_names
        self.make_obs_space()
        assert env.action_type in [ActionType.TORQUE,
                                   ActionType.TORQUE_AND_POSITION]
        if isinstance(env.action_space, gym.spaces.Dict):
            if self.rl_torque:
                self.action_space = env.action_space.spaces['torque']
                self._prev_action = np.zeros(9)
            else:
                self.action_space = gym.spaces.Box(low=-np.ones(9), high=np.ones(9))
                self._prev_action = np.zeros(9)


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
                "action": self.action_space,
                "goal_object_position": object_state_space.spaces['position'],
                "goal_object_orientation": object_state_space.spaces['orientation'],
                "object_position": object_state_space.spaces['position'],
                "object_orientation": object_state_space.spaces['orientation'],
            }

        rl_obs_space = gym.spaces.Dict(
                {k: rl_obs_spaces[k] for k in self.observation_names})
        self.full_observation_space = gym.spaces.Dict(
                {'impedance': imp_obs_space, 'rl': rl_obs_space})
        self.impedance_controller = None

        if self.goal_env:
            imp_obs_spaces = imp_obs_space.spaces
            obs_space = Dict(action=imp_obs_spaces['action'],
                             **imp_obs_spaces['observation'].spaces)
            self.observation_space = Dict(observation=obs_space,
                                          desired_goal=imp_obs_spaces['desired_goal'],
                                          achieved_goal=imp_obs_spaces['achieved_goal'])
        else:
            self.observation_space = rl_obs_space

    def reset(self):
        obs = super(ResidualPolicyWrapper, self).reset()
        self.init_impedance_controller()
        obs = self.grasp_object(obs)
        return obs

    def step(self, action):
        action = self.action(action)
        return super(ResidualPolicyWrapper, self).step(action)

    def process_observation_residual(self, observation):
        return observation

    def init_impedance_controller(self):
        init_pose, goal_pose = self.process_obs_init_goal(self._obs_dict['impedance'])
        self.impedance_controller = ImpedanceControllerPolicy(
                self.action_space, init_pose, goal_pose)
        if self._platform is None:
            self._platform = trifinger_simulation.TriFingerPlatform(
                visualization=False,
                initial_object_pose=init_pose,
            )
        self.impedance_controller.reset_policy(self._obs_dict['impedance'],
                                               self._platform)

    def grasp_object(self, obs):
        while not self.impedance_controller.mode != TrajMode.REPOSE:
            obs, _, _, _ = self.step(np.zeros(9))
        return obs

    def observation(self, observation):
        obs_dict = {'impedance': observation}
        self._obs_dict = obs_dict.copy() 
        rl_obs = self.process_observation_rl(observation)
        if self.goal_env:
            observation['observation']['action'] = observation.pop('action')
            return observation
        else:
            return rl_obs

    def process_observation_rl(self, observation):
        t = self.step_count
        robot_observation = self.platform.get_robot_observation(t)
        camera_observation = self.platform.get_camera_observation(t)
        object_observation = camera_observation.object_pose
        try:
            robot_tip_positions = self.platform.forward_kinematics(
                robot_observation.position)
            robot_tip_positions = np.array(robot_tip_positions)
        except:
            if self.impedance_controller is not None:
                robot_tip_positions = self.impedance_controller.custom_pinocchio_utils.forward_kinematics(robot_observation.position)
            else:
                print("using wrong tip pos")
                robot_tip_positions = [0.0, 0.9, -1.7, 0.0, 0.9, -1.7, 0.0, 0.9, -1.7]
            robot_tip_positions = np.array(robot_tip_positions)
        goal_pose = self.goal
        if not isinstance(goal_pose, dict):
            goal_pose = goal_pose.to_dict()

        observation = {
            "robot_position": robot_observation.position,
            "robot_velocity": robot_observation.velocity,
            "robot_tip_positions": robot_tip_positions,
            "robot_tip_forces": robot_observation.tip_force,
            "action": self._prev_action,
            "object_position": object_observation.position,
            "object_orientation": object_observation.orientation,
            "goal_object_position": np.asarray(goal_pose["position"]),
            "goal_object_orientation": np.asarray(goal_pose["orientation"]),
        }
        self._obs_dict['rl'] = observation
        # observation = np.concatenate([observation[k].flatten() for k in self.observation_names])
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
        if not self.rl_torque:
            obs['residual_ft_force'] = res_torque
        des_torque = self.impedance_controller.predict(obs)
        if self.rl_torque:
            self._prev_action = action = np.clip(res_torque + des_torque, self.action_space.low, self.action_space.high)
        else:
            self._prev_action = action = np.clip(des_torque, self.action_space.low, self.action_space.high)
        if self.env.action_type == ActionType.TORQUE_AND_POSITION:
            return {'torque': action, 'position': np.repeat(np.nan, 9)}
        return action

