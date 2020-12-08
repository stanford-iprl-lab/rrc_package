"""Custom Gym environment for the Real Robot Challenge Phase 1 (Simulation)."""
import numpy as np
import gym
import pybullet
import inspect

from gym import wrappers
from gym.spaces import Box
from gym.spaces import Discrete
from gym.spaces import MultiDiscrete
from gym.spaces import MultiBinary
from gym.spaces import Tuple
from gym.spaces import Dict
from gym.spaces import utils

from rrc_iprl_package.control import controller_utils as c_utils
from rrc_iprl_package.control.custom_pinocchio_utils import CustomPinocchioUtils
from rrc_iprl_package.envs.cube_env import CubeEnv, ActionType

from trifinger_simulation import TriFingerPlatform
from trifinger_simulation import visual_objects
from trifinger_simulation.tasks import move_cube
from scipy.spatial.transform import Rotation
from collections import deque


MAX_DIST = move_cube._max_cube_com_distance_to_center
DIST_THRESH = 0.02
_CUBOID_WIDTH = max(move_cube._CUBOID_SIZE)
_CUBOID_HEIGHT = min(move_cube._CUBOID_SIZE)

ORI_THRESH = np.pi / 8
REW_BONUS = 1
POS_SCALE = np.array([0.128, 0.134, 0.203, 0.128, 0.134, 0.203, 0.128, 0.134,
                      0.203])


def reset_camera():
    camera_pos = (0.,0.2,-0.2)
    camera_dist = 1.0
    pitch = -45.
    yaw = 0.
    if pybullet.isConnected() != 0:
        pybullet.resetDebugVisualizerCamera(cameraDistance=camera_dist,
                                    cameraYaw=yaw,
                                    cameraPitch=pitch,
                                    cameraTargetPosition=camera_pos)


def random_xy(sample_radius_min=0., sample_radius_max=None):
    # sample uniform position in circle (https://stackoverflow.com/a/50746409)
    radius = np.random.uniform(sample_radius_min, sample_radius_max)
    theta = np.random.uniform(0, 2 * np.pi)
    # x,y-position of the cube
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return x, y


def configurable(pickleable: bool = False):
    """Class decorator to allow injection of constructor arguments.

    Example usage:
    >>> @configurable()
    ... class A:
    ...     def __init__(self, b=None, c=2, d='Wow'):
    ...         ...

    >>> set_env_params(A, {'b': 10, 'c': 20})
    >>> a = A()      # b=10, c=20, d='Wow'
    >>> a = A(b=30)  # b=30, c=20, d='Wow'

    Args:
        pickleable: Whether this class is pickleable. If true, causes the pickle
            state to include the constructor arguments.
    """
    # pylint: disable=protected-access,invalid-name

    def cls_decorator(cls):
        assert inspect.isclass(cls)

        # Overwrite the class constructor to pass arguments from the config.
        base_init = cls.__init__

        def __init__(self, *args, **kwargs):

            if pickleable:
                self._pkl_env_args = args
                self._pkl_env_kwargs = kwargs

            base_init(self, *args, **kwargs)

        cls.__init__ = __init__

        # If the class is pickleable, overwrite the state methods to save
        # the constructor arguments
        if pickleable:
            # Use same pickle keys as gym.utils.ezpickle for backwards compat.
            PKL_ARGS_KEY = '_ezpickle_args'
            PKL_KWARGS_KEY = '_ezpickle_kwargs'
            def __getstate__(self):
                return {
                   PKL_ARGS_KEY: self._pkl_env_args,
                    PKL_KWARGS_KEY: self._pkl_env_kwargs,
                }
            cls.__getstate__ = __getstate__

            def __setstate__(self, data):
                saved_args = data[PKL_ARGS_KEY]
                saved_kwargs = data[PKL_KWARGS_KEY]

                inst = type(self)(*saved_args, **saved_kwargs)
                self.__dict__.update(inst.__dict__)

            cls.__setstate__ = __setstate__

        return cls

    # pylint: enable=protected-access,invalid-name
    return cls_decorator


@configurable(pickleable=True)
class FixedInitializer:
    """Initializer that uses fixed values for initial pose and goal."""

    def __init__(self, difficulty, initial_state, goal):
        """Initialize.

        Args:
            difficulty (int):  Difficulty level of the goal.  This is still
                needed even for a fixed goal, as it is also used for computing
                the reward (the cost function is different for the different
                levels).
            initial_state (move_cube.Pose):  Initial pose of the object.
            goal (move_cube.Pose):  Goal pose of the object.

        Raises:
            Exception:  If initial_state or goal are not valid.  See
            :meth:`move_cube.validate_goal` for more information.
        """
        move_cube.validate_goal(initial_state)
        move_cube.validate_goal(goal)
        self.difficulty = difficulty
        self.initial_state = initial_state
        self.goal = goal

    def get_initial_state(self):
        """Get the initial state that was set in the constructor."""
        return self.initial_state

    def get_goal(self):
        """Get the goal that was set in the constructor."""
        return self.goal


@configurable(pickleable=True)
class CurriculumInitializer:
    """Initializer that samples random initial states and goals."""

    def __init__(self, difficulty=1, initial_dist=_CUBOID_HEIGHT,
                 num_levels=4, buffer_size=5, fixed_goal=None):
        """Initialize.

        Args:
            initial_dist (float): Distance from center of arena
            num_levels (int): Number of steps to maximum radius
            buffer_size (int): Number of episodes to compute mean over
        """
        self.difficulty = difficulty
        self.num_levels = num_levels
        self._current_level = 0
        self.levels = np.linspace(initial_dist, MAX_DIST, num_levels)
        self.final_dist = np.array([np.inf for _ in range(buffer_size)])
        if difficulty == 4:
            self.final_ori = np.array([np.inf for _ in range(buffer_size)])
        self.fixed_goal = fixed_goal

    @property
    def current_level(self):
        return min(self.num_levels - 1, self._current_level)

    def random_xy(self, sample_radius_min=0., sample_radius=None):
        # sample uniform position in circle (https://stackoverflow.com/a/50746409)
        sample_radius_max = sample_radius or self.levels[self.current_level]
        return random_xy(sample_radius_min, sample_radius_max)

    def update_initializer(self, final_pose, goal_pose):
        assert np.all(goal_pose.position == self.goal_pose.position)
        self.final_dist = np.roll(self.final_dist, 1)
        final_dist = np.linalg.norm(goal_pose.position - final_pose.position)
        self.final_dist[0] = final_dist
        if self.difficulty == 4:
            self.final_ori = np.roll(self.final_ori, 1)
            self.final_ori[0] = compute_orientation_error(goal_pose, final_pose)

        update_level = np.mean(self.final_dist) < DIST_THRESH
        if self.difficulty == 4:
            update_level = update_level and np.mean(self.final_ori) < ORI_THRESH

        if update_level and self._current_level < self.num_levels - 1:
            pre_sample_dist = self.goal_sample_radius
            self._current_level += 1
            post_sample_dist = self.goal_sample_radius
            print("Old sampling distances: {}/New sampling distances: {}".format(
                pre_sample_dist, post_sample_dist))

    def get_initial_state(self):
        """Get a random initial object pose (always on the ground)."""
        x, y = self.random_xy()
        self.initial_pose = move_cube.sample_goal(difficulty=-1)
        z = self.initial_pose.position[-1]
        self.initial_pose.position = np.array((x, y, z))
        return self.initial_pose

    @property
    def goal_sample_radius(self):
        if self.fixed_goal:
            goal_dist = np.linalg.norm(self.fixed_goal.position)
            return (goal_dist, goal_dist)
        sample_radius_min = 0.
        sample_radius_max = self.levels[min(self.num_levels - 1, self._current_level + 1)]
        return (sample_radius_min, sample_radius_max)

    def get_goal(self):
        """Get a random goal depending on the difficulty."""
        if self.fixed_goal:
            self.goal_pose = self.fixed_goal
            return self.fixed_goal
        # goal_sample_radius is further than past distances
        sample_radius_min, sample_radius_max = self.goal_sample_radius
        x, y = self.random_xy(sample_radius_min, sample_radius_max)
        self.goal_pose = move_cube.sample_goal(difficulty=self.difficulty)
        self.goal_pose.position = np.array((x, y, self.goal_pose.position[-1]))
        return self.goal_pose


@configurable(pickleable=True)
class ReorientInitializer:
    """Initializer that samples random initial states and goals."""
    def_goal_pose = move_cube.Pose(np.array([0,0,_CUBOID_HEIGHT/2]), np.array([0,0,0,1]))

    def __init__(self, difficulty=1, initial_dist=_CUBOID_HEIGHT, seed=None):
        self.difficulty = difficulty
        self.initial_dist = initial_dist
        self.goal_pose = self.def_goal_pose
        self.set_seed(seed)

    def set_seed(self, seed):
        self.random = np.random.RandomState(seed=seed)

    def get_initial_state(self):
        """Get a random initial object pose (always on the ground)."""
        x, y = random_xy(self.initial_dist, MAX_DIST)
        self.initial_pose = move_cube.sample_goal(difficulty=-1)
        z = self.initial_pose.position[-1]
        self.initial_pose.position = np.array((x, y, z))
        return self.initial_pose

    def get_goal(self):
        """Get a random goal depending on the difficulty."""
        if self.difficulty >= 2:
            self.goal_pose = move_cube.sample_goal(self.difficulty)
        return self.goal_pose


class RandomGoalOrientationInitializer:
    init_pose = move_cube.Pose(np.array([0,0,_CUBOID_HEIGHT/2]), np.array([0,0,0,1]))

    def __init__(self, difficulty=1, max_dist=np.pi):
        self.difficulty = difficulty
        self.max_dist = max_dist
        self.random = np.random.RandomState()

    def get_initial_state(self):
        return self.init_pose

    def get_goal(self):
        goal =  move_cube.sample_goal(-1)
        goal.position = np.zeros(3)
        return goal


class RandomOrientationInitializer:
    goal = move_cube.Pose(np.array([0,0,_CUBOID_HEIGHT/2]), np.array([0,0,0,1]))

    def __init__(self, difficulty=4):
        self.difficulty = difficulty

    def get_initial_state(self):
        return move_cube.sample_goal(-1)

    def get_goal(self):
        return self.goal


@configurable(pickleable=True)
class SparseCubeEnv(CubeEnv):
    def __init__(
            self,
            initializer,
            action_type=ActionType.POSITION,
            frameskip=1,
            visualization=False,
            pos_thresh=DIST_THRESH,
            ori_thresh=ORI_THRESH
            ):
        super(SparseCubeEnv, self).__init__(initializer, action_type,
                frameskip, visualization)
        self.pos_thresh = pos_thresh
        self.ori_thresh = ori_thresh

    def compute_reward(self, achieved_goal, desired_goal, info):
        goal_pose = move_cube.Pose.from_dict(desired_goal)
        obj_pose = move_cube.Pose.from_dict(achieved_goal)
        pos_error = np.linalg.norm(goal_pose.position - obj_pose.position)
        ori_error = compute_orientation_error(goal_pose, obj_pose)
        return float(pos_error < self.pos_thresh and ori_error < self.ori_thresh)


@configurable(pickleable=True)
class TaskSpaceWrapper(gym.ActionWrapper):
    def __init__(self, env, goal_env=False, relative=False, scale=.008, ac_pen=0.001,
                 save_npz=None):
        super(TaskSpaceWrapper, self).__init__(env)
        if hasattr(self.unwrapped, 'save_npz'):
            self._save_npz = self.unwrapped.save_npz
            self.unwrapped.save_npz = None
            self.action_log = self.unwrapped.action_log
        else:
            self.action_log = []
            self._save_npz = save_npz

        # assert self.unwrapped.action_type in [ActionType.POSITION, ActionType.TORQUE]
        spaces = TriFingerPlatform.spaces
        self.goal_env = goal_env
        self.relative = relative
        low = np.array([spaces.object_position.low]* 3).flatten()
        high = np.array([spaces.object_position.high]* 3).flatten()
        if relative:
            low = -np.ones_like(low)
            high = np.ones_like(high)
        self.action_space = gym.spaces.Box(low=low, high=high)
        self.scale = scale
        self.pinocchio_utils = None
        self.ac_pen = ac_pen
        if self.goal_env:
            obs_space = self.observation_space.spaces['observation']
            obs_dict = obs_space.spaces
        else:
            obs_dict = self.observation_space.spaces
        obs_dict['last_action'] = self.action_space

    def reset(self):
        if hasattr(self.unwrapped, 'save_npz'):
            self.unwrapped.save_npz = self._save_npz
        obs = super(TaskSpaceWrapper, self).reset()
        if hasattr(self.unwrapped, 'save_npz'):
            self.unwrapped.save_npz = None
        platform = self.unwrapped.platform
        if self.pinocchio_utils is None:
            self.pinocchio_utils = CustomPinocchioUtils(
                    platform.simfinger.finger_urdf_path,
                    platform.simfinger.tip_link_names)
        self._prev_obs = obs
        self._last_action = np.zeros_like(self.action_space.sample())
        obs_dict = obs
        if self.goal_env:
            obs_dict = obs['observation']
        obs_dict['last_action'] = self._last_action
        return obs

    def write_action_log(self, observation, action, reward):
        if self._save_npz:
            self.action_log.append(dict(
                observation=observation, rl_action=action,
                action=self.action(action), t=self.step_count,
                reward=reward))

    def step(self, action):
        o, r, d, i = super(TaskSpaceWrapper, self).step(action)
        self.write_action_log(o, action, r)
        self._prev_obs = o
        if self.relative:
            r -= self.ac_pen * np.linalg.norm(action)
        else:
            r -= self.ac_pen * np.linalg.norm(self._last_action - action)
        self._last_action =  action
        obs_dict = o
        if self.goal_env:
            obs_dict = obs_dict['observation']
        obs_dict['last_action'] = self._last_action
        return o, r, d, i

    def action(self, action):
        obs = self._prev_obs
        poskey, velkey = 'robot_position', 'robot_velocity'
        if self.goal_env:
            obs, poskey, velkey = obs['observation'], 'position', 'velocity'
        current_position, current_velocity = obs[poskey], obs[velkey]

        if self.relative:
            fingertip_goals = self.pinocchio_utils.forward_kinematics(current_position.flatten())
            fingertip_goals = np.asarray(fingertip_goals)
            fingertip_goals = fingertip_goals + self.scale * action.reshape((3,3))
        else:
            fingertip_goals = action
        if self.unwrapped.action_type == ActionType.TORQUE:
            dt = self.frameskip * .001
            desired_velocity = self.scale * action.reshape((3,3)) / dt
            # TODO: use tip_forces_wf to indicate desired contact with object on fingertip
            torque = c_utils.impedance_controller(
                    tip_pos_desired_list=fingertip_goals, tip_vel_desired_list=desired_velocity,
                    q_current=current_position, dq_current=current_velocity,
                    custom_pinocchio_utils=self.pinocchio_utils, tip_forces_wf=None)
            ac = np.clip(torque, self.unwrapped.action_space.low,
                             self.unwrapped.action_space.high)
        else:
            ac, ft_err = self.pinocchio_utils.inverse_kinematics(fingertip_goals,
                                                                 current_position)
        return ac


@configurable(pickleable=True)
class ScaledActionWrapper(gym.ActionWrapper):
    def __init__(self, env, goal_env=False, relative=True, scale=POS_SCALE,
                 lim_penalty=0.0):
        super(ScaledActionWrapper, self).__init__(env)
        self._save_npz = self.unwrapped.save_npz
        self.unwrapped.save_npz = None
        assert self.unwrapped.action_type == ActionType.POSITION, 'position control only'
        self.spaces = TriFingerPlatform.spaces
        self.goal_env = goal_env
        self.relative = relative
        low = self.action_space.low
        high = self.action_space.high
        if relative:
            low = -np.ones_like(low)
            high = np.ones_like(high)
        self.action_space = gym.spaces.Box(low=low, high=high)
        self.scale = scale
        self.lim_penalty = lim_penalty
        self.action_log = []

    def write_action_log(self, observation, action, reward):
        if self._save_npz:
            self.action_log.append(dict(
                observation=observation, action=action,
                scaled_action=self.action(action), t=self.step_count,
                reward=reward))

    def reset(self):
        self.unwrapped.save_npz = self._save_npz
        obs = super(ScaledActionWrapper, self).reset()
        self.unwrapped.save_npz = None
        self._prev_obs = obs
        self._clipped_action = self._last_action = np.zeros_like(self.action_space.sample())
        return obs

    def step(self, action):
        o, r, d, i = super(ScaledActionWrapper, self).step(action)
        self.write_action_log(o, action, r)
        self._prev_obs = o
        self._last_action =  action
        r += np.sum(self._clipped_action) * self.lim_penalty
        return o, r, d, i

    def action(self, action):
        obs = self._prev_obs
        poskey, velkey = 'robot_position', 'robot_velocity'
        if self.goal_env:
            obs, poskey, velkey = obs['observation'], 'position', 'velocity'
        current_position, current_velocity = obs[poskey], obs[velkey]
        if self.relative:
            goal_position = current_position + self.scale * action
            pos_low, pos_high = self.env.action_space.low, self.env.action_space.high
        else:
            pos_low, pos_high = self.spaces.robot_position.low, self.spaces.robot_position.high
            pos_low = np.max([current_position - self.scale, pos_low], axis=0)
            pos_high = np.min([current_position + self.scale, pos_high], axis=0)
            goal_position = action
        action = np.clip(goal_position, pos_low, pos_high)
        self._clipped_action = np.abs(action - goal_position)
        return action


@configurable(pickleable=True)
class RelativeGoalWrapper(gym.ObservationWrapper):
    def __init__(self, env, keep_goal=False):
        super(RelativeGoalWrapper, self).__init__(env)
        self._observation_keys = list(env.observation_space.spaces.keys())
        assert 'goal_object_position' in self._observation_keys, 'goal_object_position missing in observation'
        self.position_only = 'goal_object_orientation' not in self._observation_keys
        self.observation_names =  [k for k in self._observation_keys if 'goal_object' not in k]
        self.observation_names.append('relative_goal_object_position')
        if keep_goal:
            self.observation_names.append('goal_object_position')
        if not self.position_only:
            self.observation_names.append('relative_goal_object_orientation')
            if keep_goal:
                self.observation_names.append('goal_object_orientation')
        obs_dict = {}
        for k in self.observation_names:
            if 'relative_goal_object' not in k:
                obs_dict[k] = env.observation_space.spaces[k]
            elif k == 'relative_goal_object_position':
                high = env.observation_space['goal_object_position'].high - env.observation_space['goal_object_position'].low
                low = -high
                obs_dict[k] = gym.spaces.Box(low=low, high=high)
            elif k == 'relative_goal_object_orientation':
                high = env.observation_space['goal_object_orientation'].high - env.observation_space['goal_object_orientation'].low
                low = -high
                obs_dict[k] = gym.spaces.Box(low=low, high=high)
        self.observation_space = gym.spaces.Dict(obs_dict)

    def observation(self, obs):
        obs_dict = {k: obs[k] for k in self.observation_names if k in self._observation_keys}
        obs_dict['relative_goal_object_position'] = obs['goal_object_position'] - obs['object_position']
        if not self.position_only:
            offset = (Rotation.from_quat(obs['goal_object_orientation'])
                      * Rotation.from_quat(obs['object_orientation']).inv()).as_quat()
            obs_dict['relative_goal_object_orientation'] = offset
        return obs_dict


@configurable(pickleable=True)
class ReorientWrapper(gym.Wrapper):
    def __init__(self, env, goal_env=True, rew_bonus=REW_BONUS,
                 dist_thresh=0.09, ori_thresh=np.pi/6):
        super(ReorientWrapper, self).__init__(env)
        if not isinstance(self.unwrapped.initializer, ReorientInitializer):
            initializer = ReorientInitializer(initial_dist=0.1)
            self.unwrapped.initializer = initializer
        self.goal_env = goal_env
        self.rew_bonus = rew_bonus
        self.dist_thresh = dist_thresh
        self.ori_thresh = ori_thresh

    def step(self, action):
        o, r, d, i = super(ReorientWrapper, self).step(action)
        i['is_success'] = self.is_success(o)
        if i['is_success']:
            r += self.rew_bonus
        return o, r, d, i

    def is_success(self, observation):
        if self.goal_env:
            goal_pose = move_cube.Pose.from_dict(observation['desired_goal'])
            obj_pose = move_cube.Pose.from_dict(observation['achieved_goal'])
        else:
            goal_pose = move_cube.Pose.from_dict(self.unwrapped.goal)
            obj_pose = move_cube.Pose.from_dict(
                    dict(position=observation['object_position'],
                         orientation=observation['object_orientation']))

        obj_dist = np.linalg.norm(obj_pose.position - goal_pose.position)
        ori_dist = compute_orientation_error(goal_pose, obj_pose)
        return obj_dist < self.dist_thresh and ori_dist < self.ori_thresh


class FlattenGoalWrapper(gym.ObservationWrapper):
    """Wrapper to make rrc env baselines and VDS compatible"""
    def __init__(self, env):
        super(FlattenGoalWrapper, self).__init__(env)
        self._sample_goal_fun = None
        self._max_episode_steps = env._max_episode_steps
        self.observation_space = gym.spaces.Dict({
            k: flatten_space(v)
            for k, v in env.observation_space.spaces.items()
            })

    def update_goal_sampler(self, goal_sampler):
        self._sample_goal_fun = goal_sampler

    def sample_goal_fun(self, **kwargs):
        return self._sample_goal_fun(**kwargs)

    @property
    def goal(self):
        return np.concatenate([self.unwrapped.goal['position'],
                               self.unwrapped.goal['orientation']])

    @goal.setter
    def goal(self, g):
        if isinstance(g, dict):
            self.unwrapped.goal = g
            return
        pos, ori = g[...,:3], g[...,3:]
        self.unwrapped.goal = {'position': pos, 'orientation': ori}

    def compute_reward(self, achieved_goal, desired_goal, info):
        if len(achieved_goal.shape) > 1:
            r = []
            info = {"difficulty": self.initializer.difficulty}
            for i in range(achieved_goal.shape[0]):
                pos, ori = achieved_goal[i,:3], achieved_goal[i,3:]
                ag = dict(position=pos, orientation=ori)
                pos, ori = desired_goal[i,:3], desired_goal[i,3:]
                dg = dict(position=pos, orientation=ori)
                r.append(self.env.compute_reward(ag, dg, info))
            return np.array(r)
        achieved_goal = dict(position=achieved_goal[...,:3], orientation=achieved_goal[...,3:])
        desired_goal = dict(position=desired_goal[...,:3], orientation=desired_goal[...,3:])
        return self.env.compute_reward(achieved_goal, desired_goal, info)

    def _sample_goal(self):
        return np.concatenate(list(self.initializer.get_goal().to_dict().values()))

    def observation(self, observation):
        observation = {k: gym.spaces.flatten(self.env.observation_space[k], v)
                       for k, v in observation.items()}
        return observation


# DEPRECATED, USE CubeRewardWrapper INSTEAD
class DistRewardWrapper(gym.RewardWrapper):
    def __init__(self, env, target_dist=0.2, dist_coef=1., ori_coef=1.,
                 ac_norm_pen=0.2, final_step_only=True, augment_reward=True,
                 rew_fn='lin'):
        super(DistRewardWrapper, self).__init__(env)
        self._target_dist = target_dist  # 0.156
        self._dist_coef = dist_coef
        self._ori_coef = ori_coef
        self._ac_norm_pen = ac_norm_pen
        self._last_action = None
        self.final_step_only = final_step_only
        self.augment_reward = augment_reward
        self.rew_fn = rew_fn
        print('DistRewardWrapper is deprecated, use CubeRewardWrapper instead')

    @property
    def target_dist(self):
        target_dist = self._target_dist
        if target_dist is None:
            if isinstance(self.initializer, CurriculumInitializer):
                _, target_dist = self.initializer.goal_sample_radius
                target_dist = 2 * target_dist  # use sample diameter
            else:
                target_dist = move_cube._ARENA_RADIUS
        return target_dist

    @property
    def difficulty(self):
        return self.unwrapped.initializer.difficulty

    def reset(self, **reset_kwargs):
        self._last_action = None
        return super(DistRewardWrapper, self).reset(**reset_kwargs)

    def step(self, action):
        self._last_action = action
        observation, reward, done, info = self.env.step(action)
        if self.final_step_only and done:
            return observation, reward, done, info
        else:
            return observation, self.reward(reward, info), done, info

    def reward(self, reward, info):
        final_dist = self.compute_goal_dist(info)
        if self.rew_fn == 'lin':
            rew = self._dist_coef * (1 - final_dist/self.target_dist)
            if self.info['difficulty'] == 4:
                rew += self._ori_coef * (1 - self.compute_orientation_error())
        elif self.rew_fn == 'exp':
            rew = self._dist_coef * np.exp(-final_dist/self.target_dist)
            if self.info['difficulty'] == 4:
                rew += self._ori_coef * np.exp(-self.compute_orientation_error())
        if self.augment_reward:
            rew += reward
        if self._ac_norm_pen:
            rew -= np.linalg.norm(self._last_action) * self._ac_norm_pen
        return rew

    def get_goal_object_pose(self):
        goal_pose = self.unwrapped.goal
        if not isinstance(goal_pose, move_cube.Pose):
            goal_pose = move_cube.Pose.from_dict(goal_pose)
        cube_state = self.unwrapped.platform.cube.get_state()
        object_pose = move_cube.Pose(
                np.asarray(cube_state[0]).flatten(),
                np.asarray(cube_state[1]).flatten())
        return goal_pose, object_pose

    def compute_orientation_error(self):
        goal_pose, object_pose = self.get_goal_object_pose()
        orientation_error = compute_orientation_error(goal_pose, object_pose)
        return orientation_error

    def compute_goal_dist(self, info):
        goal_pose, object_pose = self.get_goal_object_pose()
        pos_idx = 3 if info['difficulty'] > 3 else 2
        goal_dist = np.linalg.norm(object_pose.position[:pos_idx] -
                                   goal_pose.position[:pos_idx])
        return goal_dist


class CubeRewardWrapper(gym.Wrapper):
    def __init__(self, env, target_dist=0.156, pos_coef=1., ori_coef=0.,
                 fingertip_coef=0., ac_norm_pen=0.2, goal_env=False, rew_fn='exp',
                 augment_reward=False):
        super(CubeRewardWrapper, self).__init__(env)
        self._target_dist = target_dist
        self._pos_coef = pos_coef
        self._ori_coef = ori_coef
        self._fingertip_coef = fingertip_coef
        self._ac_norm_pen = ac_norm_pen
        self._goal_env = goal_env
        self._prev_action = None
        self._prev_obs = None
        self._augment_reward = augment_reward
        self.rew_fn = rew_fn
        self.obj_pos = deque(maxlen=20)
        self.obj_ori = deque(maxlen=20)

    @property
    def target_dist(self):
        target_dist = self._target_dist
        if target_dist is None:
            if isinstance(self.initializer, CurriculumInitializer):
                _, target_dist = self.initializer.goal_sample_radius
                target_dist = 2 * target_dist  # use sample diameter
            else:
                target_dist = move_cube._ARENA_RADIUS
        return target_dist

    @property
    def difficulty(self):
        return self.unwrapped.initializer.difficulty

    def reset(self, **reset_kwargs):
        self._prev_action = None
        self._prev_obs = super(CubeRewardWrapper, self).reset(**reset_kwargs)
        return self._prev_obs

    def step(self, action):
        self._prev_action = action
        observation, r, done, info = super(CubeRewardWrapper, self).step(action)
        if self._goal_env:
            reward = self.compute_reward(observation['achieved_goal'],
                                         observation['desired_goal'], info)
            if self._fingertip_coef:
                reward += self.compute_fingertip_reward(observation['observation'],
                                                  self._prev_obs['observation'])
        else:
            goal_pose, prev_object_pose = self.get_goal_object_pose(self._prev_obs)
            goal_pose, object_pose = self.get_goal_object_pose(observation)
            reward = self._compute_reward(goal_pose, object_pose, prev_object_pose)
            if self._fingertip_coef:
                reward += self.compute_fingertip_reward(observation, self._prev_obs)
        if self._augment_reward:
            reward += r

        self._prev_obs = observation
        return observation, reward, done, info

    def compute_fingertip_reward(self, observation, previous_observation):
        if not isinstance(observation, dict):
            obs_space = self.unwrapped.observation_space
            if self._goal_env: obs_space = obs_space.spaces['observation']
            observation = self.unflatten_observation(observation, obs_space)
            previous_observation = self.unflatten_observation(previous_observation, obs_space)

        if 'robot_tip_position' in observation:
            prev_ftip_pos = previous_observation['robot_tip_position']
            curr_ftip_pos = observation['robot_tip_position']
        else:
            prev_ftip_pos = self.platform.forward_kinematics(previous_observation['robot_position'])
            curr_ftip_pos = self.platform.forward_kinematics(observation['robot_position'])

        current_distance_from_block = np.linalg.norm(
           curr_ftip_pos - observation["object_position"]
        )
        previous_distance_from_block = np.linalg.norm(
            prev_ftip_pos
            - previous_observation["object_position"]
        )

        step_ftip_rew = (
            previous_distance_from_block - current_distance_from_block
        )
        return self._fingertip_coef * step_ftip_rew

    def compute_reward(self, achieved_goal, desired_goal, info):
        if isinstance(achieved_goal, dict):
            obj_pos, obj_ori = achieved_goal['position'], achieved_goal['orientation']
            goal_pos, goal_ori = desired_goal['position'], desired_goal['orientation']
        else:
            obj_pos, obj_ori = achieved_goal[:3], achieved_goal[3:]
            goal_pos, goal_ori = desired_goal[:3], desired_goal[3:]
        self.obj_pos.append(obj_pos)
        self.obj_ori.append(obj_ori)
        obj_pos, obj_ori = np.mean(self.obj_pos, axis=0), np.mean(self.obj_ori, axis=0)
        goal_pose = move_cube.Pose(position=goal_pos, orientation=goal_ori)
        object_pose = move_cube.Pose(position=obj_pos, orientation=obj_ori)
        return self._compute_reward(goal_pose, object_pose, info=info)

    def _compute_reward(self, goal_pose, object_pose, prev_object_pose=None, info=None):
        info = info or self.unwrapped.info
        pos_error = self.compute_position_error(goal_pose, object_pose)

        # compute previous object pose error
        if prev_object_pose is not None:
            prev_pos_error = self.compute_position_error(goal_pose, prev_object_pose)
            step_rew = step_pos_rew = prev_pos_error - pos_error
        else:
            step_rew = 0

        if self.difficulty == 4 or self._ori_coef:
            ori_error = compute_orientation_error(goal_pose, object_pose)
            if prev_object_pose is not None:
                prev_ori_error = compute_orientation_error(goal_pose, prev_object_pose)
                step_ori_rew = prev_ori_error - ori_error
                step_rew = (step_pos_rew * self._pos_coef +
                            step_ori_rew * self._ori_coef)
            else:
                step_rew = 0

        # compute position and orientation joint reward based on chosen rew_fn 
        if self.rew_fn == 'lin':
            rew = self._pos_coef * (1 - pos_error/self.target_dist)
            if self.difficulty == 4 or self._ori_coef:
                rew += self._ori_coef * (1 - ori_error)
        elif self.rew_fn == 'exp':
            # pos error penalty
            if pos_error >= self.target_dist:
                rew = -1
            else:
                rew = self._pos_coef * np.exp(-pos_error/self.target_dist)
            if self.difficulty == 4 or self._ori_coef:
                rew += self._ori_coef * np.exp(-ori_error)

        # Add to info dict
        # compute action penalty
        ac_penalty = -np.linalg.norm(self._prev_action) * self._ac_norm_pen
        info['ac_penalty'] = ac_penalty
        if step_rew:
            info['step_rew'] = step_rew
        info['rew'] = rew
        info['pos_error'] = pos_error
        if self.difficulty == 4 or self._ori_coef:
            info['ori_error'] = ori_error
        total_rew = step_rew + rew + ac_penalty
        # if pos and ori error are below threshold 
        if pos_error < DIST_THRESH or ori_error < ORI_THRESH:
            return 2.5 * ((pos_error < DIST_THRESH) + (ori_error < ORI_THRESH))
        return total_rew

    def unflatten_observation(self, observation, obs_space=None):
        filter_keys = []
        env = self.env
        while env != self.unwrapped:
            if isinstance(env, wrappers.FilterObservation):
                filter_keys = env._filter_keys
            env = env.env

        obs_space = obs_space or self.unwrapped.observation_space
        if isinstance(obs_space, gym.spaces.Dict):
            if filter_keys:
                obs_space = gym.spaces.Dict({obs_space[k] for k in filter_keys})
            observation = utils.unflatten(obs_space, observation)
        return observation

    def get_goal_object_pose(self, observation):
        goal_pose = self.unwrapped.goal
        goal_pose = move_cube.Pose.from_dict(goal_pose)
        if not self._goal_env:
            if not isinstance(observation, dict):
                observation = self.unflatten_observation(observation)
            pos, ori = observation['object_position'], observation['object_orientation'],
        elif 'achieved_goal' in observation:
            pos, ori = observation['achieved_goal'][:3], observation['achieved_goal'][3:]
        object_pose = move_cube.Pose(position=pos,
                                     orientation=ori)
        return goal_pose, object_pose

    def compute_position_error(self, goal_pose, object_pose):
        pos_error = np.linalg.norm(object_pose.position - goal_pose.position)
        return pos_error


class LogInfoWrapper(gym.Wrapper):
    valid_keys = ['dist', 'score', 'ori_dist', 'ori_scaled',
                  'is_success', 'is_success_ori', 'is_success_ori_dist']

    def __init__(self, env, info_keys=[]):
        super(LogInfoWrapper, self).__init__(env)
        if isinstance(env.initializer, CurriculumInitializer):
            new_keys = ['init_sample_radius','goal_sample_radius']
            [self.valid_keys.append(k) for k in new_keys if k not in self.valid_keys]
        for k in info_keys:
            assert k.split('final_')[-1] in self.valid_keys, f'{k} is not a valid key'
        self.info_keys = info_keys

    def get_goal_object_pose(self):
        goal_pose = self.unwrapped.goal
        if not isinstance(goal_pose, move_cube.Pose):
            goal_pose = move_cube.Pose.from_dict(goal_pose)
        cube_state = self.unwrapped.platform.cube.get_state()
        object_pose = move_cube.Pose(
                np.asarray(cube_state[0]).flatten(),
                np.asarray(cube_state[1]).flatten())
        return goal_pose, object_pose

    def compute_position_error(self, info, score=False):
        goal_pose, object_pose = self.get_goal_object_pose()
        if score:
            return move_cube.evaluate_state(goal_pose, object_pose,
                                            info['difficulty'])
        pos_idx = 3 if info['difficulty'] > 3 else 2
        return np.linalg.norm(object_pose.position[:pos_idx] -
                              goal_pose.position[:pos_idx])

    def compute_orientation_error(self, info):
        goal_pose, object_pose = self.get_goal_object_pose()
        return compute_orientation_error(goal_pose, object_pose)

    def step(self, action):
        o, r, d, i = super(LogInfoWrapper, self).step(action)
        for k in self.info_keys:
            if k not in i:
                shortened_k = k.split('final_')[-1]
                final = shortened_k != k
                if shortened_k == 'score' and final == d:
                    i[k] = self.compute_position_error(i, score=True)
                elif shortened_k == 'dist' and final == d:
                    i[k] = self.compute_position_error(i, score=False)
                elif shortened_k == 'ori_dist' and final == d:
                    i[k] = self.compute_orientation_error(i)
                elif shortened_k == 'ori_scaled' and final ==  d:
                    i[k] = self.compute_orientation_error(i)
                elif k == 'is_success' and d:
                    i[k] = self.compute_position_error(i) < DIST_THRESH
                elif k == 'is_success_ori' and d:
                    if 'is_success' not in self.info_keys and 'is_success' not in i:
                        k = 'is_success'
                    i[k] = self.compute_orientation_error(i) < ORI_THRESH
                elif k == 'is_success_ori_dist' and d:
                    if 'is_success' not in self.info_keys and 'is_success' not in i:
                        k = 'is_success'
                    i[k] = (self.compute_orientation_error(i) < ORI_THRESH and
                            self.compute_position_error(i) < DIST_THRESH)
                elif k == 'init_sample_radius' and d:
                    initializer = self.unwrapped.initializer
                    sample_radius = np.linalg.norm(initializer.initial_pose.position[:2])
                    i[k] = sample_radius
                elif k == 'goal_sample_radius' and d:
                    initializer = self.unwrapped.initializer
                    sample_radius = np.linalg.norm(initializer.goal_pose.position[:2])
                    i[k] = sample_radius
        self.info = i
        return o, r, d, self.info


class StepRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super(StepRewardWrapper, self).__init__(env)
        self._last_rew = 0.

    def reset(self):
        self._last_rew = 0.
        return super(StepRewardWrapper, self).reset()

    def reward(self, reward):
        step_reward = reward - self._last_rew
        self._last_rew = reward
        return step_reward


# Phase 3 orientation error
def compute_orientation_error(goal_pose, actual_pose):
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


def compute_orientation_error_old(goal_pose, actual_pose, scale=False,
                              yaw_only=False, quad=False):
    if yaw_only:
        goal_ori = Rotation.from_quat(goal_pose.orientation).as_euler('xyz')
        goal_ori[:2] = 0
        goal_rot = Rotation.from_euler('xyz', goal_ori)
        actual_ori = Rotation.from_quat(actual_pose.orientation).as_euler('xyz')
        actual_ori[:2] = 0
        actual_rot = Rotation.from_euler('xyz', actual_ori)
    else:
        goal_rot = Rotation.from_quat(goal_pose.orientation)
        actual_rot = Rotation.from_quat(actual_pose.orientation)
    error_rot = goal_rot.inv() * actual_rot
    orientation_error = error_rot.magnitude()
    # computes orientation error symmetric to 4 quadrants of the cube
    if quad:
        orientation_error = orientation_error % (np.pi/2)
        if orientation_error > np.pi/4:
            orientation_error = np.pi/2 - orientation_error
    if scale:
        orientation_error = orientation_error / np.pi
    return orientation_error


def flatten_space(space):
    """Flatten a space into a single ``Box``.
    This is equivalent to ``flatten()``, but operates on the space itself. The
    result always is a `Box` with flat boundaries. The box has exactly
    ``flatdim(space)`` dimensions. Flattening a sample of the original space
    has the same effect as taking a sample of the flattenend space.
    Raises ``NotImplementedError`` if the space is not defined in
    ``gym.spaces``.
    Example::
        >>> box = Box(0.0, 1.0, shape=(3, 4, 5))
        >>> box
        Box(3, 4, 5)
        >>> flatten_space(box)
        Box(60,)
        >>> flatten(box, box.sample()) in flatten_space(box)
        True
    Example that flattens a discrete space::
        >>> discrete = Discrete(5)
        >>> flatten_space(discrete)
        Box(5,)
        >>> flatten(box, box.sample()) in flatten_space(box)
        True
    Example that recursively flattens a dict::
        >>> space = Dict({"position": Discrete(2),
        ...               "velocity": Box(0, 1, shape=(2, 2))})
        >>> flatten_space(space)
        Box(6,)
        >>> flatten(space, space.sample()) in flatten_space(space)
        True
    """
    if isinstance(space, Box):
        return Box(space.low.flatten(), space.high.flatten())
    if isinstance(space, Discrete):
        return Box(low=0, high=1, shape=(space.n, ))
    if isinstance(space, Tuple):
        space = [flatten_space(s) for s in space.spaces]
        return Box(
            low=np.concatenate([s.low for s in space]),
            high=np.concatenate([s.high for s in space]),
        )
    if isinstance(space, Dict):
        space = [flatten_space(s) for s in space.spaces.values()]
        return Box(
            low=np.concatenate([s.low for s in space]),
            high=np.concatenate([s.high for s in space]),
        )
    if isinstance(space, MultiBinary):
        return Box(low=0, high=1, shape=(space.n, ))
    if isinstance(space, MultiDiscrete):
        return Box(
            low=np.zeros_like(space.nvec),
            high=space.nvec,
        )
    raise NotImplementedError

