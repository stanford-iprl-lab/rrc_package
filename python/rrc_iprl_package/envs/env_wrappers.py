"""Custom Gym environment for the Real Robot Challenge Phase 0 (Simulation)."""
import numpy as np
import gym
import pybullet

from gym import wrappers

from rrc_iprl_package.control import controller_utils as c_utils
from rrc_iprl_package.envs import cube_env, custom_env
from rrc_iprl_package.envs.env_utils import configurable, flatten_space
from rrc_iprl_package.control.custom_pinocchio_utils import CustomPinocchioUtils
from rrc_iprl_package.envs.cube_env import CubeEnv, ActionType
from rrc_iprl_package.envs.initializers import *

from trifinger_simulation import TriFingerPlatform
from trifinger_simulation import visual_objects
from trifinger_simulation.tasks import move_cube
from trifinger_simulation import trifingerpro_limits
from scipy.spatial.transform import Rotation
from collections import deque


MAX_DIST = move_cube._max_cube_com_distance_to_center
DIST_THRESH = 0.02
_CUBOID_WIDTH = max(move_cube._CUBOID_SIZE)
_CUBOID_HEIGHT = min(move_cube._CUBOID_SIZE)

ORI_THRESH = np.pi / 8
REW_BONUS = 10
REW_PENALTY = -10
POS_SCALE = np.array([0.128, 0.134, 0.203, 0.128, 0.134, 0.203, 0.128, 0.134,
                      0.203])


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
    def __init__(self, env, goal_env=False, relative=False, scale=.3,
                 ac_pen=0.):
        super(TaskSpaceWrapper, self).__init__(env)
        self.action_log = []
        self._save_npz = self.unwrapped.save_npz

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
        pose = np.zeros(7)
        pose[2] = _CUBOID_HEIGHT / 2
        pose[-1] = 1
        pose = move_cube.Pose.from_dict(dict(position=pose[:3], orientation=pose[3:]))
        self._platform = TriFingerPlatform(
                visualization=False,
                initial_object_pose=pose,
            )
        self.pinocchio_utils = CustomPinocchioUtils(
                    self._platform.simfinger.finger_urdf_path,
                    self._platform.simfinger.tip_link_names)
        self.ac_pen = ac_pen
        if self.goal_env:
            obs_space = self.observation_space.spaces['observation']
            obs_dict = obs_space.spaces
        else:
            obs_dict = self.observation_space.spaces

        if 'action' not in obs_dict:
            obs_dict['action'] = self.action_space

    def reset(self, **kwargs):
        obs = super(TaskSpaceWrapper, self).reset(**kwargs)
        self._prev_obs = obs
        self._last_action = np.zeros_like(self.action_space.sample())
        obs_dict = obs
        if self.goal_env:
            obs_dict = obs['observation']
        obs_dict['action'] = self._last_action
        return obs

    def write_action_log(self, observation, action, reward):
        self.action_log.append(dict(
            observation=observation, rl_action=action,
            action=self.action(action), t=self.step_count,
            reward=reward))

    def step(self, action):
        o, r, d, i = super(TaskSpaceWrapper, self).step(action)
        self._prev_obs = o
        if self.relative:
            r -= self.ac_pen * np.linalg.norm(action)
        else:
            r -= self.ac_pen * np.linalg.norm(self._last_action - action)
        obs_dict = o
        if self.goal_env:
            obs_dict = obs_dict['observation']
        obs_dict['action'] = action
        self._last_action =  action
        return o, r, d, i

    def action(self, action, obs=None):
        obs = obs or self._prev_obs
        poskey, velkey = 'robot_position', 'robot_velocity'
        obj_dict, objpos_key = obs, 'object_position'
        if self.goal_env:
            obj_dict, objpos_key = obs['achieved_goal'], 'position'
            obs, poskey, velkey = obs['observation'], 'position', 'velocity'
        current_position, current_velocity = obs[poskey], obs[velkey]
        obj_position = obj_dict[objpos_key]
        if self.relative:
            action = action * 0.3  # scale all positions to be in -.3, .3 range
            # fingertip_goals = obj_position + action.reshape((3,3))
            # fingertip_goals = fingertip_goals.flatten()
            if 'robot_tip_positions' in obs or 'tip_positions' in obs:
                ftip_key = 'tip_positions' if self.goal_env else 'robot_tip_positions'
                fingertip_goals = obs[ftip_key].flatten()
            else:
                fingertip_goals = self.pinocchio_utils.forward_kinematics(
                        current_position.flatten())
            fingertip_goals = np.asarray(fingertip_goals)
            fingertip_goals = fingertip_goals + action
        else:
            fingertip_goals = action

        # currently not implemented
        if self.unwrapped.action_type == ActionType.TORQUE:
            raise NotImplementedError
            # compute desired velocity
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
            if obs and np.all(action == 0):
                ac = current_position
            else:
                ac, ft_err = self.pinocchio_utils.inverse_kinematics(
                        fingertip_goals.reshape((3,3)), current_position)
            ac = np.clip(ac.flatten(), self.unwrapped.action_space.low,
                         self.unwrapped.action_space.high)
        return ac


@configurable(pickleable=True)
class ScaledActionWrapper(gym.ActionWrapper):
    def __init__(self, env, goal_env=False, relative=True, scale=POS_SCALE,
                 lim_penalty=0.0):
        super(ScaledActionWrapper, self).__init__(env)
        self._save_npz = self.unwrapped.save_npz
        assert self.unwrapped.action_type == ActionType.POSITION, \
                'position control only'
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
        self.dt = self.frameskip * .001
        self.action_log = []

    def write_action_log(self, observation, action, reward):
        self.action_log.append(dict(
            observation=observation, action=action,
            scaled_action=self.action(action), t=self.step_count,
            reward=reward))

    def reset(self, **kwargs):
        obs = super(ScaledActionWrapper, self).reset(**kwargs)
        self._prev_obs = obs
        self._clipped_action = self._last_action = 0*self.action_space.sample()
        return obs

    def step(self, action):
        o, r, d, i = super(ScaledActionWrapper, self).step(action)
        self._prev_obs = o
        self._last_action =  action
        r += np.sum(self._clipped_action) * self.lim_penalty
        return o, r, d, i

    def action(self, action, obs=None):
        obs = obs or self._prev_obs
        poskey, velkey = 'robot_position', 'robot_velocity'
        if self.goal_env:
            obs, poskey, velkey = obs['observation'], 'position', 'velocity'
        current_position, current_velocity = obs[poskey], obs[velkey]
        if self.relative:
            goal_position = current_position + self.scale * action
            pos_low, pos_high = (self.env.action_space.low,
                                 self.env.action_space.high)
        else:
            pos_low, pos_high = (self.spaces.robot_position.low,
                                 self.spaces.robot_position.high)
            pos_low = np.max([current_position - self.scale, pos_low], axis=0)
            pos_high = np.min([current_position + self.scale, pos_high],
                              axis=0)
            goal_position = action
        action = np.clip(goal_position, pos_low, pos_high)
        assert action in self.env.action_space, f'action {action} not in action_space'
        self._clipped_action = np.abs(action - goal_position)
        return action


@configurable(pickleable=True)
class RelativeGoalWrapper(gym.ObservationWrapper):
    def __init__(self, env, keep_goal=True, use_quat=False):
        super(RelativeGoalWrapper, self).__init__(env)
        self._observation_keys = list(env.observation_space.spaces.keys())
        assert 'goal_object_position' in self._observation_keys, 'goal_object_position missing in observation'
        self.position_only = False # 'goal_orientation' not in self._observation_keys
        self.use_quat = use_quat
        self.observation_names =  [k for k in self._observation_keys if 'goal_object' not in k]
        self.observation_names.append('relative_goal_object_position')
        # add goal_object position if keep_goal, and orientation if not position_only 
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
                high = (env.observation_space['goal_object_position'].high
                        - env.observation_space['goal_object_position'].low)
                low = -high
                obs_dict[k] = gym.spaces.Box(low=low, high=high)
            elif k == 'relative_goal_object_orientation':
                if use_quat:
                    high = (env.observation_space['goal_object_orientation'].high
                            - env.observation_space['goal_object_orientation'].low)
                    low = -high
                    obs_dict[k] = gym.spaces.Box(low=low, high=high)
                else:
                    low, high = -np.pi, np.pi
                    obs_dict[k] = gym.spaces.Box(low=low, high=high, shape=(1,))
        self.observation_space = gym.spaces.Dict(obs_dict)

    def observation(self, obs):
        obs_dict = {k: obs[k] for k in self.observation_names if k in self._observation_keys}
        obs_dict['relative_goal_object_position'] = obs['goal_object_position'] - obs['object_position']
        if not self.position_only:
            goal_rot = Rotation.from_quat(obs['goal_object_orientation'])
            actual_rot = Rotation.from_quat(obs['object_orientation'])
            if self.use_quat:
                obs_dict['relative_goal_object_orientation'] = (
                        goal_rot*actual_rot.inv()).as_quat()
            else:
                obs_dict['relative_goal_object_orientation'] = get_theta_z_wf(
                        goal_rot, actual_rot)
        return obs_dict


@configurable(pickleable=True)
class ReorientWrapper(gym.Wrapper):
    def __init__(self, env, goal_env=True, rew_bonus=REW_BONUS,
                 rew_penalty=REW_PENALTY,
                 dist_thresh=0.09, ori_thresh=np.pi/6):
        super(ReorientWrapper, self).__init__(env)
        if not isinstance(self.unwrapped.initializer, ReorientInitializer):
            initializer = ReorientInitializer(initial_dist=np.min(MAX_DIST, 0.01 + dist_thresh))
            self.unwrapped.initializer = initializer
        self.goal_env = goal_env
        self.rew_bonus = rew_bonus
        self.rew_penalty = rew_penalty
        self.dist_thresh = dist_thresh
        self.ori_thresh = ori_thresh

    def step(self, action):
        o, r, d, i = super(ReorientWrapper, self).step(action)
        i['is_success'] = self.is_success(o)
        if not i['is_success'] and d:
            r += self.rew_penalty
        elif i['is_success']:
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


@configurable(pickleable=True)
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
        return np.concatenate([self.unwrapped.goal.orientation,
                               self.unwrapped.goal.position])

    @goal.setter
    def goal(self, g):
        if isinstance(g, dict):
            self.unwrapped.goal = move_cube.Pose(**g)
            return
        pos, ori = g[...,4:], g[...,:4]
        self.unwrapped.goal = move_cube.Pose(pos, ori)

    def compute_reward(self, achieved_goal, desired_goal, info):
        if len(achieved_goal.shape) > 1:
            r = []
            info = {"difficulty": self.initializer.difficulty}
            for i in range(achieved_goal.shape[0]):
                pos, ori = achieved_goal[i,4:], achieved_goal[i,:4]
                ag = dict(position=pos, orientation=ori)
                pos, ori = desired_goal[i,4:], desired_goal[i,:4]
                dg = dict(position=pos, orientation=ori)
                r.append(self.env.compute_reward(ag, dg, info))
            return np.array(r)
        achieved_goal = dict(position=achieved_goal[4:], orientation=achieved_goal[:4])
        desired_goal = dict(position=desired_goal[4:], orientation=desired_goal[:4])
        return self.env.compute_reward(achieved_goal, desired_goal, info)

    def _sample_goal(self):
        goal = self.initializer.get_goal().to_dict()
        return np.concatenate([goal['orientation'], goal['position']])

    def observation(self, observation):
        observation = {k: gym.spaces.flatten(self.env.observation_space[k], v)
                       for k, v in observation.items()}
        return observation


@configurable(pickleable=True)
class CubeRewardWrapper(gym.Wrapper):
    def __init__(self, env, target_dist=0.156, pos_coef=1., ori_coef=0.,
                 fingertip_coef=0., ac_norm_pen=0.2, goal_env=False, rew_fn='exp',
                 augment_reward=False, min_ftip_height=0.01, max_velocity=0.175):
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
        self._min_ftip_height = min_ftip_height
        self._max_velocity = max_velocity
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
        self.obj_pos.clear()
        self.obj_ori.clear()
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

        if self._goal_env:
            observation = observation['observation']

        if (observation['robot_velocity'] > self._max_velocity).any():
            curr_vel = observation['robot_velocity']
            reward += -np.sum(np.clip(curr_vel - self._max_velocity, 0, np.inf))*10
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
            prev_ftip_pos = np.asarray(self.platform.forward_kinematics(previous_observation['robot_position']))
            curr_ftip_pos = np.asarray(self.platform.forward_kinematics(observation['robot_position']))

        current_distance_from_block = np.linalg.norm(
           curr_ftip_pos - observation["object_position"]
        )
        previous_distance_from_block = np.linalg.norm(
            prev_ftip_pos
            - previous_observation["object_position"]
        )

        ftip_rew = (
            previous_distance_from_block - current_distance_from_block
        ) * self._fingertip_coef
        ftip_pen = -np.sum(np.array(curr_ftip_pos[::3]) < self._min_ftip_height)
        if ftip_pen != 0:
            ftip_rew = ftip_pen
        self.info['fingertip_rew'] = ftip_rew
        return ftip_rew

    def compute_reward(self, achieved_goal, desired_goal, info):
        if isinstance(achieved_goal, dict):
            obj_pos, obj_ori = achieved_goal['position'], achieved_goal['orientation']
            goal_pos, goal_ori = desired_goal['position'], desired_goal['orientation']
        else:
            obj_pos, obj_ori = achieved_goal[:3], achieved_goal[3:]
            goal_pos, goal_ori = desired_goal[:3], desired_goal[3:]
        self.obj_pos.append(obj_pos)
        self.obj_ori.append(obj_ori)
        # sliding window average
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
            if isinstance(observation['achieved_goal'], dict):
                pos = observation['achieved_goal']['position']
                ori = observation['achieved_goal']['orientation']
            else:
                pos, ori = observation['achieved_goal'][:3], observation['achieved_goal'][3:]
        object_pose = move_cube.Pose(position=pos,
                                     orientation=ori)
        return goal_pose, object_pose

    def compute_position_error(self, goal_pose, object_pose):
        pos_error = np.linalg.norm(object_pose.position - goal_pose.position)
        return pos_error


@configurable(pickleable=True)
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

    def compute_orientation_error(self, info, scale=False):
        goal_pose, object_pose = self.get_goal_object_pose()
        ori_error = compute_orientation_error(goal_pose, object_pose)
        if not scale:
            ori_error = ori_error * np.pi
        return ori_error

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


@configurable(pickleable=True)
class StepRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super(StepRewardWrapper, self).__init__(env)
        self._last_rew = 0.

    def reset(self, **kwargs):
        self._last_rew = 0.
        return super(StepRewardWrapper, self).reset(**kwargs)

    def reward(self, reward):
        step_reward = reward - self._last_rew
        self._last_rew = reward
        return step_reward


@configurable(pickleable=True)
class ObservationNoiseParams:
    def __init__(self, object_pos_std=.01, object_ori_std=0.,
                 robot_pos_std=0., robot_vel_std=0., action_noise_loc=-0.01,
                 action_noise_scale=0.005, object_mass=0.016,
                 object_friction=1., joint_friction_scale=0.01):
        self.object_pos_std = object_pos_std
        self.object_ori_std = object_ori_std
        self.robot_pos_std = robot_pos_std
        self.robot_vel_std = robot_vel_std
        self.action_noise_loc = action_noise_loc
        self.action_noise_scale = action_noise_scale
        self.object_mass = object_mass
        self.object_friction = object_friction
        self.joint_friction_scale = joint_friction_scale
        self.joint_friction = np.random.sample(9) * joint_friction_scale

    def randomize(self, **kwargs):
        for kwarg in kwargs:
            if kwarg in self.__dict__:
                self.__dict__[kwarg] = kwargs.get(kwarg)
        return


@configurable(pickleable=True)
class ObservationNoiseWrapper(gym.ObservationWrapper, gym.ActionWrapper):
    def __init__(self, env, noise_params=None, goal_env=False):
        super(ObservationNoiseWrapper, self).__init__(env)
        self.noise_params = noise_params or ObservationNoiseParams()
        self.goal_env = goal_env

    def randomize_params(self):
        self.noise_params.randomize()

    def reset(self, randomize=False, **kwargs):
        if randomize:
            self.randomize_params(**kwargs)
        ret = super(ObservationNoiseWrapper, self).reset(
                joint_friction=self.noise_params.joint_friction,
                object_mass=self.noise_params.object_mass)
        if self.noise_params.object_friction:
            lateral_friction = self.noise_params.object_friction
            spinning_friction = .001 * self.noise_params.object_friction
            pybullet.changeDynamics(
                    bodyUniqueId=self.platform.cube.block_id, 
                    linkIndex=-1,
                    lateralFriction=lateral_friction,
                    spinningFriction=spinning_friction)
        return ret

    def step(self, action):
        action = self.action(action)
        return super(ObservationNoiseWrapper, self).step(action)

    def action(self, action):
        if self.unwrapped.resetting:
            return action
        action_noise = np.random.normal(self.noise_params.action_noise_loc,
                                   scale=self.noise_params.action_noise_scale,
                                   size=action.shape)
        if self.action_type == ActionType.TORQUE_AND_POSITION:
            action['torque'] += action_noise
            action['torque'] = np.clip(action['torque'], self.action_space.low,
                                       self.action_space.high)
        else:
            action += action_noise
            action = np.clip(action, self.action_space.low, self.action_space.high)
        return action

    def observation(self, obs):
        if self.goal_env:
            object_position_key = 'position'
            object_orientation_key = 'orientation'
            object_dict = obs['achieved_goal']
        else:
            object_position_key = 'object_position'
            object_orientation_key = 'object_orientation'
            object_dict = obs
        object_dict[object_position_key] += np.random.normal(
                scale=self.noise_params.object_pos_std,
                size=3
            )
        rot = Rotation.from_quat(object_dict[object_orientation_key])
        xyz = rot.as_euler('xyz')
        xyz += np.random.normal(
                scale=self.noise_params.object_ori_std,
                size=3
            )
        xyz = xyz % 2*np.pi
        object_dict[object_orientation_key] = Rotation.from_euler('xyz', xyz).as_quat()
        if self.goal_env:
            robot_position_key = 'position'
            robot_velocity_key = 'velocity'
            robot_dict = obs['observation']
        else:
            robot_position_key = 'robot_position'
            robot_velocity_key = 'robot_velocity'
            robot_dict = obs

        robot_dict[robot_position_key] += np.random.normal(
                scale=self.noise_params.robot_pos_std,
                size=9
            )
        robot_dict[robot_velocity_key] += np.random.normal(
                scale=self.noise_params.robot_vel_std,
                size=9
            )
        return obs


@configurable(pickleable=True)
class SingleFingerWrapper(gym.ObservationWrapper):

    def __init__(self, env, finger_id=0, relative=False):
        super(SingleFingerWrapper, self).__init__(env)
        assert 0 <= finger_id < 3, f'finger_id was {finger_id}, must be in [0, 3)'
        self.finger_id = finger_id
        self.relative = relative
        if not relative:
            if self.action_type == ActionType.TORQUE_AND_POSITION:
                self.action_space = gym.spaces.Box(
                        low=-np.ones(3),
                        high=np.ones(3))
            else:
                self.action_space = gym.spaces.Box(
                    low=self.env.action_space.low[:3],
                    high=self.env.action_space.high[:3])
        else:
            self.action_space = gym.spaces.Box(
                low=-np.ones(3),
                high=np.ones(3))
            self.scale = POS_SCALE[:3]

        self.observation_names = self.env.observation_names
        obs_space_dict = self.observation_space.spaces

        self._initial_action = np.array([0.,.8,-2]*3)
        self._initial_action[finger_id*3:(finger_id+1)*3] = (
                np.array([0.,  0.75, -1.24]))
        if self.action_type == ActionType.TORQUE_AND_POSITION:
            self.unwrapped._initial_action = {
                    'position': self._initial_action,
                    'torque': np.zeros(9)
                    }
        else:
            self.unwrapped._initial_action = self._initial_action

        for obs_key, obs_space in obs_space_dict.items():
            if obs_key == 'robot_tip_forces':
                obs_space_dict[obs_key] = gym.spaces.Box(
                        low=0, high=1, shape=(1,))
            elif obs_key == 'action':
                obs_space_dict[obs_key] = self.action_space
            elif 'robot' in obs_key:
                obs_space_dict[obs_key] = gym.spaces.Box(
                        low=obs_space.low[:3], high=obs_space.high[:3])
        return

    def observation(self, obs):
        for key in self.observation_names:
            if key == 'robot_tip_forces':
                obs[key] = obs[key][self.finger_id:self.finger_id+1]
            elif key == 'action':
                obs[key] = obs[key][self.finger_id*3:(self.finger_id+1)*3]
            elif 'robot' in key:
                obs[key] = obs[key][self.finger_id*3:(self.finger_id+1)*3]
        return obs

    def reset(self, **kwargs):
        self._prev_obs = None
        return super(SingleFingerWrapper, self).reset(**kwargs)

    def step(self, action):
        obs, r, d, i = super(SingleFingerWrapper, self).step(self.action(action))
        self._prev_obs = obs
        return obs, r, d, i

    def action(self, action):
        if self.relative and self.action_type == ActionType.POSITION:
            if self._prev_obs is None:
                current_pos = self._initial_action[:]
            else:
                if isinstance(self.unwrapped, gym.GoalEnv):
                    current_pos = self._prev_obs['observation']['position']
                else:
                    current_pos = self._prev_obs['robot_position']
            action = current_pos[self.finger_id*3:(self.finger_id+1)*3] + self.scale * action
        elif self.action_type == ActionType.TORQUE_AND_POSITION:
            action = action * 0.397
        t_action = self._initial_action[:]
        if self.action_type == ActionType.TORQUE_AND_POSITION:
            torque = np.zeros(9)
            torque[self.finger_id*3:(self.finger_id+1)*3] = action
            t_action = {'position': t_action, 'torque': torque}
        else:
            t_action[self.finger_id*3:(self.finger_id+1)*3] = action
            t_action = np.clip(t_action, self.env.action_space.low,
                               self.env.action_space.high)
        return t_action


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


