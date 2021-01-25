"""Gym environment for the Real Robot Challenge Phase 1 (Simulation)."""
import enum
import gym
import numpy as np
import os.path as osp

try:
    import robot_interfaces
    import robot_fingers
    import trifinger_cameras
    from robot_interfaces.py_trifinger_types import Action 
except ImportError:
    robot_interfaces = robot_fingers = None
    from trifinger_simulation.action import Action

import trifinger_simulation
import trifinger_simulation.visual_objects
import rrc_iprl_package.pybullet_utils as pbutils
from scipy.spatial.transform import Rotation
from trifinger_simulation import trifingerpro_limits
from trifinger_simulation.tasks import move_cube
from dm_control.utils import rewards as dmr


DIST_THRESH = 0.05
_CUBOID_WIDTH = max(move_cube._CUBOID_SIZE)
_CUBOID_HEIGHT = min(move_cube._CUBOID_SIZE)

ORI_THRESH = np.pi / 8
REW_BONUS = 10
REW_PENALTY = -10
POS_SCALE = np.array([0.128, 0.134, 0.203, 0.128, 0.134, 0.203, 0.128, 0.134,
                      0.203])


class ActionType(enum.Enum):
    """Different action types that can be used to control the robot."""

    #: Use pure torque commands.  The action is a list of torques (one per
    #: joint) in this case.
    TORQUE = enum.auto()
    #: Use joint position commands.  The action is a list of angular joint
    #: positions (one per joint) in this case.  Internally a PD controller is
    #: executed for each action to determine the torques that are applied to
    #: the robot.
    POSITION = enum.auto()
    #: Use both torque and position commands.  In this case the action is a
    #: dictionary with keys "torque" and "position" which contain the
    #: corresponding lists of values (see above).  The torques resulting from
    #: the position controller are added to the torques in the action before
    #: applying them to the robot.
    TORQUE_AND_POSITION = enum.auto()


class RealRobotCubeEnv(gym.GoalEnv):
    """Gym environment for moving cubes with simulated TriFingerPro."""
    observation_names = ["position",
            "velocity",
            "torque",
            "tip_positions",
            "action",
            "cam0_timestamp"]

    def __init__(
        self,
        cube_goal_pose: dict,
        cube_initial_pose: dict = None,
        goal_difficulty: int = 1,
        action_type: ActionType = ActionType.POSITION,
        default_position: np.ndarray = np.array([0.0, 0.75, -1.6] * 3),
        visualization: bool = True,
        frameskip: int = 1,
        num_steps: int = None,
        save_npz: str = None,
        alpha: float = 0.01
    ):
        """Initialize.

        Args:
            cube_goal_pose (dict): Goal pose for the cube.  Dictionary with
                keys "position" and "orientation".
            goal_difficulty (int): Difficulty level of the goal (needed for
                reward computation).
            action_type (ActionType): Specify which type of actions to use.
                See :class:`ActionType` for details.
            frameskip (int):  Number of actual control steps to be performed in
                one call of step().
        """
        # Basic initialization
        # ====================

        self.goal = move_cube.Pose.from_dict(cube_goal_pose)
        self.info = {"difficulty": goal_difficulty}
        if cube_initial_pose:
            self.initial_pose = move_cube.Pose.from_dict(cube_initial_pose)
        else:
            self.initial_pose = move_cube.sample_goal(-1)

        self.action_type = action_type

        # TODO: The name "frameskip" makes sense for an atari environment but
        # not really for our scenario.  The name is also misleading as
        # "frameskip = 1" suggests that one frame is skipped while it actually
        # means "do one step per step" (i.e. no skip).
        if frameskip < 1:
            raise ValueError("frameskip cannot be less than 1.")
        self.frameskip = frameskip
        self.episode_length = num_steps * frameskip if num_steps else move_cube.episode_length
        self.num_resets = 0

        # will be initialized in reset()
        self.platform = None
        self.visualization = visualization

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
        self.save_npz = save_npz

        # verify that the given goal pose is contained in the cube state space
        if not object_state_space.contains(self.goal.to_dict()):
            raise ValueError("Invalid goal pose.")

        self.default_position = default_position
        if self.action_type == ActionType.TORQUE:
            # assert self.action_type in [ActionType.POSITION,
            #         ActionType.TORQUE_AND_POSITION]
            self.action_space = robot_torque_space
            self._initial_action = trifingerpro_limits.robot_torque.default
        elif self.action_type == ActionType.POSITION:
            self.action_space = robot_position_space
            self._initial_action = default_position
        elif self.action_type == ActionType.TORQUE_AND_POSITION:
            self.action_space = gym.spaces.Dict(
                {
                    "torque": robot_torque_space,
                    "position": robot_position_space,
                }
            )
            self._initial_action = {
                "torque": trifingerpro_limits.robot_torque.default,
                "position": default_position,
            }
        else:
            raise ValueError("Invalid action_type")    

        observation_state_space = {
                "position": robot_position_space,
                "velocity": robot_velocity_space,
                "torque": robot_torque_space,
                "tip_positions": gym.spaces.Box(
                    low=np.concatenate([trifingerpro_limits.object_position.low]*3),
                    high=np.concatenate([trifingerpro_limits.object_position.high]*3)),
                "action": self.action_space,
                "cam0_timestamp": gym.spaces.Box(low=0., high=np.inf, shape=())
            }
        observation_state_space = gym.spaces.Dict({
                k: observation_state_space[k] 
                    for k in self.observation_names
            })
        self.observation_space = gym.spaces.Dict(
            {
                "observation": observation_state_space,
                "desired_goal": object_state_space,
                "achieved_goal": object_state_space,
            }
        )
        if osp.exists('/output'):
            self.observation_space.spaces['filtered_achieved_goal'] = object_state_space
        self.save_npz = save_npz
        self.action_log = []
        self.filtered_position = None
        self.filtered_orientation = None
        self.alpha = alpha

    def compute_reward_old(self, achieved_goal, desired_goal, info):
        """Compute the reward for the given achieved and desired goal.

        Args:
            achieved_goal (dict): Current pose of the object.
            desired_goal (dict): Goal pose of the object.
            info (dict): An info dictionary containing a field "difficulty"
                which specifies the difficulty level.

        Returns:
            float: The reward that corresponds to the provided achieved goal
            w.r.t. to the desired goal. Note that the following should always
            hold true::

                ob, reward, done, info = env.step()
                assert reward == env.compute_reward(
                    ob['achieved_goal'],
                    ob['desired_goal'],
                    info,
                )
        """
        return -move_cube.evaluate_state(
            move_cube.Pose.from_dict(desired_goal),
            move_cube.Pose.from_dict(achieved_goal),
            info["difficulty"],
        )

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

    def compute_fingertip_error(self, ftip_pos, object_pose):
        ftip_err = np.linalg.norm(ftip_pos.reshape((3,3))
                                  - object_pose.position, axis=1)
        return ftip_err

    def compute_reward(self, achieved_goal, desired_goal, info):
        object_pose = move_cube.Pose.from_dict(achieved_goal)
        goal_pose = move_cube.Pose.from_dict(desired_goal)
        position_error = self.compute_position_error(goal_pose, object_pose)
        orientation_error = self.compute_orientation_error(goal_pose, object_pose)
        corner_error = self.compute_corner_error(goal_pose, object_pose).sum()
        ftip_error = self.compute_fingertip_error(info.get('tip_positions'),
                                                  object_pose).sum()
        reward = dmr.tolerance(position_error, (0., DIST_THRESH/2),
                               margin=DIST_THRESH/2, sigmoid='long_tail')
        reward += dmr.tolerance(orientation_error, (0., ORI_THRESH/2),
                                margin=ORI_THRESH/2, sigmoid='long_tail')
        reward += dmr.tolerance(corner_error, (0., DIST_THRESH*3),
                                margin=DIST_THRESH*3, sigmoid='long_tail')
        reward += dmr.tolerance(ftip_error, (0., _CUBOID_HEIGHT*3/2),
                                margin=_CUBOID_HEIGHT*3/2, sigmoid='long_tail')
        info['pos_error'] = position_error
        info['ori_error'] = orientation_error
        info['corner_error'] = corner_error
        return reward

    def step(self, action):
        """Run one timestep of the environment's dynamics.

        When end of episode is reached, you are responsible for calling
        ``reset()`` to reset this environment's state.

        Args:
            action: An action provided by the agent (depends on the selected
                :class:`ActionType`).

        Returns:
            tuple:

            - observation (dict): agent's observation of the current
              environment.
            - reward (float) : amount of reward returned after previous action.
            - done (bool): whether the episode has ended, in which case further
              step() calls will return undefined results.
            - info (dict): info dictionary containing the difficulty level of
              the goal.
        """
        if self.platform is None:
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
            t = self.platform.append_desired_action(robot_action)

            observation = self._create_observation(t, action)

            self.step_count += t - self.t_prev
            self.t_prev = t
            # make sure to not exceed the episode length
            if self.step_count >= self.episode_length or self.t_prev == 120*1000 - 1:
                break

        self.observation = observation['observation']
        reward += self.compute_reward(
            observation["achieved_goal"],
            observation["desired_goal"],
            self.info,
        )

        is_done = self.step_count >= self.episode_length
        # self.write_action_log(observation, action, reward)
        self.info['num_steps'] = self.step_count

        return observation, reward, is_done, self.info

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

    def reset(self, **platform_kwargs):
        # By changing the `_reset_*` method below you can switch between using
        # the platform frontend, which is needed for the submission system, and
        # the direct simulation, which may be more convenient if you want to
        # pre-train locally in simulation.
        self.resetting = True
        if robot_fingers is not None:
            self._reset_platform_frontend(**platform_kwargs)
        else:
            self._reset_direct_simulation(**platform_kwargs)

        self.step_count = 0
        if self.num_resets * self.episode_length >= 120*1000:
            print('Not performing full reset, reached maximum number of resets')
            return self.action_log[-1]['observation']

        # need to already do one step to get initial observation
        if self.frameskip != 1:
            temp_frameskip = self.frameskip
            self.frameskip = 1
        else:
            temp_frameskip = None

        observation, reward, _, _ = self.step(self._initial_action)

        # set the initial pose to what is returned from first call to 
        # create_observation
        if robot_fingers is not None:
            self.initial_pose = move_cube.Pose.from_dict(observation['achieved_goal'])

        cur_vel = observation["observation"]["velocity"]
        cur_pos = observation["observation"]["position"]
        if self.num_resets != -1:
            # import pdb; pdb.set_trace()
            while not all(np.abs(vel) < 0.01 for vel in cur_vel) or \
                    np.abs(cur_pos - self.default_position).max() >= .05:
                observation, reward, _, _ = self.step(self._initial_action)
                cur_vel = observation["observation"]["velocity"]
                cur_pos = observation["observation"]["position"]

        if temp_frameskip is not None:
            self.frameskip = temp_frameskip

        self.filtered_position = self.filtered_orientation = None
        self.resetting = False
        return observation

    def _reset_platform_frontend(self, **platform_kwargs):
        """Reset the platform frontend."""
        # reset is not really possible
        if self.platform is not None:
            self.num_resets += 1
        else:
            self.platform = robot_fingers.TriFingerPlatformFrontend()
            platform = trifinger_simulation.TriFingerPlatform(
                visualization=False,
                initial_object_pose=self.initial_pose,
            )
            self.kinematics = platform.simfinger.kinematics
            self.t_prev = 0

    def _reset_direct_simulation(self, **platform_kwargs):
        """Reset direct simulation.

        With this the env can be used without backend.
        """
        # reset simulation
        del self.platform

        # initialize simulation
        self.platform = trifinger_simulation.TriFingerPlatform(
            visualization=self.visualization,
            initial_object_pose=self.initial_pose,
            **platform_kwargs,
        )
        self.kinematics = self.platform.simfinger.kinematics
        self.t_prev = 0

        # visualize the goal
        if self.visualization:
            self.goal_marker = trifinger_simulation.visual_objects.CuboidMarker(
                size=move_cube._CUBOID_SIZE,
                position=self.goal.position,
                orientation=self.goal.orientation,
                pybullet_client_id=self.platform.simfinger._pybullet_client_id,
            )
            pbutils.reset_camera()

    def seed(self, seed=None):
        """Sets the seed for this env’s random number generator.

        .. note::

           Spaces need to be seeded separately.  E.g. if you want to sample
           actions directly from the action space using
           ``env.action_space.sample()`` you can set a seed there using
           ``env.action_space.seed()``.

        Returns:
            List of seeds used by this environment.  This environment only uses
            a single seed, so the list contains only one element.
        """
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        move_cube.random = self.np_random
        return [seed]

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

        ftip_pos = np.asarray(self.kinematics.forward_kinematics(
            robot_observation.position))
        obs_dict = {
                "position": robot_observation.position,
                "velocity": robot_observation.velocity,
                "torque": robot_observation.torque,
                "tip_positions": ftip_pos,
                "action": action,
                "cam0_timestamp": camera_observation.cameras[0].timestamp
            }
        obs_dict = {k: obs_dict[k] for k in self.observation_names}
        self.info.update(obs_dict)
        observation = {
            "observation": obs_dict,
            "desired_goal": self.goal.to_dict(),
            "achieved_goal": {
                "position": camera_observation.object_pose.position,
                "orientation": camera_observation.object_pose.orientation,
            },
        }

        # if osp.exists("/output"):
        #     obj_pose = self.get_camera_pose(camera_observation)
        #     observation["filtered_achieved_goal"] = {
        #         "position": obj_pose.position,
        #         "orientation": obj_pose.orientation}
        return observation

    def _gym_action_to_robot_action(self, gym_action):
        # construct robot action depending on action type
        if self.action_type == ActionType.TORQUE:
            if self.resetting:
                robot_action = Action(torque=gym_action, position=self.default_position)
            else:
                robot_action = Action(torque=gym_action, position=np.array([np.nan]*9))
        elif self.action_type == ActionType.POSITION:
            robot_action = Action(position=gym_action, torque=np.zeros(9))
        elif self.action_type == ActionType.TORQUE_AND_POSITION:
            robot_action = Action(
                torque=gym_action["torque"], position=gym_action["position"]
            )
        else:
            raise ValueError("Invalid action_type")

        return robot_action


class CubeEnv(RealRobotCubeEnv):
    def __init__(
        self,
        initializer,
        goal_difficulty: int = 1,
        action_type: ActionType = ActionType.POSITION,
        default_position: np.ndarray = np.array([0.0, 0.75, -1.6] * 3),
        visualization: bool = False,
        frameskip: int = 1,
        num_steps: int = None,
        save_npz: str = None
    ):
        """Initialize.

        Args:
            cube_goal_pose (dict): Goal pose for the cube.  Dictionary with
                keys "position" and "orientation".
            goal_difficulty (int): Difficulty level of the goal (needed for
                reward computation).
            action_type (ActionType): Specify which type of actions to use.
                See :class:`ActionType` for details.
            frameskip (int):  Number of actual control steps to be performed in
                one call of step().
        """
        self.initializer = initializer
        initial_pose = self.initializer.get_initial_state().to_dict()
        goal_pose = self.initializer.get_goal().to_dict()
        super().__init__(goal_pose, initial_pose, goal_difficulty,
            action_type=action_type, default_position=default_position, visualization=visualization,
            frameskip=frameskip, num_steps=num_steps, save_npz=save_npz)

    def reset(self): 
        self.initial_pose = self.initializer.get_initial_state()
        self.goal = self.initializer.get_goal()
        return super(CubeEnv, self).reset()


