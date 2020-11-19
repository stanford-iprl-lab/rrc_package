"""Gym environment for the Real Robot Challenge Phase 1 (Simulation)."""
import csv
import enum
import gym
import numpy as np
import sys
import time

try:
    import robot_interfaces
    import robot_fingers
    from robot_interfaces.py_trifinger_types import Action 
except ImportError:
    robot_interfaces = robot_fingers = None
    from trifinger_simulation.action import Action

import trifinger_simulation
import trifinger_simulation.visual_objects
import rrc_iprl_package.pybullet_utils as pbutils
from trifinger_simulation import trifingerpro_limits
from trifinger_simulation.tasks import move_cube


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

    def __init__(
        self,
        cube_goal_pose: dict,
        cube_initial_pose: dict = None,
        goal_difficulty: int = 1,
        action_type: ActionType = ActionType.POSITION,
        visualization: bool = True,
        frameskip: int = 1,
        num_steps: int = None,
        save_npz: str = None,
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

        self.goal = cube_goal_pose
        if not isinstance(cube_goal_pose, dict):
            self.goal = cube_goal_pose.as_dict()
        self.info = {"difficulty": goal_difficulty}
        self.initial_pose = move_cube.Pose.from_dict(cube_initial_pose) if cube_initial_pose else move_cube.sample_goal(-1)

        self.action_type = action_type

        # CSV logging file path
        self.csv_filepath = "/output/virtual_reset.csv"

        with open(self.csv_filepath, mode="a") as file:
            writer  = csv.writer(file, delimiter=",")
            writer.writerow(["timestamp", "reward"])

        # TODO: The name "frameskip" makes sense for an atari environment but
        # not really for our scenario.  The name is also misleading as
        # "frameskip = 1" suggests that one frame is skipped while it actually
        # means "do one step per step" (i.e. no skip).
        if frameskip < 1:
            raise ValueError("frameskip cannot be less than 1.")
        self.frameskip = frameskip
        self.episode_length = num_steps * frameskip if num_steps else move_cube.episode_length
        self.max_resets = (60 * 1000 - 150) // self.episode_length     # number of virtual resets

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
        if not object_state_space.contains(self.goal):
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

        self.observation_space = gym.spaces.Dict(
            {
                "observation": gym.spaces.Dict(
                    {
                        "position": robot_position_space,
                        "velocity": robot_velocity_space,
                        "torque": robot_torque_space,
                    }
                ),
                "action": self.action_space,
                "desired_goal": object_state_space,
                "achieved_goal": object_state_space,
            }
        )
        self.save_npz = save_npz
        self.action_log = []

    def compute_reward(self, achieved_goal, desired_goal, info):
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

            reward += self.compute_reward(
                observation["achieved_goal"],
                observation["desired_goal"],
                self.info,
            )

            self.step_count = t
            # make sure to not exceed the episode length
            if self.step_count >= self.episode_length - 1:
                break

        is_done = self.step_count == self.episode_length
        self.write_action_log(observation, action, reward)

        self._last_obs = observation
        self._last_reward = reward

        return observation, reward, is_done, self.info

    def write_action_log(self, observation, action, reward):
        if self.save_npz:
            self.action_log.append(dict(
                observation=observation, action=action, t=self.step_count,
                reward=reward))

    def save_action_log(self):
        if self.save_npz and self.action_log:
            self.action_log.append(dict(initial_pose=self.initial_pose.to_dict(),
                                   goal_pose=self.goal))

            np.savez(self.save_npz, action_log=self.action_log, allow_pickle=True)
            del self.action_log
        self.action_log = []

    def reset(self):
        # By changing the `_reset_*` method below you can switch between using
        # the platform frontend, which is needed for the submission system, and
        # the direct simulation, which may be more convenient if you want to
        # pre-train locally in simulation.
        print("Resetting")
        if self.save_npz and self.action_log:
            self.save_action_log()

        if robot_fingers is not None:
            self._reset_platform_frontend()
        else:
            self._reset_direct_simulation()

        self.step_count = 0

        self.init_time = 0.0            # the time when episodes started to run
        self.reset_time = 0.0           # the time when reset finishes

        # need to already do one step to get initial observation
        # TODO disable frameskip here?
        if self.num_reset == 0:     # if this is the first (real) reset
            observation, reward, _, _ = self.step(self._initial_action)
            self._last_obs = observation
            self._last_reward = reward
            self.init_time = time.time()
            csv_row = "{}, {}".format(self.init_time, self._last_reward)
        elif self.num_reset == self.max_resets:     # if all virtual resets are completed
            return self._last_obs
        else:
            print("Further resetting")
            observation, reward, _, _ = self.step(self._initial_action)  # try resetting fingers so we can check velocity
            print("Velocity 0: ", observation["observation"]["velocity"])
            # virtual reset is done only when all joints velocity are zero
            while any(vel < 0.01 for vel in observation["observation"]["velocity"]) is False:
                print("Keep resetting, velocity: ", observation["observation"]["velocity"])
                observation, reward, _, _ = self.step(self._initial_action)
            print("Resetting finished")
            self.reset_time = time.time() - self.init_time
            self._last_obs = observation   
            self._last_reward = reward
            csv_row = "{}, {}".format(self.reset_time, self._last_reward)

        with open(self.csv_filepath, mode="a") as file:
            writer  = csv.writer(file, delimiter=",")
            writer.writerow(csv_row)      

        return observation

    def _reset_platform_frontend(self):
        """Reset the platform frontend."""
        # reset is not really possible
        import pdb; pdb.set_trace()
        print("Hardware Resetting")
        if self.platform is None:
            print("Hardware Resetting Init")
            self.platform = robot_fingers.TriFingerPlatformFrontend()
            self.num_reset = 0  
        else:
            print("Hardware Resetting Further")
            self.num_reset += 1        

    def _reset_direct_simulation(self):
        """Reset direct simulation.

        With this the env can be used without backend.
        """
        print("Simulation Resetting")
        # initialize number of resets here too so overall reset() function works
        self.num_reset = 0
        
        # reset simulation
        del self.platform

        # initialize simulation
        self.platform = trifinger_simulation.TriFingerPlatform(
            visualization=self.visualization,
            initial_object_pose=self.initial_pose,
        )

        # visualize the goal
        if self.visualization:
            self.goal_marker = trifinger_simulation.visual_objects.CubeMarker(
                width=0.065,
                position=self.goal["position"],
                orientation=self.goal["orientation"],
                physicsClientId=self.platform.simfinger._pybullet_client_id,
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

    def _create_observation(self, t, action):
        robot_observation = self.platform.get_robot_observation(t)
        camera_observation = self.platform.get_camera_observation(t)

        observation = {
            "observation": {
                "position": robot_observation.position,
                "velocity": robot_observation.velocity,
                "torque": robot_observation.torque,
            },
            "action": action,
            "desired_goal": self.goal,
            "achieved_goal": {
                "position": camera_observation.object_pose.position,
                "orientation": camera_observation.object_pose.orientation,
            },
        }
        return observation

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


class CubeEnv(RealRobotCubeEnv):
    def __init__(
        self,
        initializer,
        goal_difficulty: int,
        action_type: ActionType = ActionType.POSITION,
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
            action_type, visualization, frameskip, num_steps, save_npz)

    def reset(self): 
        self.initial_pose = self.initializer.get_initial_state()
        self.goal = self.initializer.get_goal().to_dict()
        return super(CubeEnv, self).reset()


