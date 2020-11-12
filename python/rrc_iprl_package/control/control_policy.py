"""
Implements ImpedanceControllerPolicy class which returns actions to be compatible
with Gym environment
"""

import os
import os.path as osp
import numpy as np
import joblib
import copy
import time
from scipy.interpolate import interp1d
import csv

from datetime import date
from trifinger_simulation import TriFingerPlatform
from trifinger_simulation.tasks import move_cube
from trifinger_simulation import pinocchio_utils
from trifinger_simulation import visual_objects

from rrc_iprl_package.control.custom_pinocchio_utils import CustomPinocchioUtils
from rrc_iprl_package.control import controller_utils as c_utils
from rrc_iprl_package.control.controller_utils import PolicyMode

try:
    import torch
except ImportError:
    torch = None


# Parameters for tuning gains
KP = [200, 200, 200,
      200, 200, 200,
      200, 200, 200]
KV = [0.5, 0.5, 0.5, 
      0.5, 0.5, 0.5,
      0.5, 0.5, 0.5]

# Sine wave parameters
SINE_WAVE_DIM = 1
class ImpedanceControllerPolicy:
    def __init__(self, action_space=None, initial_pose=None, goal_pose=None,
                 npz_file=None, debug_waypoints=False):
        self.action_space = action_space
        self.flipping = False
        self.debug_waypoints = debug_waypoints
        self.debug_fingertip_tracking = True
        self.set_init_goal(initial_pose, goal_pose)
        self.finger_waypoints = None # visual object (for debugging)
        self.done_with_primitive = True
        self.init_face = None
        self.goal_face = None
        self.platform = None
        print("KP: {}".format(KP))
        print("KV: {}".format(KV))
        self.start_time = None
        self.WAIT_TIME = 2

        # Counters
        self.step_count = 0 # Number of times predict() is called
        self.traj_waypoint_counter = 0

        self.grasped = False
        self.traj_to_object_computed = False

        # CSV logging file path
        self.csv_filepath = "/output/control_policy_data.csv"


    def reset_policy(self, platform=None):
        self.step_count = 0
        if platform:
            self.platform = platform

        self.custom_pinocchio_utils = CustomPinocchioUtils(
                self.platform.simfinger.finger_urdf_path,
                self.platform.simfinger.tip_link_names)

        init_position = np.array([0.0, 0.9, -1.7, 0.0, 0.9, -1.7, 0.0, 0.9, -1.7])
        self.init_ft_pos = self.get_fingertip_pos_wf(init_position)
        self.init_ft_pos = np.asarray(self.init_ft_pos).flatten()

        self.ft_pos_traj = np.tile(self.init_ft_pos, (10000,1))
        self.ft_vel_traj = np.zeros((10000,9))
        self.l_wf_traj = None

        csv_header = ["step", "timestamp"]
        # Formulate row to print csv_row = "{},".format(self.step_count)
        for i in range(9):
            csv_header.append("desired_ft_pos_{}".format(i))
        for i in range(9):
            csv_header.append("desired_ft_vel_{}".format(i))
        with open(self.csv_filepath, mode="a") as fid:
            writer  = csv.writer(fid, delimiter=",")
            writer.writerow(csv_header)

        # Define nlp for finger traj opt
        nGrid = 40
        dt = 0.04
        self.finger_nlp = c_utils.define_static_object_opt(nGrid, dt)


    def set_init_goal(self, initial_pose, goal_pose, flip=False):
        self.done_with_primitive = False
        self.goal_pose = goal_pose
        self.x0 = np.concatenate([initial_pose.position, initial_pose.orientation])[None]
        if not flip:
            self.flipping = False
        else:
            self.flipping = True
        init_goal_dist = np.linalg.norm(goal_pose.position - initial_pose.position)
        #print(f'init position: {initial_pose.position}, goal position: {goal_pose.position}, '
        #      f'dist: {init_goal_dist}')
        #print(f'init orientation: {initial_pose.orientation}, goal orientation: {goal_pose.orientation}')

    """
    Get contact point parameters for either lifting of flipping
    """
    def set_cp_params(self, observation):
        # Get object pose
        obj_pose = get_pose_from_observation(observation)

        if self.flipping:
            self.cp_params, self.init_face, self.goal_face = c_utils.get_flipping_cp_params(
                obj_pose, self.goal_pose)
        else:
            self.cp_params = c_utils.get_lifting_cp_params(obj_pose)

    """
    Run trajectory optimization to move object given fixed contact points
    """
    def set_traj_lift_object(self, observation, nGrid = 50, dt = 0.01):
        self.traj_waypoint_counter = 0

        # Get object pose
        obj_pose = get_pose_from_observation(observation)
            
        x0 = np.concatenate([obj_pose.position, obj_pose.orientation])[None]
        x_goal = x0.copy()
        x_goal[0, :3] = self.goal_pose.position

        print(x0)
        print(x_goal)
        # Get initial fingertip positions in world frame
        current_position, _ = get_robot_position_velocity(observation)
        
        self.x_soln, self.dx_soln, l_wf = c_utils.run_fixed_cp_traj_opt(
                obj_pose, self.cp_params, current_position, self.custom_pinocchio_utils,
                x0, x_goal, nGrid, dt)

        ft_pos = np.zeros((nGrid, 9))
        ft_vel = np.zeros((nGrid, 9))
        for t_i in range(nGrid):
            # Set fingertip goal positions and velocities from x_soln, dx_soln
            next_cube_pos_wf = self.x_soln[t_i, 0:3]
            next_cube_quat_wf = self.x_soln[t_i, 3:]

            ft_pos_list = c_utils.get_cp_pos_wf_from_cp_params(
                    self.cp_params, next_cube_pos_wf, next_cube_quat_wf)

            ft_pos[t_i, :] = np.asarray(ft_pos_list).flatten()
            ft_vel[t_i, :] = np.tile(self.dx_soln[t_i, 0:3],3)

        # Number of interpolation points
        interp_n = 4

        # Linearly interpolate between each position waypoint (row) and force waypoint
        # Initial row indices
        row_ind_in = np.arange(nGrid)
        # Output row coordinates
        row_coord_out = np.linspace(0, nGrid - 1, interp_n * (nGrid-1) + nGrid)
        # scipy.interpolate.interp1d instance
        itp_pos = interp1d(row_ind_in, ft_pos, axis=0)
        #itp_vel = interp1d(row_ind_in, ft_vel, axis=0)
        itp_lwf = interp1d(row_ind_in, l_wf, axis=0)
        self.ft_pos_traj = itp_pos(row_coord_out)
        #self.ft_vel_traj = itp_vel(row_coord_out)
        self.l_wf_traj = itp_lwf(row_coord_out)

        # Zero-order hold for velocity waypoints
        self.ft_vel_traj = np.repeat(ft_vel, repeats=interp_n+1, axis=0)[:-interp_n, :]

    """
    Run trajectory optimization to move fingers to contact points on object
    """
    def set_traj_to_object(self, observation):
        self.traj_waypoint_counter = 0
        # First, set cp_params based on mode
        self.set_cp_params(observation)

        # Get object pose
        obj_pose = get_pose_from_observation(observation)

        # Get list of desired fingertip positions
        cp_wf_list = c_utils.get_cp_pos_wf_from_cp_params(self.cp_params, obj_pose.position, obj_pose.orientation)
        ft_goal = np.asarray(cp_wf_list).flatten()
        self.run_finger_traj_opt(observation, ft_goal)

    """
    Run trajectory optimization for fingers, given fingertip goal positions
    ft_goal: (9,) array of fingertip x,y,z goal positions in world frame
    """
    def run_finger_traj_opt(self, observation, ft_goal):
        nGrid = self.finger_nlp.nGrid
        self.traj_waypoint_counter = 0
        # Get object pose
        obj_pose = get_pose_from_observation(observation)
        # Get initial fingertip positions in world frame
        current_position, _ = get_robot_position_velocity(observation)

        # Where the fingers start on the real robot (once they retract)
        #current_position = np.array([0.0, 0.9, -1.7, 0.0, 0.9, -1.7, 0.0, 0.9, -1.7])
        #self.init_ft_pos = self.get_fingertip_pos_wf(current_position)
        #self.init_ft_pos = np.asarray(current_ft_pos).flatten()
        #self.ft_tracking_init_pos_list = []
        #self.ft_tracking_init_pos_list.append(np.array([0.08, 0.07, 0.07]))
        #self.ft_tracking_init_pos_list.append(np.array([0.01, -0.1, 0.07]))
        #self.ft_tracking_init_pos_list.append(np.array([-0.1, 0.04, 0.07]))

        ft_pos, ft_vel = c_utils.get_finger_waypoints(self.finger_nlp, ft_goal, current_position, obj_pose)

        print("FT_GOAL: {}".format(ft_goal))
        print(ft_pos[-1,:])
    
        # Number of interpolation points
        interp_n = 13

        # Linearly interpolate between each waypoint (row)
        # Initial row indices
        row_ind_in = np.arange(nGrid)
        # Output row coordinates
        row_coord_out = np.linspace(0, nGrid - 1, interp_n * (nGrid-1) + nGrid)
        # scipy.interpolate.interp1d instance
        itp_pos = interp1d(row_ind_in, ft_pos, axis=0)
        #itp_vel = interp1d(row_ind_in, ft_vel, axis=0)
        self.ft_pos_traj = itp_pos(row_coord_out)
        #self.ft_vel_traj = itp_vel(row_coord_out)
        self.l_wf_traj = None
        # Zero-order hold for velocity waypoints
        self.ft_vel_traj = np.repeat(ft_vel, repeats=interp_n+1, axis=0)[:-interp_n, :]

    def predict(self, full_observation):
        self.step_count += 1
        observation = full_observation['observation']
        current_position, current_velocity = observation['position'], observation['velocity']

        if self.start_time is None:
            self.start_time = time.time()
            t = 0
        else:
            t = time.time() - self.start_time

        if not self.traj_to_object_computed and t > self.WAIT_TIME:
            self.set_traj_to_object(full_observation)
            self.traj_to_object_computed = True

        # HANDLE ANY TRAJECTORY RECOMPUTATION HERE
        if self.traj_waypoint_counter >= self.ft_pos_traj.shape[0] and not self.grasped:
            # If at end of trajectory, do fixed cp traj opt
            self.set_traj_lift_object(full_observation, nGrid = 50, dt = 0.08)
            self.grasped = True

        if self.traj_waypoint_counter >= self.ft_pos_traj.shape[0]:
            traj_waypoint_i = self.ft_pos_traj.shape[0] - 1
        else:
            traj_waypoint_i = self.traj_waypoint_counter

        fingertip_pos_goal_list = []
        fingertip_vel_goal_list = []
        for f_i in range(3):
            new_pos = self.ft_pos_traj[traj_waypoint_i, f_i*3:f_i*3+3]
            new_vel = self.ft_vel_traj[traj_waypoint_i, f_i*3:f_i*3+3]
            fingertip_pos_goal_list.append(new_pos)
            fingertip_vel_goal_list.append(new_vel)

        if self.l_wf_traj is None:
            self.tip_forces_wf = None
        else:
            self.tip_forces_wf = self.l_wf_traj[traj_waypoint_i, :]

        # Print fingertip goal position and velocities to stdout for logging
        row = [self.step_count, time.time()]
        # Formulate row to print csv_row = "{},".format(self.step_count)
        for f_i in range(3):
            for d in range(3):
                row.append(fingertip_pos_goal_list[f_i][d])
        for f_i in range(3):
            for d in range(3):
                row.append(fingertip_vel_goal_list[f_i][d])
        with open(self.csv_filepath, mode="w") as fid:
            writer  = csv.writer(fid, delimiter=",")
            writer.writerow(row)

        # Compute torque with impedance controller, and clip
        torque = c_utils.impedance_controller(fingertip_pos_goal_list,
                                              fingertip_vel_goal_list,
                                              current_position, current_velocity,
                                              self.custom_pinocchio_utils,
                                              tip_forces_wf=self.tip_forces_wf,
                                              Kp = KP, Kv = KV)

        torque = np.clip(torque, self.action_space.low, self.action_space.high)

        self.traj_waypoint_counter += 1

        return torque

    """
    Get fingertip positions in world frame given current joint q
    """
    def get_fingertip_pos_wf(self, current_q):
        fingertip_pos_wf = self.custom_pinocchio_utils.forward_kinematics(current_q)
        return fingertip_pos_wf

class HierarchicalControllerPolicy:
    DIST_THRESH = 0.09
    ORI_THRESH = np.pi / 6

    RESET_TIME_LIMIT = 50
    RL_RETRY_STEPS = 70
    MAX_RETRIES = 3

    default_robot_position = TriFingerPlatform.spaces.robot_position.default

    def __init__(self, action_space=None, initial_pose=None, goal_pose=None,
                 npz_file=None, load_dir='', start_mode=PolicyMode.RL_PUSH, 
                 difficulty=1, deterministic=True, debug_waypoints=False):
        self.full_action_space = action_space
        action_space = action_space['torque']
        self.impedance_controller = ImpedanceControllerPolicy(
                action_space, initial_pose, goal_pose, npz_file, debug_waypoints=debug_waypoints)
        self.load_policy(load_dir, deterministic)
        self.start_mode = start_mode
        self._platform = None
        self.steps_from_reset = 0
        self.step_count = self.rl_start_step = 0
        self.traj_initialized = False
        self.rl_retries = int(self.start_mode == PolicyMode.RL_PUSH)
        self.difficulty = difficulty

    def reset_policy(self, platform=None):
        self.mode = self.start_mode
        self.traj_initialized = False
        self.steps_from_reset = self.step_count = self.rl_start_step = 0
        if platform:
            self._platform = platform
        self.impedance_controller.reset_policy(platform)

    @property
    def platform(self):
        assert self._platform is not None, 'HierarchicalControlPolicy.platform is not set'
        return self._platform

    @platform.setter
    def platform(self, platform):
        assert platform is not None, 'platform is not yet initialized'
        self._platform = platform
        self.impedance_controller.platform = platform

    def load_policy(self, load_dir, deterministic=False):
        self.observation_names = []
        if not load_dir:
            self.rl_frameskip = 1
            self.rl_observation_space = None
            self.rl_policy = lambda obs: self.impedance_controller.predict(obs)
        elif osp.exists(load_dir) and 'pyt_save' in os.listdir(load_dir):
            self.load_spinup_policy(load_dir, deterministic=deterministic)
        else:
            self.load_sb_policy(load_dir)

    def load_sb_policy(self, load_dir):
        # loads make_env, make_reorient_env, and make_model helpers
        assert 'HER-SAC' in load_dir, 'only configured HER-SAC policies so far'
        if '_push' in load_dir:
            self.rl_env = sb_utils.make_env()
        else:
            self.rl_env = sb_utils.make_reorient_env()
        self.rl_frameskip = self.rl_env.unwrapped.frameskip
        self.observation_names = list(self.rl_env.unwrapped.observation_space.spaces.keys())
        self.rl_observation_space = self.rl_env.observation_space
        self.sb_policy = sb_utils.make_her_sac_model(None, None)
        self.sb_policy.load(load_dir)
        self.rl_policy = lambda obs: self.sb_policy.predict(obs)[0]

    def load_spinup_policy(self, load_dir, load_itr='last', deterministic=False):
        self.rl_env, self.rl_policy = load_policy_and_env(load_dir, load_itr, deterministic)
        if self.rl_env:
            self.rl_frameskip = self.rl_env.frameskip
        else:
            self.rl_frameskip = 10
        self.observation_names = list(self.rl_env.unwrapped.observation_space.spaces.keys())
        self.rl_observation_space = self.rl_env.observation_space
        print('loaded policy from {}'.format(load_dir))

    def activate_rl(self, obj_pose):
        if self.start_mode != PolicyMode.RL_PUSH or self.rl_retries == self.MAX_RETRIES:
            if self.rl_retries == self.MAX_RETRIES and self.difficulty == 4:
                self.difficulty = 3
            return False
        return np.linalg.norm(obj_pose.position[:2] - np.zeros(2)) > self.DIST_THRESH

    def initialize_traj_opt(self, observation):
        obj_pose = get_pose_from_observation(observation)
        goal_pose = get_pose_from_observation(observation, goal_pose=True)

        # TODO: check orientation error
        if (self.activate_rl(obj_pose) and
            self.start_mode == PolicyMode.RL_PUSH and
            self.mode != PolicyMode.RESET):
            if self.mode != PolicyMode.RL_PUSH:
                self.mode = PolicyMode.RL_PUSH
                self.rl_start_step = self.step_count
            elif self.step_count - self.rl_start_step == self.RL_RETRY_STEPS:
                self.mode = PolicyMode.RESET
            return False
        elif self.mode == PolicyMode.RL_PUSH:
            if self.step_count > 0:
                self.mode = PolicyMode.RESET
                return False
            else: # skips reset if starting at RL_PUSH
                self.mode = PolicyMode.TRAJ_OPT
                return True
        elif (self.mode == PolicyMode.RESET and
              (self.steps_from_reset >= self.RESET_TIME_LIMIT and
               obj_pose.position[2] < 0.034)):
            self.steps_from_reset = 0
            if self.activate_rl(obj_pose):
                self.rl_retries += 1
                self.mode = PolicyMode.RL_PUSH
                self.rl_start_step = self.step_count
                return False
            else:
                self.mode = PolicyMode.TRAJ_OPT
                return True
        elif self.mode == PolicyMode.TRAJ_OPT:
            return True
        else:
            if (self.impedance_controller.flipping and 
                self.impedance_controller.done_with_primitive):
                self.mode = PolicyMode.RESET
                return False
            return True

    def set_waypoints(self, observation):
        if self.mode == PolicyMode.TRAJ_OPT:
            init_pose = get_pose_from_observation(observation)
            goal_pose = get_pose_from_observation(observation, goal_pose=True)
            if self.difficulty == 4:
                self.impedance_controller.set_init_goal(
                        init_pose, goal_pose, flip=flip_needed(init_pose, goal_pose))
            else:
                self.impedance_controller.set_init_goal(init_pose, goal_pose)

            #self.impedance_controller.set_traj_to_object(observation)
            self.traj_initialized = True  # pre_traj_wp are initialized
            self.mode = PolicyMode.IMPEDANCE

    def reset_action(self, observation):
        robot_position = observation['observation']['position']
        time_limit_step = self.RESET_TIME_LIMIT // 3
        if self.steps_from_reset < time_limit_step:  # middle
            robot_position[1::3] = self.default_robot_position[1::3]
        elif time_limit_step <= self.steps_from_reset < 2*time_limit_step:  # tip
            robot_position[2::3] = self.default_robot_position[2::3]
        else:  # base
            robot_position[::3] = self.default_robot_position[::3]
        self.steps_from_reset += 1
        return robot_position

    def predict(self, observation):
        if not self.traj_initialized and self.initialize_traj_opt(observation['impedance']):
            self.set_waypoints(observation['impedance'])

        if self.mode == PolicyMode.RL_PUSH and self.rl_observation_space is not None:
            ac = self.rl_policy(observation['rl'])
            ac = np.clip(ac, self.full_action_space['position'].low,
                         self.full_action_space['position'].high)
        elif self.mode == PolicyMode.RESET:
            ac = self.reset_action(observation['impedance'])
            ac = np.clip(ac, self.full_action_space['position'].low,
                         self.full_action_space['position'].high)
        elif self.mode == PolicyMode.IMPEDANCE:
            ac = self.impedance_controller.predict(observation['impedance'])
            if self.impedance_controller.done_with_primitive:
                self.traj_initialized = False
        else:
            assert False, 'use a different start mode, started with: {}'.format(self.start_mode)
        self.step_count += 1
        return ac
    

class ResidualControllerPolicy(HierarchicalControllerPolicy):
    DIST_THRESH = 0.09
    ORI_THRESH = np.pi / 6
    default_robot_position = TriFingerPlatform.spaces.robot_position.default

    def __init__(self, action_space=None, initial_pose=None, goal_pose=None,
                 npz_file=None, start_mode=PolicyMode.RL_PUSH, difficulty=1, 
                 rl_torque=True, rl_tip_pos=False, rl_cp_params=False,
                 debug_waypoints=False):
        super(action_space, initial_pose, goal_pose, npz_file, load_dir='',
              start_mode=PolicyMode.TRAJ_OPT, difficulty=difficulty, deterministic=True,
              debug_waypoints=debug_waypoints)
        self.rl_torque = rl_torque
        self.rl_tip_pos = rl_tip_pos
        self.rl_cp_params = rl_cp_params

    def process_observation_rl(self, observation, torque):
        if self.rl_torque:
            observation = np.concatenate([observation, torque])
        return observation

    def predict(self, observation):
        if not self.traj_initialized and self.initialize_traj_opt(observation['impedance']):
            self.set_waypoints(observation['impedance'])
        torque = self.impedance_controller.predict(observation['impedance'])
        rl_obs = self.process_observation_rl(observation['rl'], torque)
        if self.rl_torque:
            torque += self.rl_policy(rl_obs)
        return torque


def get_pose_from_observation(observation, goal_pose=False):
    key = 'achieved_goal' if not goal_pose else 'desired_goal'
    return move_cube.Pose.from_dict(observation[key])

def flip_needed(init_pose, goal_pose):
    return (c_utils.get_closest_ground_face(init_pose) !=
            c_utils.get_closest_ground_face(goal_pose))

def get_robot_position_velocity(observation):
    observation = observation['observation']
    return observation['position'], observation['velocity']


def load_policy_and_env(fpath, itr='last', deterministic=False):
    """
    Load a policy from save, whether it's TF or PyTorch, along with RL env.

    Not exceptionally future-proof, but it will suffice for basic uses of the
    Spinning Up implementations.

    Checks to see if there's a tf1_save folder. If yes, assumes the model
    is tensorflow and loads it that way. Otherwise, loads as if there's a
    PyTorch save.
    """

    backend = 'pytorch'

    # handle which epoch to load from
    if itr=='last':
        # check filenames for epoch (AKA iteration) numbers, find maximum value

        if backend == 'pytorch':
            pytsave_path = osp.join(fpath, 'pyt_save')
            # Each file in this folder has naming convention 'modelXX.pt', where
            # 'XX' is either an integer or empty string. Empty string case
            # corresponds to len(x)==8, hence that case is excluded.
            saves = [int(x.split('.')[0][5:]) for x in os.listdir(pytsave_path) if len(x)>8 and 'model' in x]

        itr = '%d'%max(saves) if len(saves) > 0 else ''
    else:
        assert isinstance(itr, int), \
            "Bad value provided for itr (needs to be int or 'last')."
        itr = '%d'%itr

    # load the get_action function
    get_action = load_pytorch_policy(fpath, itr, deterministic)

    # try to load environment from save
    # (sometimes this will fail because the environment could not be pickled)
    state = joblib.load(osp.join(fpath, 'vars'+itr+'.pkl'))
    env = state['env']

    return env, get_action


def load_pytorch_policy(fpath, itr, deterministic=False):
    """ Load a pytorch policy saved with Spinning Up Logger."""

    fname = osp.join(fpath, 'pyt_save', 'model'+itr+'.pt')
    print('\n\nLoading from %s.\n\n'%fname)

    model = torch.load(fname)

    # make function for producing an action given a single state
    def get_action(x):
        with torch.no_grad():
            x = torch.as_tensor(x, dtype=torch.float32)
            if deterministic:
                action = model.pi(x)[0].mean.numpy()
            else:
                action = model.act(x)
        return action

    return get_action

