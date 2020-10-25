#!/usr/bin/env python3
"""
Testing impedance controller for lifting object
Right now a bit of a mess, but will clean up soon.
"""
import argparse
import time
from datetime import date, datetime
import matplotlib.pyplot as plt
import pybullet
import os
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

import cv2
import numpy as np

from trifinger_simulation import trifinger_platform, sample, visual_objects
from trifinger_simulation.tasks import move_cube
from trifinger_simulation.control.custom_pinocchio_utils import CustomPinocchioUtils
from trifinger_simulation.control.controller_utils import *
from trifinger_simulation.traj_opt.fixed_contact_point_opt import FixedContactPointOpt
from trifinger_simulation.gym_wrapper.envs import custom_env

num_fingers = 3
episode_length = move_cube.episode_length
#episode_length = 500

DIFFICULTY = 2
TEST_FLIPPING = True
#cube_half_size = move_cube._CUBE_WIDTH/2
cube_half_size = move_cube._CUBE_WIDTH/2 + 0.008 # Fudge the cube dimensions slightly for computing contact point positions in world frame to account for fingertip radius

def main(args):
  if args.npz_file is not None:
    # Open .npz file and parse
    npzfile   = np.load(args.npz_file)
    nGrid     = npzfile["t"].shape[0]
    x_goal    = npzfile["x_goal"]
    x0        = npzfile["x0"]
    x_soln    = npzfile["x"]
    l_wf_soln = npzfile["l_wf"]
    dt        = npzfile["dt"]
    cp_params = npzfile["cp_params"]

  else:
    #45 degrees around x
    theta = 0
    theta0 = 0
    #theta = np.pi/2
    #theta = -np.pi/2
    y_theta = R.from_quat(np.array([0,np.sin(theta/2), 0, np.cos(theta/2)]))
    x_theta = R.from_quat(np.array([np.sin(theta/2),0, 0, np.cos(theta/2)]))
    z_theta = R.from_quat(np.array([0, 0, np.sin(theta/2),np.cos(theta/2)]))

    x0 = np.array([[0.02,0.02,0.0325,0,0.707,0,0.707]])  # face 3
    #x0 = np.array([[0.02,0.02,0.0325,0,-0.707,0,0.707]])  # face 5
    #x0 = np.array([[0.02,0.02,0.0325,-0.707,0,0,0.707]])  # face 2

    #x0 = np.array([[0.0,0.0,0.0325,0,0,np.sin(theta0/2),np.cos(theta0/2)]]) 
    x_goal = np.array([[0,0,0.0325+0.05,0,0,0,1]]) 

    # Ground face 1
    base_quat = R.from_quat(np.array([0.707, 0, 0, 0.707]))
    quat = base_quat * y_theta

    # Ground face 2
    #base_quat = R.from_quat(np.array([-0.707, 0, 0, 0.707]))
    #quat = base_quat * y_theta

    # Ground face 3
    #base_quat = R.from_quat(np.array([0,0.707, 0, 0.707]))
    #quat = base_quat * x_theta

    # Ground face 4
    #base_quat = R.from_quat(np.array([0,-1, 0, 0]))
    #quat = base_quat * z_theta

    # Ground face 5
    #base_quat = R.from_quat(np.array([0,-0.707, 0, 0.707]))
    #quat = base_quat * x_theta

    # Ground face 6
    #z45 = R.from_quat(np.array([0,0,np.sin(theta/2),np.cos(theta/2)]))
    #base_quat = R.from_quat(np.array([0, 0, 0, 1]))
    #quat = base_quat * z45
    ##x0 = np.array([[0,0,0.0325,0,0,0,1]]) 

    #x0[0,3:] = quat.as_quat()

    x_goal[0,3:] = quat.as_quat()

    nGrid = 50
    dt = 0.01

  # Save directory
  x_goal_str = "-".join(map(str,x_goal[0,:].tolist()))
  x0_str = "-".join(map(str,x0[0,:].tolist()))
  today_date = date.today().strftime("%m-%d-%y")
  save_dir = "./logs/{}/x0_{}_xgoal_{}_nGrid_{}_dt_{}".format(today_date ,x0_str, x_goal_str, nGrid, dt)
  # Create directory if it does not exist
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)
  
  # Set initial object pose to match npz file
  x0_pos = x0[0,0:3]
  x0_quat = x0[0,3:]
  #x0_pos = np.array([0, 0.01, 0.0325]) # test
  #x0_pos = np.array([1, 1, 0.0325]) # test
  #x0_quat = np.array([0, 0, -0.707, 0.707]) # test
  init_object_pose = move_cube.Pose(
                position=x0_pos,
                orientation=x0_quat,
            )
  goal_object_pose = move_cube.Pose(
                position=x_goal[0,0:3],
                orientation=x_goal[0,3:],
            )
  #init_object_pose = None # Use default init pose
  
  platform = trifinger_platform.TriFingerPlatform(
      visualization=args.visualize, enable_cameras=args.enable_cameras, initial_object_pose=init_object_pose
  )

  # Instantiate custom pinocchio utils class for access to Jacobian
  custom_pinocchio_utils = CustomPinocchioUtils(platform.simfinger.finger_urdf_path, platform.simfinger.tip_link_names) 

  if TEST_FLIPPING:
    current_position = platform.get_robot_observation(0).position
    fingertips_current = custom_pinocchio_utils.forward_kinematics(current_position)
    cp_params = get_flipping_cp_params(init_object_pose, goal_object_pose, cube_half_size)
    x_soln = None
    l_wf_soln = None
  else: 
    if args.npz_file is None:
      obj_pose = platform.get_object_pose(0)
      current_position = platform.get_robot_observation(0).position
      x_soln, l_wf_soln, cp_params = run_traj_opt(obj_pose, current_position, custom_pinocchio_utils, x0, x_goal, nGrid, dt, save_dir)

  #cp_params = np.array([
  #                      [-0.7, 1, 0.7],
  #                      [-0.7, -1, 0.7],
  #                      [-1.6, 1, 0]])

  fingertip_pos_list, x_pos_list, x_quat_list, x_goal, fingertip_goal_list = run_episode(platform,
                                                                              custom_pinocchio_utils,
                                                                              nGrid,
                                                                              x0,
                                                                              x_goal,
                                                                              x_soln,
                                                                              l_wf_soln,
                                                                              cp_params,
                                                                              )

  plot_state(save_dir, fingertip_pos_list, x_pos_list, x_quat_list, x_goal,fingertip_goal_list)

"""
Run episode
Inputs:
nGrid
"""
def run_episode(platform, custom_pinocchio_utils,
                nGrid,
                x0,
                x_goal,
                x_soln,
                l_wf_soln,
                cp_params,
                ):

  # Lists for storing values to plot
  fingertip_pos_list = [[],[],[]] # Containts 3 lists, one for each finger
  fingertip_goal_log = [[],[],[]] # Containts 3 lists, one for each finger
  x_pos_list = [] # Object positions
  x_quat_list = [] # Object positions

  x0_pos = x0[0,0:3]
  x0_quat = x0[0,3:]
  init_object_pose = move_cube.Pose(
                position=x0_pos,
                orientation=x0_quat,
            )

  #pybullet.resetDebugVisualizerCamera(cameraDistance=1.54, cameraYaw=4.749999523162842, cameraPitch=-42.44065475463867, cameraTargetPosition=(-0.11500892043113708, 0.6501579880714417, -0.6364855170249939))
  custom_env.reset_camera()
  # MP4 logging
  mp4_save_string = "./test.mp4"
  if args.save_viz_mp4:
    pybullet.startStateLogging(pybullet.STATE_LOGGING_VIDEO_MP4, mp4_save_string)

  # Take first action
  finger_action = platform.Action(position=platform.spaces.robot_position.default)
  t = platform.append_desired_action(finger_action)
  # Get object pose
  obj_pose = platform.get_object_pose(t)

  # Visual markers
  init_cps = visual_objects.Marker(number_of_goals=num_fingers, goal_size=0.008)
  finger_waypoints = visual_objects.Marker(number_of_goals=num_fingers, goal_size=0.008)

  # Draw target contact points
  #target_cps_wf = get_cp_wf_list_from_cp_params(cp_params, x0_pos, x0_quat, cube_half_size)
  #init_cps.set_state(target_cps_wf)

  # Get initial fingertip positions in world frame
  current_position = platform.get_robot_observation(t).position
  fingertips_init = custom_pinocchio_utils.forward_kinematics(current_position)

  # Get initial contact points and waypoints to them
  finger_waypoints_list = []
  for f_i in range(3):
    #print("finger {}".format(f_i))
    tip_current = custom_pinocchio_utils.forward_kinematics(current_position)[f_i]
    waypoints = get_waypoints_to_cp_param(obj_pose, cube_half_size, tip_current, cp_params[f_i])
    finger_waypoints_list.append(waypoints)
  
  pre_traj_waypoint_i = 0
  traj_waypoint_i = 0
  goal_reached = False
  reward = 0
  goal_pose = move_cube.Pose(position=x_goal[0,0:3], orientation=x_goal[0,3:])

  flipping_wp = None
  for timestep in tqdm(range(episode_length)):

    # Get joint positions        
    current_position = platform.get_robot_observation(t).position
    # Joint velocities
    current_velocity = platform.get_robot_observation(t).velocity
  
    # Follow trajectory to position fingertips before moving to object
    if pre_traj_waypoint_i < len(finger_waypoints_list[0]):
      # Get fingertip goals from finger_waypoints_list
      fingertip_goal_list = []
      for f_i in range(num_fingers):
        fingertip_goal_list.append(finger_waypoints_list[f_i][pre_traj_waypoint_i])
      #print(fingertip_goal_list)
      tol = 0.009
      tip_forces_wf = None
    # Follow trajectory to lift object
    else:
      if TEST_FLIPPING:
        fingertip_goal_list = flipping_wp
        tip_forces_wf = None
      else:
        if traj_waypoint_i < nGrid:
          fingertip_goal_list = []
          next_cube_pos_wf = x_soln[traj_waypoint_i, 0:3]
          next_cube_quat_wf = x_soln[traj_waypoint_i, 3:]
          fingertip_goal_list = get_cp_wf_list_from_cp_params(cp_params,
                                                              next_cube_pos_wf,
                                                              next_cube_quat_wf,
                                                              cube_half_size)
          # Get target contact forces in world frame 
          tip_forces_wf = l_wf_soln[traj_waypoint_i, :]
          tol = 0.008
        
    finger_waypoints.set_state(fingertip_goal_list)

    torque, goal_reached = impedance_controller(
                                  fingertip_goal_list,
                                  current_position,
                                  current_velocity,
                                  custom_pinocchio_utils,
                                  tip_forces_wf = tip_forces_wf,
                                  tol           = tol
                                  )

    if goal_reached:
      obj_pose = platform.get_object_pose(t)
      current_position = platform.get_robot_observation(t).position
      fingertips_current = custom_pinocchio_utils.forward_kinematics(current_position)
      flipping_wp, flip_done = get_flipping_waypoint(platform.get_object_pose(t), goal_pose, fingertips_current, fingertips_init, cp_params)
      goal_reached = False
      if pre_traj_waypoint_i < len(finger_waypoints_list[0]):
        pre_traj_waypoint_i += 1
      elif traj_waypoint_i < nGrid:
        print("trajectory waypoint: {}".format(traj_waypoint_i))
        traj_waypoint_i += 1

    # Save current state for plotting
    # Add fingertip positions to list
    current_position = platform.get_robot_observation(t).position
    for finger_id in range(3):
      tip_current = custom_pinocchio_utils.forward_kinematics(current_position)[finger_id]
      fingertip_pos_list[finger_id].append(tip_current)
      fingertip_goal_log[finger_id].append(fingertip_goal_list[finger_id])
    # Add current object pose to list
    obj_pose = platform.get_object_pose(t)
    x_pos_list.append(obj_pose.position)
    x_quat_list.append(obj_pose.orientation)

    # Accumulate reward
    r = -move_cube.evaluate_state(
            goal_pose,
            obj_pose,
            DIFFICULTY,
        )
    reward += r
  
    clipped_torque = np.clip(
            np.asarray(torque),
            -platform._max_torque_Nm,
            +platform._max_torque_Nm,
        )
    # Check torque limits
    #print("Torque upper limits: {}".format(platform.spaces.robot_torque))
    if not platform.spaces.robot_torque.gym.contains(clipped_torque):
      print("Time {} Actual torque: {}".format(t, clipped_torque))
    #if not platform.spaces.robot_position.gym.contains(current_position):
    #  print("Actual position: {}".format(current_position))

    # Apply torque action
    finger_action = platform.Action(torque=clipped_torque)
    t = platform.append_desired_action(finger_action)

    #time.sleep(platform.get_time_step())

  # Compute score
  print("Reward: {}".format(reward))

  return fingertip_pos_list, x_pos_list, x_quat_list, x_goal, fingertip_goal_log
  
"""
PLOTTING
"""
def plot_state(save_dir, fingertip_pos_list, x_pos_list, x_quat_list, x_goal, fingertip_goal_list):
  total_timesteps = episode_length

  # Plot end effector trajectory
  fingertip_pos_array = np.array(fingertip_pos_list)
  fingertip_goal_array = np.array(fingertip_goal_list)
  x_pos_array = np.array(x_pos_list)
  x_quat_array = np.array(x_quat_list)

  ## Object position
  plt.figure(figsize=(12, 9))
  plt.subplots_adjust(hspace=0.3)
  for i in range(3):
    plt.subplot(3, 1, i+1)
    plt.title("Fingertip {} position".format(i))
    plt.plot(list(range(total_timesteps)), fingertip_pos_array[i,:,0], c="C0", label="x")
    plt.plot(list(range(total_timesteps)), fingertip_pos_array[i,:,1], c="C1", label="y")
    plt.plot(list(range(total_timesteps)), fingertip_pos_array[i,:,2], c="C2", label="z")
    plt.plot(list(range(total_timesteps)), fingertip_goal_array[i,:,0], ":", c="C0", label="x_goal")
    plt.plot(list(range(total_timesteps)), fingertip_goal_array[i,:,1], ":", c="C1", label="y_goal")
    plt.plot(list(range(total_timesteps)), fingertip_goal_array[i,:,2], ":", c="C2", label="z_goal")
  plt.legend()
  if args.save_state_log:
    plt.savefig("{}/fingertip_positions.png".format(save_dir))

  plt.figure()
  plt.suptitle("Object pose")
  plt.figure(figsize=(6, 12))
  plt.subplots_adjust(hspace=0.3)
  plt.subplot(2, 1, 1)
  plt.title("Object position")
  for i in range(3):
    plt.plot(list(range(total_timesteps)), x_pos_array[:,i], c="C{}".format(i), label="actual - dimension {}".format(i))
    plt.plot(list(range(total_timesteps)), np.ones(total_timesteps)*x_goal[0,i], ":", c="C{}".format(i), label="goal")
  plt.legend()
  
  plt.subplot(2, 1, 2)
  plt.title("Object Orientation")
  for i in range(4):
    plt.plot(list(range(total_timesteps)), x_quat_array[:,i], c="C{}".format(i), label="actual - dimension {}".format(i))
    plt.plot(list(range(total_timesteps)), np.ones(total_timesteps)*x_goal[0,i+3], ":", c="C{}".format(i), label="goal")
  plt.legend()

  if args.save_state_log:
    plt.savefig("{}/object_position.png".format(save_dir))

""" Get contact point positions in world frame from cp_params
"""
def get_cp_wf_list_from_cp_params(cp_params, cube_pos, cube_quat, cube_half_size=cube_half_size):
  # Get contact points in wf
  fingertip_goal_list = []
  for i in range(num_fingers):
    fingertip_goal_list.append(get_cp_wf_from_cp_param(cp_params[i], cube_pos, cube_quat, cube_half_size))
  return fingertip_goal_list

"""
For testing - hold joints at initial position
"""
def _test_hold_initial_state():
  # For testing - hold joints at initial positions
  while(1):
    finger_action = platform.Action(position=platform.spaces.robot_position.default)
    t = platform.append_desired_action(finger_action)
    time.sleep(platform.get_time_step())

    # Debug visualizer camera params
    camParams = pybullet.getDebugVisualizerCamera()
    print("cameraDistance={}, cameraYaw={}, cameraPitch={}, cameraTargetPosition={}".format(camParams[-2], camParams[-4], camParams[-3], camParams[-1]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--enable-cameras",
        "-c",
        action="store_true",
        help="Enable camera observations.",
    )
    parser.add_argument(
        "--save_state_log",
        "-s",
        action="store_true",
        help="Save plots of state over episode",
    )
    parser.add_argument(
        "--save_viz_mp4",
        "-sv",
        action="store_true",
        help="Save MP4 of visualization.",
    )
    parser.add_argument(
        "--visualize",
        "-v",
        action="store_true",
        help="Visualize with GUI.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of motions that are performed.",
    )
    parser.add_argument(
        "--save-action-log",
        type=str,
        metavar="FILENAME",
        help="If set, save the action log to the specified file.",
    )

    parser.add_argument("--npz_file")

    args = parser.parse_args()

    main(args)