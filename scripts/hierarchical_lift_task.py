#!/usr/bin/env python3
"""Demo on how to run the robot using the Gym environment

This demo creates a RealRobotCubeEnv environment and runs one episode using a
dummy policy which uses random actions.
"""
import csv
import json
import sys
import os
import os.path as osp
import numpy as np
from gym.wrappers import TimeLimit

from rrc_iprl_package.envs import cube_env, custom_env, env_wrappers
from trifinger_simulation.tasks import move_cube 
from rrc_iprl_package.control.controller_utils import PolicyMode
from rrc_iprl_package.control.control_policy import HierarchicalControllerPolicy
from rrc_iprl_package import run_rrc_sb as sb_utils 

FRAMESKIP = 1
EP_LEN = 60 * 1000 // FRAMESKIP - 150 // FRAMESKIP
MAX_STEPS = 15 * 1000

class RandomPolicy:
    """Dummy policy which uses random actions."""

    def __init__(self, action_space):
        self.action_space = action_space

    def predict(self, observation):
        return self.action_space.sample()


def main():
    # the difficulty level and the goal pose (as JSON string) are passed as
    # arguments
    difficulty = int(sys.argv[1])
    goal_pose_json = sys.argv[2]
    if os.path.exists(goal_pose_json):
        with open(goal_pose_json) as f:
            goal = json.load(f)['goal']
    else:
        goal = json.loads(goal_pose_json)
    initial_pose = move_cube.sample_goal(-1)
    initial_pose.position = np.array([0,0,move_cube._CUBOID_SIZE[2]/2])
    theta = 0.0
    initial_pose.orientation = np.array([0, 0, np.sin(theta/2), np.cos(theta/2)])
   
    if osp.exists('/output'):
        save_path = '/output/action_log.npz'
    else:
        save_path = 'action_log.npz'

    if difficulty == -2:
        env = cube_env.RealRobotCubeEnv(
            goal, initial_pose.to_dict(), difficulty,
            cube_env.ActionType.TORQUE, frameskip=FRAMESKIP,
            num_steps=MAX_STEPS, visualization=True, save_npz=save_path
        )
        if os.path.exists('/ws/src/usercode'):
            rl_load_dir = '/ws/src/usercode/models/HER.zip'
        else:
            rl_load_dir = './models/HER.zip'
        env = custom_env.ResidualPolicyWrapper(env, goal_env=True)
        env = TimeLimit(env, max_episode_steps=EP_LEN)
        env = env_wrappers.FlattenGoalWrapper(env)
        policy = sb_utils.make_model(env, None)
        policy.load(rl_load_dir)
    else:
        env = cube_env.RealRobotCubeEnv(
            goal, initial_pose.to_dict(), difficulty,
            cube_env.ActionType.TORQUE_AND_POSITION, frameskip=FRAMESKIP,
            num_steps=MAX_STEPS, visualization=True, save_npz=save_path
        )
        rl_load_dir, start_mode = '', PolicyMode.TRAJ_OPT
        goal_pose = move_cube.Pose.from_dict(goal)
        policy = HierarchicalControllerPolicy(action_space=env.action_space,
                       initial_pose=initial_pose, goal_pose=goal_pose,
                       load_dir=rl_load_dir, difficulty=difficulty,
                       start_mode=start_mode)
        env = custom_env.HierarchicalPolicyWrapper(env, policy)
    observation = env.reset()

    accumulated_reward = 0
    is_done = False
    steps_so_far = 0
    if difficulty == -2:
        while not is_done:
            if MAX_STEPS is not None and steps_so_far == MAX_STEPS: break
            action, _ = policy.predict(observation)
            observation, reward, is_done, info = env.step(action)
            accumulated_reward += reward
            steps_so_far += 1
    else:
        old_mode = policy.mode
        while steps_so_far != EP_LEN:
            # if virtual episode is not done running and real EP_LEN hasn't been reached, keep running
            if not is_done:
                # print("--------------normal stepping-------------", steps_so_far)    
                action = policy.predict(observation)
                observation, reward, is_done, info = env.step(action)
                if old_mode != policy.mode:
                    # print('mode changed: {} to {}'.format(old_mode, policy.mode))
                    old_mode = policy.mode
                #print("reward:", reward)
                accumulated_reward += reward
                steps_so_far += 1
            # if current virtual episode is done, but hasn't reached the end of real episode,
            # reset and run the next episode 
            else:
                print("--------------resetting-------------", steps_so_far)
                observation = env.reset()
                initial_pose = move_cube.Pose.from_dict(observation['impedance']['achieved_goal'])
                policy.impedance_controller.set_init_goal(initial_pose, goal_pose)
                is_done = False
                steps_so_far += 1
                continue
    env.save_action_log()

    print("------")
    print("Accumulated Reward: {:.3f}".format(accumulated_reward))


if __name__ == "__main__":
    main()
