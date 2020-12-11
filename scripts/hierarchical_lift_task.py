#!/usr/bin/env python3
"""Demo on how to run the robot using the Gym environment

This demo creates a RealRobotCubeEnv environment and runs one episode using a
dummy policy which uses random actions.
"""
import json
import sys
import os
import os.path as osp
import numpy as np

from rrc_iprl_package.envs import cube_env, custom_env
from trifinger_simulation.tasks import move_cube 
from rrc_iprl_package.control.controller_utils import PolicyMode
from rrc_iprl_package.control.control_policy import HierarchicalControllerPolicy

FRAMESKIP = 1
MAX_STEPS = 120 * 1000
EP_LEN = None

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
    initial_pose.position = np.array([0.0,0.0,move_cube._CUBOID_SIZE[2]/2])
    theta = 0
    initial_pose.orientation = np.array([0, 0, np.sin(theta/2), np.cos(theta/2)])
   
    eps_so_far = 0
    if osp.exists('/output'):
        save_path = lambda: '/output/action_log{}.npz'.format(eps_so_far)
    else:
        save_path = lambda: 'action_log{}.npz'.format(eps_so_far)
    ep_len = EP_LEN or MAX_STEPS
    env = cube_env.RealRobotCubeEnv(
        goal, initial_pose.to_dict(), difficulty,
        cube_env.ActionType.TORQUE_AND_POSITION, frameskip=FRAMESKIP,
        num_steps=ep_len, visualization=True, save_npz=None
    )

    if difficulty == 4:
        if osp.exists('/ws/src/usercode'):
            rl_load_dir = '/ws/src/usercode/models/ppo-scaled-rew-cuboid/ppo-scaled-rew-cuboid_s0'
        else:
            rl_load_dir = './models/ppo-scaled-rew-cuboid/ppo-scaled-rew-cuboid_s0'
        start_mode = PolicyMode.RL_PUSH
    else:
        rl_load_dir, start_mode = '', PolicyMode.TRAJ_OPT
    goal_pose = move_cube.Pose.from_dict(goal)
    policy = HierarchicalControllerPolicy(action_space=env.action_space,
                   initial_pose=initial_pose, goal_pose=goal_pose,
                   load_dir=rl_load_dir, difficulty=difficulty,
                   start_mode=start_mode)
    policy.load_policy(rl_load_dir, ac_wrappers=('scaled',), relative=(False,False,True), 
                       pos_coef=.5, ori_coef=.5, frameskip=15, ep_len=600)
    env = custom_env.HierarchicalPolicyWrapper(env, policy)
    observation = env.reset()

    accumulated_reward = 0
    is_done = False
    old_mode = policy.mode
    steps_so_far = 0
    try:
        while not is_done:
            if MAX_STEPS is not None and steps_so_far == MAX_STEPS: break
            action = policy.predict(observation)
            old_obs = observation
            observation, reward, is_done, info = env.step(action)
            env.write_action_log(old_obs, action, reward)
            if old_mode != policy.mode:
                #print('mode changed: {} to {}'.format(old_mode, policy.mode))
                old_mode = policy.mode
            #print("reward:", reward)
            accumulated_reward += reward
            steps_so_far += env.frameskip
            if EP_LEN is not None and is_done and (eps_so_far+1)*EP_LEN < MAX_STEPS:
                is_done = False
                eps_so_far += 1
                env.save_action_log(save_path())
                observation = env.reset()

    except Exception as e:
        print("Error encounted: {}. Saving logs and exiting".format(e))
        env.save_action_log(save_path())
        policy.impedance_controller.save_log()
        raise e

    env.save_action_log(save_path())
    # Save control_policy_log
    policy.impedance_controller.save_log()

    #print("------")
    #print("Accumulated Reward: {:.3f}".format(accumulated_reward))


if __name__ == "__main__":
    main()
