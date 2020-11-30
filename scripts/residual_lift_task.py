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
from gym.wrappers import TimeLimit

from rrc_iprl_package.envs import cube_env, custom_env, env_wrappers
from trifinger_simulation.tasks import move_cube 
from rrc_iprl_package.control.controller_utils import PolicyMode
from rrc_iprl_package.control.control_policy import HierarchicalControllerPolicy
from rrc_iprl_package import run_rrc_sb as sb_utils 

FRAMESKIP = 1
#MAX_STEPS = 3 * 1000 // FRAMESKIP
EP_LEN = 120 * 1000 // FRAMESKIP - 150 // FRAMESKIP
MAX_STEPS = 120 * 1000

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

    if osp.exists('/output'):
        save_path = '/output/action_log.npz'
    else:
        save_path = 'action_log.npz'

    env = cube_env.RealRobotCubeEnv(
        goal, initial_pose.to_dict(), difficulty,
        cube_env.ActionType.TORQUE, frameskip=FRAMESKIP,
        num_steps=MAX_STEPS, visualization=True, save_npz=save_path
    )
    if osp.exists('/ws/src/usercode'):
        rl_load_dir = '/ws/src/usercode/models/HER.zip'
    else:
        rl_load_dir = './models/HER.zip'
    env = custom_env.ResidualPolicyWrapper(env, goal_env=True)
    env = TimeLimit(env, max_episode_steps=EP_LEN)
    env = env_wrappers.FlattenGoalWrapper(env)
    policy = sb_utils.make_model(env, None)
    policy.load(rl_load_dir)
    observation = env.reset()

    accumulated_reward = 0
    is_done = False
    steps_so_far = 0
    while not is_done:
        if MAX_STEPS is not None and steps_so_far == MAX_STEPS: break
        action, _ = policy.predict(observation)
        observation, reward, is_done, info = env.step(action)
        accumulated_reward += reward
        steps_so_far = info.get('num_steps', steps_so_far + 1)
    env.save_action_log()

    print("------")
    print("Accumulated Reward: {:.3f}".format(accumulated_reward))


if __name__ == "__main__":
    main()
