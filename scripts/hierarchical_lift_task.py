#!/usr/bin/env python3
"""Demo on how to run the robot using the Gym environment

This demo creates a RealRobotCubeEnv environment and runs one episode using a
dummy policy which uses random actions.
"""
import json
import sys
import numpy as np

from rrc_iprl_package.envs import cube_env, custom_env
from trifinger_simulation.tasks import move_cube 
from rrc_iprl_package.control.controller_utils import PolicyMode
from rrc_iprl_package.control.control_policy import HierarchicalControllerPolicy

#MAX_STEPS = 3 * 1000 / 1
MAX_STEPS = None
FRAMESKIP = 1

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
    goal = json.loads(goal_pose_json)
    initial_pose = move_cube.sample_goal(-1)
    initial_pose.position = np.array([0,0,.0325])

    env = cube_env.RealRobotCubeEnv(
        goal, initial_pose, difficulty,
        cube_env.ActionType.TORQUE_AND_POSITION, frameskip=FRAMESKIP
    )
    rl_load_dir, start_mode = '', PolicyMode.TRAJ_OPT
    initial_pose = move_cube.sample_goal(difficulty=-1)
    initial_pose.position = np.array([0,0,.0325])
    goal_pose = move_cube.Pose.from_dict(goal)
    policy = HierarchicalControllerPolicy(action_space=env.action_space,
                   initial_pose=initial_pose, goal_pose=goal_pose,
                   load_dir=rl_load_dir, difficulty=difficulty,
                   start_mode=start_mode)
    env = custom_env.ResidualPolicyWrapper(env, policy)
    observation = env.reset()

    accumulated_reward = 0
    is_done = False
    old_mode = policy.mode
    steps_so_far = 0
    while not is_done:
        if MAX_STEPS is not None and steps_so_far == MAX_STEPS: break
        action = policy.predict(observation)
        observation, reward, is_done, info = env.step(action)
        if old_mode != policy.mode:
            #print('mode changed: {} to {}'.format(old_mode, policy.mode))
            old_mode = policy.mode
        #print("reward:", reward)
        accumulated_reward += reward
        steps_so_far += 1

    #print("------")
    #print("Accumulated Reward: {:.3f}".format(accumulated_reward))


if __name__ == "__main__":
    main()
