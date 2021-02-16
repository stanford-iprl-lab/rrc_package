import gym
import numpy as np
import os.path as osp
import time

from trifinger_simulation.tasks import move_cube
from rrc_iprl_package.envs import rrc_utils, custom_env, cube_env, env_wrappers
from rrc_iprl_package.envs import initializers
from rrc_iprl_package.envs.cube_env import ActionType
from spinup.utils.test_policy import load_policy_and_env
from trifinger_simulation import trifingerpro_limits
from scipy.spatial.transform import Rotation


def run_eval(
        n_episodes,
        level,
        env=None,
        policy=None,
        gamma=1.,
        visualize=False,
        residual=False,
        randomize=False,
        random_policy=False,
        res_torque=False,
        task_space=False,
        info_kwargs=['is_success']):
    init_pose = initializers.FixedInitializer.def_initial_pose
    goal_pose = initializers.FixedInitializer.def_goal_pose

    ori = Rotation.from_quat(init_pose.orientation).as_euler('xyz')
    ori += np.array([0,0,np.pi/2])
    ori = Rotation.from_euler('xyz', ori).as_quat()

    init_pose.position += np.array([0,0.05,0])
    init_pose.orientation = ori
    goal_pose.orientation = ori
    initializer = initializers.FixedInitializer(level, initial_state=init_pose,
                                                goal=goal_pose)
    print(initializer.initial_state.orientation)
    if env is None:
        if residual:
            action_type = ActionType.TORQUE
            env = cube_env.CubeEnv(initializer, goal_difficulty=level,
                                   action_type=action_type, frameskip=1)
        else:
            action_type = ActionType.TORQUE
            env = custom_env.PushCubeEnv(initializer, action_type)
            # env = env_wrappers.TaskSpaceWrapper(env, relative=True)
            env = env_wrappers.SingleFingerWrapper(env, finger_id=0)
    if randomize:
        goal_env = isinstance(env, gym.GoalEnv)
        env = env_wrappers.ObservationNoiseWrapper(env, goal_env=goal_env)
    if residual:
        env = custom_env.ResidualPolicyWrapper(env, goal_env=True, 
                rl_torque=res_torque)
    elif task_space:
        env = rrc_utils.build_env_fn(task_space=True, ts_relative=True,
                                     scaled_ac=False,
                                     goal_relative=True, action_type='pos',
                                     ep_len=600, frameskip=25)()
    env.unwrapped.visualization = visualize
    if policy is None:
        if random_policy:
            predict_fn = lambda x: env.action_space.sample()
        else:
            if not isinstance(env, gym.Wrapper) and action_type == ActionType.POSITION:
                predict_fn = lambda x: trifingerpro_limits.robot_position.default
            else:
                predict_fn = lambda x: env.action_space.sample()
    else:
        predict_fn = policy

    done = False
    total_rew = 0
    info_kwargs = {k: [] for k in info_kwargs}
    info_sum = {k: [] for k in info_kwargs}
    returns = []
    try:
        for ep in range(n_episodes):
            start = time.time()
            if residual:
                obs = env.reset(timed=False)
            else:
                obs = env.reset()
            end = time.time()
            print(str(start - end), 'seconds')
            ep_len = 0 
            while not done:
                ac = predict_fn(obs)
                obs, r, done, i = env.step(ac)
                total_rew = total_rew * gamma + r
                for k in info_kwargs:
                    if 'Final' not in k and k in i:
                        info_kwargs[k].append(i.get(k))
                ep_len += 1
            print("Episode {} total return: {}".format(ep, total_rew))
            for k in info_kwargs:
                if "Final" == k[-5:]:
                    shortk = k[:-5]
                    if shortk in i:
                        info_sum[k].append(info_kwargs[shortk][-1])
                elif k in i:
                    info_sum[k].append(np.mean(info_kwargs[k]))
            info_kwargs = {k: [] for k in info_kwargs}
            returns.append(total_rew)
            total_rew = 0
            done = False
    except KeyboardInterrupt:
        print('Last observation:', obs)
        print('Last info:', i)

    for k in info_sum:
        print("{} mean: {}".format(k, np.mean(info_sum[k])))
        print("{} std: {}".format(k, np.std(info_sum[k])))


def load_policy(load_path, itr='last', deterministic=True):
    if load_path and osp.exists(osp.expanduser(load_path)):
        return load_policy_and_env(load_path, itr, deterministic)
    return None, None


def main(args):
    env, policy = load_policy(args.load_path)
    run_eval(n_episodes=args.num_episodes, level=args.level, env=env,
             policy=policy, visualize=args.visualize, residual=args.residual,
             randomize=args.randomize, task_space=args.task_space,
             random_policy=args.random_policy)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path', type=str)
    parser.add_argument('--steps_per_epoch', type=int, default=None)
    parser.add_argument('--num_episodes', '-n', type=int, default=10)
    parser.add_argument('--level', type=int, default=4)
    parser.add_argument('--visualize', '-v', action='store_true')
    parser.add_argument('--residual', '--res', action='store_true')
    parser.add_argument('--randomize', action='store_true')
    parser.add_argument('--random_policy', '--rp', action='store_true')
    parser.add_argument('--task_space', action='store_true')
    args = parser.parse_args()
    main(args)
