import gym
import numpy as np
import os.path as osp

from rrc_iprl_package.envs import rrc_utils, custom_env, cube_env, env_wrappers
from rrc_iprl_package.envs.cube_env import ActionType
from spinup.utils.test_policy import load_policy_and_env


def run_eval(
        n_episodes,
        level,
        env=None,
        policy=None,
        gamma=1.,
        visualize=False,
        residual=False,
        randomize=False,
        info_kwargs=[]):
    initializer = env_wrappers.RandomInitializer(level)
    if env is None:
        action_type = ActionType.POSITION if not residual else ActionType.TORQUE
        env = cube_env.CubeEnv(initializer, goal_difficulty=level,
                               action_type=action_type)
    if residual:
        env = custom_env.ResidualPolicyWrapper(env)
    if randomize:
        env = env_wrappers.ObservationNoiseWrapper(env)
    env.unwrapped.visualization = visualize
    if policy is None:
        predict_fn = lambda x: np.zeros(9)
    else:
        predict_fn = policy

    done = False
    total_rew = 0
    info_kwargs = {k: [] for k in info_kwargs}
    info_sum = {k: [] for k in info_kwargs}
    returns = []
    for ep in range(n_episodes):
        obs = env.reset()
        while not done:
            obs, r, done, i = env.step(predict_fn(obs))
            total_rew = total_rew * gamma + r
            for k in info_kwargs:
                if 'Final' not in k:
                    info_kwargs[k].append(i.get(k))
        print("Episode {} total return: {}".format(ep, total_rew))
        for k in info_kwargs:
            if "Final" == k[-5:]:
                shortk = k[:-5]
                info_sum[k].append(info_kwargs[shortk][-1])
            else:
                info_sum[k].append(np.mean(info_kwargs[k]))
        info_kwargs = {k: [] for k in info_kwargs}
        returns.append(total_rew)
        total_rew = 0
        done = False

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
             randomize=args.randomize)


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
    args = parser.parse_args()
    main(args)
