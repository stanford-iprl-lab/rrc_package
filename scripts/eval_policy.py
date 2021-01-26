import gym
import numpy as np

from rrc_iprl_package.envs import rrc_utils, custom_env, cube_env, env_wrappers


def run_eval(n_episodes=10, level=1, policy=None, gamma=.99,
        info_kwargs=[]):
    initializer = env_wrappers.RandomInitializer(level)
    env = cube_env.CubeEnv(initializer, goal_difficulty=level)
    if policy is None:
        predict_fn = lambda x: np.zeros(9)
    else:
        predict_fn = policy.predict

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


if __name__ == '__main__':
    run_eval()
