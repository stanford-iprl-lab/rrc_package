import IPython

import gym
import os
import os.path as osp
import time
import numpy as np
import functools

from rrc_iprl_package.envs import cube_env, custom_env, env_wrappers
from spinup.utils import rrc_utils
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import HER, SAC
from stable_baselines.common.atari_wrappers import FrameStack


def make_l2_goal_env():
    info_keys = ['is_success', 'is_success_ori_dist', 'dist', 'final_dist', 'final_score',
                 'final_ori_dist']

    wrappers = [gym.wrappers.ClipAction,
                functools.partial(env_wrappers.LogInfoWrapper,info_keys=info_keys),
                functools.partial(env_wrappers.CubeRewardWrapper, pos_coef=1.,
                                ori_coef=1.,
                                ac_norm_pen=0.2, rew_fn='exp',
                                goal_env=True),
                functools.partial(env_wrappers.ReorientWrapper, goal_env=True,
                                dist_thresh=0.06),
                functools.partial(custom_env.ResidualPolicyWrapper, goal_env=True),
                functools.partial(gym.wrappers.TimeLimit, max_episode_steps=rrc_utils.EPLEN),
                env_wrappers.FlattenGoalWrapper]
    initializer = env_wrappers.ReorientInitializer(2, 0.1)
    env_fn = rrc_utils.make_env_fn('real_robot_challenge_phase_2-v1', wrapper_params=wrappers,
                                   action_type=cube_env.ActionType.TORQUE,
                                   goal_difficulty=2,
                                   initializer=initializer,
                                   frameskip=rrc_utils.FRAMESKIP,
                                   )
    env = env_fn()
    return env


info_keys = ['is_success', 'is_success_ori_dist', 'dist', 'final_dist', 'final_score',
             'final_ori_dist']

wrappers = [gym.wrappers.ClipAction,
            functools.partial(env_wrappers.LogInfoWrapper,info_keys=info_keys),
            functools.partial(env_wrappers.CubeRewardWrapper, pos_coef=1.,
                            ori_coef=1.,
                            ac_norm_pen=0.2, rew_fn='exp',
                            goal_env=True),
            functools.partial(env_wrappers.ReorientWrapper, goal_env=True,
                              dist_thresh=0.06),
            functools.partial(custom_env.ResidualPolicyWrapper, goal_env=False),
            functools.partial(gym.wrappers.TimeLimit, max_episode_steps=rrc_utils.EPLEN),
            ]
initializer = env_wrappers.ReorientInitializer(3, 0.1)
l3_env_fn = rrc_utils.make_env_fn('real_robot_challenge_phase_2-v1', wrapper_params=wrappers,
                               action_type=cube_env.ActionType.TORQUE,
                               goal_difficulty=3,
                               initializer=initializer,
                               frameskip=rrc_utils.FRAMESKIP,
                               )
 

def make_l3_env(**kwargs):
    return l3_env_fn(**kwargs)




def make_env():
    info_keys = ['is_success', 'is_success_ori_dist', 'dist', 'final_dist', 'final_score',
                 'final_ori_dist', 'init_sample_radius']

    wrappers = [gym.wrappers.ClipAction,
                functools.partial(env_wrappers.LogInfoWrapper, info_keys=info_keys),
                functools.partial(gym.wrappers.TimeLimit, max_episode_steps=rrc_utils.EPLEN),
                env_wrappers.FlattenGoalWrapper]
    cube_wrapper = functools.partial(env_wrappers.CubeRewardWrapper, pos_coef=1., ori_coef=1.,
                                ac_norm_pen=0.2, rew_fn='exp',
                                goal_env=True)
    initializer = env_wrappers.ReorientInitializer(1, 0.1)
    env_fn = rrc_utils.make_env_fn('real_robot_challenge_phase_2-v2', wrapper_params=wrappers,
                                   action_type=cube_env.ActionType.TORQUE,
                                   initializer=initializer,
                                   frameskip=rrc_utils.FRAMESKIP,
                                   )
    env = env_fn()
    return env


def make_exp_dir():
    exp_root = './data'
    hms_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    exp_name = 'HER-SAC_sparse_push'
    exp_dir = osp.join(exp_root, exp_name, hms_time)
    os.makedirs(exp_dir)
    return exp_dir


def make_model(env, exp_dir):
    model = HER('MlpPolicy', env, SAC, n_sampled_goal=4,
                tensorboard_log=exp_dir,
                goal_selection_strategy='future',
                verbose=1, buffer_size=int(1e4),
                learning_rate=3e-5,
                gamma=0.95, batch_size=256,
                policy_kwargs=dict(layers=[256, 256]))
    return model


def train_save_model(model, exp_dir, steps=1e6, reset_num_timesteps=False):
# Train for 1e6 steps
    model.learn(int(steps), reset_num_timesteps=reset_num_timesteps)
# Save the trained agent
    model.save(osp.join(exp_dir, '{}-steps'.format(model.num_timesteps)))
    return model


def main():
    env = make_reorient_env()
    exp_dir = make_exp_dir()
    model = make_model(env, exp_dir)
    train_save_model(model, exp_dir, 1e6)
    IPython.embed()


if __name__ == '__main__':
    main()
