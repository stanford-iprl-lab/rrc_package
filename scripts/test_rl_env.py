import gym
import numpy as np
from rrc_iprl_package.envs import rrc_utils


steps = 1000
episodes = 10

env = rrc_utils.p2_reorient_env_fn(visualization=True)

for ep in range(episodes):
    obs = env.reset()
    for step in range(steps):
        env.step(np.array([0.0, 0.75, -1.6] * 3))

