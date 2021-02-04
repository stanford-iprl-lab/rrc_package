import gym
from gym.envs.registration import register
from . import env_wrappers

registered_envs = [spec.id for spec in gym.envs.registry.all()]

if "real_robot_challenge_phase_2-v0" not in registered_envs:
    register(
        id="real_robot_challenge_phase_2-v0",
        entry_point="rrc_iprl_package.envs.cube_env:RealRobotCubeEnv",
    )

if "real_robot_challenge_phase_2-v1" not in registered_envs:
    register(
        id="real_robot_challenge_phase_2-v1",
        entry_point="rrc_iprl_package.envs.cube_env:CubeEnv",
    )

for lv in [1, 2, 3, 4]:
    if f"real_robot_challenge_phase_2_lv{lv}-v1" not in registered_envs:
        register(
            id=f"real_robot_challenge_phase_2_lv{lv}-v1",
            entry_point="rrc_iprl_package.envs.cube_env:CubeEnv",
            max_episode_steps=600,
            kwargs={'goal_difficulty': lv,
                    'frameskip': 25}
        )

if "real_robot_challenge_phase_2-v2" not in registered_envs:
    register(
        id="real_robot_challenge_phase_2-v2",
        entry_point="rrc_iprl_package.envs.custom_env:PushCubeEnv",
    )

if "real_robot_challenge_phase_2-v3" not in registered_envs:
    register(
        id="real_robot_challenge_phase_2-v3",
        entry_point="rrc_iprl_package.envs.cube_env:FlattenedCubeEnv",
    )


