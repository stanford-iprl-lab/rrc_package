from gym.envs.registration import register

register(
    id="real_robot_challenge_phase_2-v1",
    entry_point="rrc_simulation.gym_wrapper.envs.cube_env:CubeEnv",
)

register(
    id="real_robot_challenge_phase_2-v2",
    entry_point="rrc_simulation.gym_wrapper.envs.custom_env:PushCubeEnv",
)

