import IPython
import json

from rrc_iprl_package.envs import custom_env, cube_env
from trifinger_simulation.tasks import move_cube


def main():
    gp = json.load(open('../goal.json'))['goal']
    ip = move_cube.sample_goal(-1)
    env = cube_env.RealRobotCubeEnv(gp, ip, 1, cube_env.ActionType.TORQUE_AND_POSITION)
    env = custom_env.ResidualPolicyWrapper(env, True)
    try:
        env.reset()
    except Exception as e:
        IPython.embed()
        raise e


if __name__ == '__main__':
    main()
