import argparse
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import json

from trifinger_simulation.tasks import move_cube 
import trifinger_object_tracking.py_tricamera_types as tricamera

"""
Plot observed object position from camera_data.dat and goal pose from goal.json
To run on singularity image:
singularity run ~/realrobotchallenge.sif python3 ~/rrc_package/scripts/plotting/plot_obj_pos.py
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("logdir", type=str)
    args = parser.parse_args()

    log_reader = tricamera.LogReader(osp.join(args.logdir, "camera_data.dat"))
    goal_pose_json = "goal.json"
    with open(goal_pose_json) as f:
        goal = json.load(f)['goal']
    goal_pose = move_cube.Pose.from_dict(goal)

    positions = np.array(
        [observation.object_pose.position for observation in log_reader.data]
    )

    plt.plot(positions[:, 0], c="r", label="x position - observed")
    plt.plot(positions[:, 1], c="g", label="y position - observed")
    plt.plot(positions[:, 2], c="b", label="z position - observed")

    goal_x = goal_pose.position[0]
    goal_y = goal_pose.position[1]
    goal_z = goal_pose.position[2]
    plt.plot(np.ones(positions[:, 1].shape) * goal_x, "--", c="r", label="x position - actual")
    plt.plot(np.ones(positions[:, 1].shape) * goal_y, "--", c="g", label="y position - actual")
    plt.plot(np.ones(positions[:, 1].shape) * goal_z, "--", c="b", label="z position - actual")

    plt.legend()
    plt.title("Observed object position")
    # plt.savefig("obj_pos.png")
    plt.show()
