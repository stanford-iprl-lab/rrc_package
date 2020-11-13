import argparse
import os.path as osp
import numpy as np
import json

from trifinger_simulation.tasks import move_cube 
import trifinger_object_tracking.py_tricamera_types as tricamera

"""
Plot observed object position from camera_data.dat and goal pose from goal.json
To run on singularity image:
singularity run ~/realrobotchallenge.sif python3 ~/rrc_package/scripts/save_obj_pos.py
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
    np.savez('positions.npz', positions=positions, goal=dict(position=goal_pose.position,
                                                             orientation=goal_pose.orientation))
    return
