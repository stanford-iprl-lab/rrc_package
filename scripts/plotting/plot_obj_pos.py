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

log_reader = tricamera.LogReader("camera_data.dat")
goal_pose_json = "goal.json"
with open(goal_pose_json) as f:
    goal = json.load(f)['goal']
goal_pose = move_cube.Pose.from_dict(goal)

positions = np.array(
    [observation.object_pose.position for observation in log_reader.data]
)

filtered_positions = np.array(
    [observation.filtered_object_pose.position for observation in log_reader.data]
)

plt.figure()

plt.plot(positions[:, 0], c="r", label="x position - observed")
plt.plot(positions[:, 1], c="g", label="y position - observed")
plt.plot(positions[:, 2], c="b", label="z position - observed")

plt.plot(filtered_positions[:, 0],"--", label="x position - filtered")
plt.plot(filtered_positions[:, 1],"--", label="y position - filtered")
plt.plot(filtered_positions[:, 2],"--", label="z position - filtered")

goal_x = goal_pose.position[0]
goal_y = goal_pose.position[1]
goal_z = goal_pose.position[2]
plt.plot(np.ones(positions[:, 1].shape) * goal_x, "--", c="r", label="x position - desired")
plt.plot(np.ones(positions[:, 1].shape) * goal_y, "--", c="g", label="y position - desired")
plt.plot(np.ones(positions[:, 1].shape) * goal_z, "--", c="b", label="z position - desired")

plt.xlabel("Time (0.1 sec)")
plt.ylabel("Position (m)")

plt.legend()
plt.title("Observed object position")
plt.savefig("obj_pos.png")

# Plot object orientation
quats = np.array(
    [observation.object_pose.orientation for observation in log_reader.data]
)

filtered_quats = np.array(
    [observation.filtered_object_pose.orientation for observation in log_reader.data]
)

plt.figure()

plt.plot(quats[:, 0], c="r", label="qx - observed")
plt.plot(quats[:, 1], c="g", label="qy - observed")
plt.plot(quats[:, 2], c="b", label="qz - observed")
plt.plot(quats[:, 3], c="c", label="qw - observed")

plt.plot(filtered_quats[:, 0], label="qx - filtered")
plt.plot(filtered_quats[:, 1], label="qy - filtered")
plt.plot(filtered_quats[:, 2], label="qz - filtered")
plt.plot(filtered_quats[:, 3], label="qw - filtered")


goal_qx = goal_pose.orientation[0]
goal_qy = goal_pose.orientation[1]
goal_qz = goal_pose.orientation[2]
goal_qw = goal_pose.orientation[3]
#plt.plot(np.ones(quats[:, 1].shape) * goal_qx, "--", c="r", label="qx - desired")
#plt.plot(np.ones(quats[:, 1].shape) * goal_qy, "--", c="g", label="qy - desired")
#plt.plot(np.ones(quats[:, 1].shape) * goal_qz, "--", c="b", label="qz - desired")
#plt.plot(np.ones(quats[:, 1].shape) * goal_qw, "--", c="c", label="qw - desired")

plt.xlabel("Time (0.1 sec)")
plt.ylabel("Quaternion element")

plt.legend()
plt.title("Observed object quaternion elements")
plt.savefig("obj_quat.png")
