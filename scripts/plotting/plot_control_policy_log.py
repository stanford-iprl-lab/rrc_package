import numpy as np
import argparse
import matplotlib.pyplot as plt
import os.path as osp

parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str)
args = parser.parse_args()

output_dir = osp.dirname(args.filename)
# Open npz file
data = np.load(args.filename)

# Get arrays from npz file
step_count        = data["step_count"]
timestamp         = data["timestamp"]
desired_ft_pos    = data["desired_ft_pos"]
desired_ft_vel    = data["desired_ft_vel"]
actual_ft_pos     = data["actual_ft_pos"]
desired_obj_pose  = data["desired_obj_pose"]
observed_obj_pose = data["observed_obj_pose"]
desired_ft_force  = data["desired_ft_force"]
desired_torque    = data["desired_torque"]

# Plot desired object trajectory
steps = step_count.shape[0]

plt.figure(figsize=(20,10))
plt.suptitle("Fingertip positions")
plt.subplots_adjust(hspace=0.3)
for f_i in range(3):
    for d_i, dim in enumerate(["x","y","z"]):
        plt.subplot(3,3,f_i*3+d_i+1)
        plt.title("Finger {} dimension {}".format(f_i, dim))
        plt.plot(range(steps), desired_ft_pos[:,f_i*3+d_i], '.', label="Desired")
        plt.plot(range(steps), actual_ft_pos[:,f_i*3+d_i], '.', label="Observed")
        plt.legend()
plt.savefig("{}/ft_pos.png".format(output_dir))

plt.figure(figsize=(10,10))
plt.suptitle("Object position")
for d_i, dim in enumerate(["x","y","z"]):
    plt.subplot(3,1,d_i+1)
    plt.title("Dimension {}".format(dim))
    plt.plot(range(steps), desired_obj_pose[:,d_i], '.', label="Desired")
    plt.plot(range(steps), observed_obj_pose[:,d_i], '.', label="Observed")
plt.savefig("{}/obj_pos.png".format(output_dir))




