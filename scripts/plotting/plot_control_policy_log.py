import numpy as np
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str)
args = parser.parse_args()

# Open npz file
data = np.load(args.filename)

# Get arrays from npz file
step_count       = data["step_count"]
timestamp        = data["timestamp"]
desired_ft_pos   = data["desired_ft_pos"]
desired_ft_vel   = data["desired_ft_vel"]
actual_ft_pos    = data["actual_ft_pos"]
desired_obj_pose = data["desired_obj_pose"]
desired_ft_force = data["desired_ft_force"]

# Plot desired object trajectory
steps = step_count.shape[0]
plt.figure()
plt.plot(range(steps), desired_obj_pose[:,2], '.')
plt.savefig("temp.png")




