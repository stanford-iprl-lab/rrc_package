import numpy as np
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str)
args = parser.parse_args()

# Open npz file
data = np.load(args.filename)

step_count = data["step_count"]
timestamp = data["timestamp"]
desired_ft_pos = data["desired_ft_pos"]
desired_ft_vel = data["desired_ft_vel"]
actual_ft_pos = data["actual_ft_pos"]
desired_obj_pos = data["desired_obj_pose"]
desired_ft_force = data["desired_ft_force"]

print(step_count.shape)
print(timestamp.shape)
print(desired_ft_pos.shape)
print(desired_ft_vel.shape)
print(actual_ft_pos.shape)
print(desired_obj_pos.shape)
print(desired_ft_force.shape)

