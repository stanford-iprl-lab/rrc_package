import numpy as np
import pandas
import argparse
import matplotlib.pyplot as plt

import rrc_iprl_package.traj_opt.kinematics_utils as k_utils

"""
Test fingertip position tracking
"""
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str)
    parser.add_argument("stdout", type=str)
    parser.add_argument("fig_file", type=str)
    args = parser.parse_args()

    data = pandas.read_csv(args.filename, delim_whitespace=True, header=0, low_memory=False)
    stdout = pandas.read_csv(args.stdout, delim_whitespace=False, header=0, low_memory=False)

    n = int(len(data.index) / 4) + 1
    # Get actual joint positions from data
    observation_ft_pos = np.zeros((n, 9))
    i = 0
    for index, row in data.iterrows():
        if index % 4 != 0: continue
        print(index)
        print(i)
        cur_q = [
                row["observation_position_0"],
                row["observation_position_1"],
                row["observation_position_2"],
                row["observation_position_3"],
                row["observation_position_4"],
                row["observation_position_5"],
                row["observation_position_6"],
                row["observation_position_7"],
                row["observation_position_8"],
                ]
        cur_ft_pos = k_utils.FK(cur_q)
        observation_ft_pos[i, :] = cur_ft_pos
        #print(custom_pinocchio_utils.forward_kinematics(np.array(cur_q)))
        i += 1

    # Goal fingertip positions...? Need to be parsed from user_stdout.txt

    # Plot
    plt.figure(figsize=(20,20))
    plt.suptitle("Fingertip positions")
    plt.subplots_adjust(hspace=0.3)
    for f_i in range(3):
        for d_i, dim in enumerate(["x","y","z"]):
            plt.subplot(3,3,f_i*3+d_i+1)
            plt.title("Finger {} dimension {}".format(f_i, dim))
            plt.scatter(range(observation_ft_pos.shape[0]), observation_ft_pos[:,f_i*3+d_i],label="Observed")
            plt.scatter(range(len(stdout.index)), stdout[["desired_ft{}".format(f_i*3+d_i)]].to_numpy(), label="Desired")
            plt.legend()
    plt.savefig(args.fig_file)


if __name__ == "__main__":
    main()
