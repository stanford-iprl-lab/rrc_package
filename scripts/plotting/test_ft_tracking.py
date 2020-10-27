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
    args = parser.parse_args()

    data = pandas.read_csv(args.filename, delim_whitespace=True, header=0, low_memory=False)

    # Get actual joint positions from data
    for index, row in data.iterrows():
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
        print(k_utils.FK(cur_q))
        print(cur_q)
        quit()
    # Goal fingertip positions...? Need to be parsed from user_stdout.txt


if __name__ == "__main__":
    main()
