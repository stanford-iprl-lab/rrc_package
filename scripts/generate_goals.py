import json
import os
import argparse

from trifinger_simulation.tasks import move_cube


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_goals', default=3,
                        help='number of goals to sample')
    parser.add_argument('--outdir', default='./goals/',
                        help='dir to save generated goals')
    parser.add_argument('--overwrite', action='store_true', 
                        help='overwriting previously generated goals')
    parser.add_argument('--difficulty', '-d', nargs='?', default=[1, 2, 3], 
                        help='difficulty levels to save goals')
    args = parser.parse_args()

    print('num_goals:', args.num_goals, 'output dir:', args.outdir, 
          'difficulty:', args.difficulty)
    gen_goals = {}
    goals = [] 
    for d in args.difficulty:
        for i in range(args.num_goals):
            if not os.path.exists(args.outdir):
                os.makedirs(args.outdir)

            offset = 0
            savepath = lambda: os.path.join(args.outdir,
                                            'goal{}{}.json'.format(d, i+offset))
            if not args.overwrite and d != 2:
                while os.path.exists(savepath()): offset += 1
            goal_path = savepath()
            
            g = move_cube.sample_goal(d)

            # level 3 goals copy level 1 goal xy positions
            if d == 3 and 1 in gen_goals:
                g.position[:2] = gen_goals[1][i].position[:2]
            goals.append(g)

            with open(goal_path, 'w') as f:
                json.dump({k: [float(x) for x in v] for k,v in g.to_dict().items()}, f)

            # only generate one level 2 goal
            if d == 2:
                break
        gen_goals[d] = goals
        goals = []

if __name__ == '__main__':
    main()
