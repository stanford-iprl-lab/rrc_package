import numpy as np

from rrc_iprl_package.envs.env_utils import configurable
from scipy.spatial.transform import Rotation
from trifinger_simulation.tasks import move_cube

MAX_DIST = move_cube._max_cube_com_distance_to_center
DIST_THRESH = 0.03

_CUBOID_WIDTH = max(move_cube._CUBOID_SIZE)
_CUBOID_HEIGHT = min(move_cube._CUBOID_SIZE)


def random_xy(sample_radius_min=0., sample_radius_max=None):
    # sample uniform position in circle (https://stackoverflow.com/a/50746409)
    radius = np.random.uniform(sample_radius_min, sample_radius_max)
    theta = np.random.uniform(0, 2 * np.pi)
    # x,y-position of the cube
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return x, y


@configurable(pickleable=True)
class FixedInitializer:
    """Initializer that uses fixed values for initial pose and goal."""
    def_goal_pose = move_cube.Pose(np.array([0,0,_CUBOID_HEIGHT/2]),
                                   np.array([0,0,0,1]))
    def_initial_pose = move_cube.Pose(np.array([0,0,_CUBOID_HEIGHT/2]),
                                      np.array([0,0,0,1]))

    def __init__(self, difficulty=1, initial_state=None, goal=None):
        """Initialize.

        Args:
            difficulty (int):  Difficulty level of the goal.  This is still
                needed even for a fixed goal, as it is also used for computing
                the reward (the cost function is different for the different
                levels).
            initial_state (move_cube.Pose):  Initial pose of the object.
            goal (move_cube.Pose):  Goal pose of the object.

        Raises:
            Exception:  If initial_state or goal are not valid.  See
            :meth:`move_cube.validate_goal` for more information.
        """
        initial_state = initial_state or self.def_initial_pose
        goal = goal or self.def_goal_pose

        move_cube.validate_goal(initial_state)
        move_cube.validate_goal(goal)
        self.difficulty = difficulty
        self.initial_state = initial_state
        self.goal = goal

    def get_initial_state(self):
        """Get the initial state that was set in the constructor."""
        return self.initial_state

    def get_goal(self):
        """Get the goal that was set in the constructor."""
        return self.goal

    def update_initializer(self, final_pose, goal_pose):
        pass


@configurable(pickleable=True)
class RandomInitializer(FixedInitializer):
    """Initializer that returns random initial pose and goal."""
    def __init__(self, difficulty):
        self.difficulty = difficulty

    def get_initial_state(self):
        return move_cube.sample_goal(difficulty=-1)

    def get_goal(self):
        return move_cube.sample_goal(difficulty=self.difficulty)


@configurable(pickleable=True)
class RandomPushInitializer(FixedInitializer):
    def __init__(self, difficulty=1, initial_state=None, goal=None,
                 push_upper=0.05, push_lower=0.):
        super(RandomPushInitializer, self).__init__(difficulty, initial_state, goal)
        self.push_lower, self.push_upper = push_lower, push_upper
        ori = Rotation.from_quat(self.def_goal_pose.orientation).as_euler('xyz')
        ori += np.array([0,0,np.pi/2])
        self.ori = Rotation.from_euler('xyz', ori).as_quat()

    def get_initial_state(self):
        init_pose = move_cube.Pose(self.def_initial_pose.position.copy(),
                                   self.def_initial_pose.orientation.copy())
        init_pose.position += np.array([0,0.05,0])
        init_pose.orientation = self.ori
        return init_pose

    def get_goal(self):
        push_dist = np.random.uniform(self.push_lower, self.push_upper)
        goal_pose = move_cube.Pose(self.def_goal_pose.position.copy(),
                                   self.def_goal_pose.orientation.copy())
        goal_pose.position += np.array([0, -push_dist, 0])
        goal_pose.orientation = self.ori
        return goal_pose


@configurable(pickleable=True)
class CurriculumInitializer(FixedInitializer):
    """Initializer that samples random initial states and goals."""

    def __init__(self, difficulty=1, initial_dist=_CUBOID_WIDTH/2,
                 num_levels=3, buffer_size=5, fixed_goal=None):
        """Initialize.

        Args:
            initial_dist (float): Distance from center of arena
            num_levels (int): Number of steps to maximum radius
            buffer_size (int): Number of episodes to compute mean over
        """
        self.difficulty = difficulty
        self.num_levels = num_levels
        self._current_level = 0
        self.levels = np.linspace(initial_dist, 0.09, num_levels)
        self.initial_pose = self.def_initial_pose
        self.final_dist = np.array([np.inf for _ in range(buffer_size)])
        if difficulty == 4:
            self.final_ori = np.array([np.inf for _ in range(buffer_size)])
        self.fixed_goal = fixed_goal is not None
        self.goal_pose = fixed_goal

    @property
    def current_level(self):
        return min(self.num_levels - 1, self._current_level)

    def update_initializer(self, final_pose, goal_pose):
        assert np.all(goal_pose.position == self.goal_pose.position)
        # add final pos and ori errors to buffers
        self.final_dist = np.roll(self.final_dist, 1)
        final_dist = np.linalg.norm(goal_pose.position - final_pose.position)
        self.final_dist[0] = final_dist
        if self.difficulty == 4:
            self.final_ori = np.roll(self.final_ori, 1)
            self.final_ori[0] = compute_orientation_error(goal_pose, final_pose)

        update_level = np.mean(self.final_dist) < DIST_THRESH
        if self.difficulty == 4:
            update_level = update_level and np.mean(self.final_ori) < ORI_THRESH

        if update_level and self._current_level < self.num_levels - 1:
            self._current_level += 1

    def get_initial_state(self):
        """Get a random initial object pose (always on the ground)."""
        sample_radius_min, sample_radius_max = DIST_THRESH, self.levels[self.current_level]
        x, y = random_xy(sample_radius_min, sample_radius_max)
        self.initial_pose = move_cube.sample_goal(difficulty=-1)
        z = self.initial_pose.position[-1]
        self.initial_pose.position = np.array((x, y, z))
        return self.initial_pose

    @property
    def goal_sample_radius(self):
        if self.fixed_goal:
            goal_dist = np.linalg.norm(self.goal_pose.position)
            return (goal_dist, goal_dist)
        level_idx = min(self.num_levels - 1, self._current_level + 1)
        sample_radius_max = self.levels[level_idx]
        return (0., sample_radius_max)

    def get_goal(self):
        """Get a random goal depending on the difficulty."""
        if self.fixed_goal:
            return self.goal_pose
        # goal_sample_radius is further than past distances
        sample_radius_min, sample_radius_max = self.goal_sample_radius
        x, y = random_xy(sample_radius_min, sample_radius_max)
        while np.linalg.norm(np.array([x,y]) - self.initial_pose.position[:2]) < DIST_THRESH:
            x, y = random_xy(sample_radius_min, sample_radius_max)
        self.goal_pose = move_cube.sample_goal(difficulty=self.difficulty)
        self.goal_pose.position = np.array((x, y, self.goal_pose.position[-1]))
        return self.goal_pose


@configurable(pickleable=True)
class ReorientInitializer(FixedInitializer):
    """Initializer that samples random initial states and goals."""
    def_goal_pose = move_cube.Pose(np.array([0,0,_CUBOID_HEIGHT/2]), np.array([0,0,0,1]))

    def __init__(self, difficulty=1, initial_dist=_CUBOID_HEIGHT, seed=None):
        self.difficulty = difficulty
        self.initial_dist = initial_dist
        self.goal_pose = self.def_goal_pose
        self.set_seed(seed)

    def set_seed(self, seed):
        self.random = np.random.RandomState(seed=seed)

    def get_initial_state(self):
        """Get a random initial object pose (always on the ground)."""
        x, y = random_xy(self.initial_dist, MAX_DIST)
        self.initial_pose = move_cube.sample_goal(difficulty=-1)
        z = self.initial_pose.position[-1]
        self.initial_pose.position = np.array((x, y, z))
        return self.initial_pose

    def get_goal(self):
        """Get a random goal depending on the difficulty."""
        if self.difficulty >= 2:
            self.goal_pose = move_cube.sample_goal(self.difficulty)
        return self.goal_pose


@configurable(pickleable=True)
class RandomGoalOrientationInitializer(FixedInitializer):
    def_initial_pose= move_cube.Pose(np.array([0,0,_CUBOID_HEIGHT/2]),
                                     np.array([0,0,0,1]))

    def __init__(self, difficulty=1, max_dist=np.pi):
        self.difficulty = difficulty
        self.max_dist = max_dist
        self.init_pose = self.def_initial_pose

    def get_initial_state(self):
        self.init_pose = move_cube.sample_goal(-1)
        x,y = random_xy(DIST_THRESH, 0.09)
        self.init_pose.position[:2] = np.array([x,y])
        return self.init_pose

    def get_goal(self):
        return self.def_goal_pose


@configurable(pickleable=True)
class RandomOrientationInitializer(FixedInitializer):
    def_goal_pose = move_cube.Pose(np.array([0,0,_CUBOID_HEIGHT/2]), np.array([0,0,0,1]))

    def __init__(self, difficulty=4):
        self.difficulty = difficulty

    def get_initial_state(self):
        return move_cube.sample_goal(-1)

    def get_goal(self):
        return self.def_goal_pose

