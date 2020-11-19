import numpy as np
import enum
from scipy.spatial.transform import Rotation
from scipy.spatial.distance import pdist, squareform

from rrc_iprl_package.control.contact_point import ContactPoint
from trifinger_simulation.tasks import move_cube
from rrc_iprl_package.traj_opt.fixed_contact_point_opt import FixedContactPointOpt
from rrc_iprl_package.traj_opt.static_object_opt import StaticObjectOpt

class PolicyMode(enum.Enum):
        RESET = enum.auto()
        TRAJ_OPT = enum.auto()
        IMPEDANCE = enum.auto()
        RL_PUSH = enum.auto()
        RESIDUAL = enum.auto()

# Object properties
OBJ_MASS = 0.016 # 16 grams
OBJ_SIZE = move_cube._CUBOID_SIZE + 0.012

# Here, hard code the base position of the fingers (as angle on the arena)
r = 0.15
theta_0 = 90
theta_1 = 310
theta_2 = 200
FINGER_BASE_POSITIONS = [
           np.array([[np.cos(theta_0*(np.pi/180))*r, np.sin(theta_0*(np.pi/180))*r, 0]]),
           np.array([[np.cos(theta_1*(np.pi/180))*r, np.sin(theta_1*(np.pi/180))*r, 0]]),
           np.array([[np.cos(theta_2*(np.pi/180))*r, np.sin(theta_2*(np.pi/180))*r, 0]]),
           ]

# Information about object faces given face_id
OBJ_FACES_INFO = {
                 1: {"center_param": np.array([0.,-1.,0.]),
                         "face_down_default_quat": np.array([0.707,0,0,0.707]),
                         "adjacent_faces": [6,4,3,5],
                         "opposite_face": 2,
                         "up_axis": np.array([0.,1.,0.]), # UP axis when this face is ground face
                         },
                 2: {"center_param": np.array([0.,1.,0.]),
                         "face_down_default_quat": np.array([-0.707,0,0,0.707]),
                         "adjacent_faces": [6,4,3,5],
                         "opposite_face": 1,
                         "up_axis": np.array([0.,-1.,0.]),
                         },
                 3: {"center_param": np.array([1.,0.,0.]),
                         "face_down_default_quat": np.array([0,0.707,0,0.707]),
                         "adjacent_faces": [1,2,4,6],
                         "opposite_face": 5,
                         "up_axis": np.array([-1.,0.,0.]),
                         },
                 4: {"center_param": np.array([0.,0.,1.]),
                         "face_down_default_quat": np.array([0,1,0,0]),
                         "adjacent_faces": [1,2,3,5],
                         "opposite_face": 6,
                         "up_axis": np.array([0.,0.,-1.]),
                         },
                 5: {"center_param": np.array([-1.,0.,0.]),
                         "face_down_default_quat": np.array([0,-0.707,0,0.707]),
                         "adjacent_faces": [1,2,4,6],
                         "opposite_face": 3,
                         "up_axis": np.array([1.,0.,0.]),
                         },
                 6: {"center_param": np.array([0.,0.,-1.]),
                         "face_down_default_quat": np.array([0,0,0,1]),
                         "adjacent_faces": [1,2,3,5],
                         "opposite_face": 4,
                         "up_axis": np.array([0.,0.,1.]),
                         },
                }

CUBOID_SHORT_FACES = [1,2]
CUBOID_LONG_FACES = [3,4,5,6]

"""
Compute joint torques to move fingertips to desired locations
Inputs:
tip_pos_desired_list: List of desired fingertip positions for each finger
q_current: Current joint angles
dq_current: Current joint velocities
tip_forces_wf: fingertip forces in world frame
tol: tolerance for determining when fingers have reached goal
"""
def impedance_controller(
                       tip_pos_desired_list,
                       tip_vel_desired_list,
                       q_current,  
                       dq_current,
                       custom_pinocchio_utils,
                       tip_forces_wf = None,
                       Kp                   = [25,25,25,25,25,25,25,25,25],
                       Kv                   = [1,1,1,1,1,1,1,1,1],
                       ):
    torque = 0
    for finger_id in range(3):
        # Get contact forces for single finger
        if tip_forces_wf is None:
            f_wf = None
        else:
            f_wf = np.expand_dims(np.array(tip_forces_wf[finger_id * 3:finger_id*3 + 3]),1)

        finger_torque = impedance_controller_single_finger(
                                                    finger_id,
                                                    tip_pos_desired_list[finger_id],
                                                    tip_vel_desired_list[finger_id],
                                                    q_current,
                                                    dq_current,
                                                    custom_pinocchio_utils,
                                                    tip_force_wf = f_wf,
                                                    Kp                   = Kp,
                                                    Kv                   = Kv,
                                                    )
        torque += finger_torque
    return torque

"""
Compute joint torques to move fingertip to desired location
Inputs:
finger_id: Finger 0, 1, or 2
tip_desired: Desired fingertip pose **ORIENTATION??**
    for orientation: transform fingertip reference frame to world frame (take into account object orientation)
    for now, just track position
q_current: Current joint angles
dq_current: Current joint velocities
tip_forces_wf: fingertip forces in world frame
tol: tolerance for determining when fingers have reached goal
"""
def impedance_controller_single_finger(
    finger_id,
    tip_pos_desired,
    tip_vel_desired,
    q_current,
    dq_current,
    custom_pinocchio_utils,
    tip_force_wf = None,
    Kp           = [25,25,25,25,25,25,25,25,25],
    Kv           = [1,1,1,1,1,1,1,1,1]
    ):
    
    Kp_x = Kp[finger_id*3 + 0]
    Kp_y = Kp[finger_id*3 + 1]
    Kp_z = Kp[finger_id*3 + 2]
    Kp = np.diag([Kp_x, Kp_y, Kp_z])

    Kv_x = Kv[finger_id*3 + 0]
    Kv_y = Kv[finger_id*3 + 1]
    Kv_z = Kv[finger_id*3 + 2]
    Kv = np.diag([Kv_x, Kv_y, Kv_z])

    # Compute current fingertip position
    x_current = custom_pinocchio_utils.forward_kinematics(q_current)[finger_id]

    delta_x = np.expand_dims(np.array(tip_pos_desired) - np.array(x_current), 1)
    #print("Current x: {}".format(x_current))
    #print("Desired x: {}".format(tip_desired))
    #print("Delta: {}".format(delta_x))
    
    # Get full Jacobian for finger
    Ji = custom_pinocchio_utils.get_tip_link_jacobian(finger_id, q_current)
    # Just take first 3 rows, which correspond to linear velocities of fingertip
    Ji = Ji[:3, :]

    # Get g matrix for gravity compensation
    _, g = custom_pinocchio_utils.get_lambda_and_g_matrix(finger_id, q_current, Ji)

    # Get current fingertip velocity
    dx_current = Ji @ np.expand_dims(np.array(dq_current), 1)

    delta_dx = np.expand_dims(np.array(tip_vel_desired),1) - np.array(dx_current)

    if tip_force_wf is not None:
        torque = np.squeeze(Ji.T @ (Kp @ delta_x + Kv @ delta_dx) + Ji.T @ tip_force_wf) + g
    else:
        torque = np.squeeze(Ji.T @ (Kp @ delta_x + Kv @ delta_dx)) + g

    #print("Finger {} delta".format(finger_id))
    #print(np.linalg.norm(delta_x))
    return torque

"""
Compute contact point position in world frame
Inputs:
cp_param: Contact point param [px, py, pz]
cube: Block object, which contains object shape info
"""
def get_cp_pos_wf_from_cp_param(cp_param, cube_pos_wf, cube_quat_wf):
    cp = get_cp_of_from_cp_param(cp_param)

    rotation = Rotation.from_quat(cube_quat_wf)
    translation = np.asarray(cube_pos_wf)

    return rotation.apply(cp.pos_of) + translation

"""
Get contact point positions in world frame from cp_params
"""
def get_cp_pos_wf_from_cp_params(cp_params, cube_pos, cube_quat):
    # Get contact points in wf
    fingertip_goal_list = []
    for i in range(len(cp_params)):
        if cp_params[i] is None:
            fingertip_goal_list.append(None)
        else:
            fingertip_goal_list.append(get_cp_pos_wf_from_cp_param(cp_params[i], cube_pos, cube_quat))
    return fingertip_goal_list

"""
Compute contact point position in object frame
Inputs:
cp_param: Contact point param [px, py, pz]
"""
def get_cp_of_from_cp_param(cp_param):
    cp_of = []
    # Get cp position in OF
    for i in range(3):
        cp_of.append(-OBJ_SIZE[i]/2 + (cp_param[i]+1)*OBJ_SIZE[i]/2)

    cp_of = np.asarray(cp_of)

    x_param = cp_param[0]
    y_param = cp_param[1]
    z_param = cp_param[2]
    # For now, just hard code quat
    if y_param == -1:
        quat = (np.sqrt(2)/2, 0, 0, np.sqrt(2)/2)
    elif y_param == 1:
        quat = (np.sqrt(2)/2, 0, 0, -np.sqrt(2)/2)
    elif x_param == 1:
        quat = (0, 0, 1, 0)
    elif z_param == 1:
        quat = (np.sqrt(2)/2, 0, np.sqrt(2)/2, 0)
    elif x_param == -1:
        quat = (1, 0, 0, 0)
    elif z_param == -1:
        quat = (np.sqrt(2)/2, 0, -np.sqrt(2)/2, 0)

    cp = ContactPoint(cp_of, quat)
    return cp

"""
Get face id on cube, given cp_param
cp_param: [x,y,z]
"""
def get_face_from_cp_param(cp_param):
    x_param = cp_param[0]
    y_param = cp_param[1]
    z_param = cp_param[2]
    # For now, just hard code quat
    if y_param == -1:
        face = 1
    elif y_param == 1:
        face = 2
    elif x_param == 1:
        face = 3
    elif z_param == 1:
        face = 4
    elif x_param == -1:
        face = 5
    elif z_param == -1:
        face = 6

    return face

"""
Trasform point p from world frame to object frame, given object pose
"""
def get_wf_from_of(p, obj_pose):
    cube_pos_wf = obj_pose.position
    cube_quat_wf = obj_pose.orientation

    rotation = Rotation.from_quat(cube_quat_wf)
    translation = np.asarray(cube_pos_wf)
    
    return rotation.apply(p) + translation

"""
Trasform point p from object frame to world frame, given object pose
"""
def get_of_from_wf(p, obj_pose):
    cube_pos_wf = obj_pose.position
    cube_quat_wf = obj_pose.orientation

    rotation = Rotation.from_quat(cube_quat_wf)
    translation = np.asarray(cube_pos_wf)
    
    rotation_inv = rotation.inv()
    translation_inv = -rotation_inv.apply(translation)

    return rotation_inv.apply(p) + translation_inv

##############################################################################
# Lift mode functions
##############################################################################
"""
Run trajectory optimization
obj_pose: current object pose (for getting contact points)
current_position: current joint positions of robot
x0: object initial position for traj opt
x_goal: object goal position for traj opt
nGrid: number of grid points
dt: delta t
"""
def run_fixed_cp_traj_opt(obj_pose, cp_params, current_position, custom_pinocchio_utils, x0, x_goal, nGrid, dt, npz_filepath = None):

    cp_params_on_obj = []
    for cp in cp_params:
        if cp is not None: cp_params_on_obj.append(cp)
    fnum = len(cp_params_on_obj)

    # Formulate and solve optimization problem
    opt_problem = FixedContactPointOpt(
                                      nGrid     = nGrid,    # Number of timesteps
                                      dt        = dt,       # Length of each timestep (seconds)
                                      fnum      = fnum,
                                      cp_params = cp_params_on_obj,
                                      x0        = x0,
                                      x_goal    = x_goal,
                                      obj_shape = OBJ_SIZE,
                                      obj_mass  = OBJ_MASS,
                                      npz_filepath = npz_filepath
                                      )
    
    x_soln     = np.array(opt_problem.x_soln)
    dx_soln    = np.array(opt_problem.dx_soln)
    l_wf_soln  = np.array(opt_problem.l_wf_soln)

    return x_soln, dx_soln, l_wf_soln
    

"""
Get initial contact points on cube
Assign closest cube face to each finger
Since we are lifting object, don't worry about wf z-axis, just care about wf xy-plane
"""
def get_lifting_cp_params(obj_pose):
    # face that is touching the ground
    ground_face = get_closest_ground_face(obj_pose)

    # Transform finger base positions to object frame
    finger_base_of = []
    for f_wf in FINGER_BASE_POSITIONS:
        f_of = get_of_from_wf(f_wf, obj_pose)
        finger_base_of.append(f_of)

    # Find distance from x axis and y axis, and store in xy_distances
    # Need some additional logic to prevent multiple fingers from being assigned to same face
    x_axis = np.array([1,0])
    y_axis = np.array([0,1])
    
    # Object frame axis corresponding to plane parallel to ground plane
    x_ind, y_ind = __get_parallel_ground_plane_xy(ground_face)
        
    xy_distances = np.zeros((3, 2)) # Row corresponds to a finger, columns are x and y axis distances
    for f_i, f_of in enumerate(finger_base_of):
        point_in_plane = np.array([f_of[0,x_ind], f_of[0,y_ind]]) # Ignore dimension of point that's not in the plane
        x_dist = __get_distance_from_pt_2_line(x_axis, np.array([0,0]), point_in_plane)
        y_dist = __get_distance_from_pt_2_line(y_axis, np.array([0,0]), point_in_plane)
        
        xy_distances[f_i, 0] = np.sign(f_of[0,y_ind]) * x_dist
        xy_distances[f_i, 1] = np.sign(f_of[0,x_ind]) * y_dist
        
    free_faces = \
        [x for x in OBJ_FACES_INFO[ground_face]["adjacent_faces"] if x not in CUBOID_SHORT_FACES]

    # For each face, choose closest finger
    finger_assignments = {}
    for face in free_faces:
        face_ind = OBJ_FACES_INFO[ground_face]["adjacent_faces"].index(face)
        if face_ind in [2,3]:
            # Check y_ind column for finger that is furthest away
            if OBJ_FACES_INFO[face]["center_param"][x_ind] < 0: 
                # Want most negative value
                f_i = np.nanargmin(xy_distances[:,1])
            else:
                # Want most positive value
                f_i = np.nanargmax(xy_distances[:,1])
        else:
            # Check x_ind column for finger that is furthest away
            if OBJ_FACES_INFO[face]["center_param"][y_ind] < 0: 
                f_i = np.nanargmin(xy_distances[:,0])
            else:
                f_i = np.nanargmax(xy_distances[:,0])
        finger_assignments[face] = f_i
        xy_distances[f_i, :] = np.nan

    # Set contact point params for two long faces
    cp_params = [None, None, None]
    height_param = -0.65 # Always want cps to be at this height
    for face, finger_id in finger_assignments.items():
        param = OBJ_FACES_INFO[face]["center_param"].copy()
        param += OBJ_FACES_INFO[OBJ_FACES_INFO[ground_face]["opposite_face"]]["center_param"] * height_param
        cp_params[finger_id] = param

    # Assign remaining finger to either face 1 or 2 (short face)
    # TODO: or, no face at all (which means need to modify fixed_cp_traj_opt)
    # TODO Hardcoded right now
    #cp_params[0] = OBJ_FACES_INFO[2]["center_param"].copy()
    return cp_params

def get_pre_grasp_ft_goal(obj_pose, fingertips_current_wf, cp_params):
    ft_goal = np.zeros(9)
    incr = 0.03

    # Get list of desired fingertip positions
    cp_wf_list = get_cp_pos_wf_from_cp_params(cp_params, obj_pose.position, obj_pose.orientation)

    for f_i in range(3):
        f_wf = cp_wf_list[f_i]
        if cp_params[f_i] is None:
            f_new_wf = fingertips_current_wf[f_i]
        else:
            # Get face that finger is on
            face = get_face_from_cp_param(cp_params[f_i])
            f_of = get_of_from_wf(f_wf, obj_pose)

            # Release object
            f_new_of = f_of - incr * OBJ_FACES_INFO[face]["up_axis"]

            # Convert back to wf
            f_new_wf = get_wf_from_of(f_new_of, obj_pose)

        ft_goal[3*f_i:3*f_i+3] = f_new_wf
    return ft_goal

"""
Set up traj opt for fingers and static object
"""
def define_static_object_opt(nGrid, dt):
    problem = StaticObjectOpt(
                 nGrid     = nGrid,
                 dt        = dt,
                 obj_shape = OBJ_SIZE,
                 )
    return problem

"""
Solve traj opt to get finger waypoints
"""
def get_finger_waypoints(nlp, ft_goal, q_cur, obj_pose, npz_filepath = None):
    nlp.solve_nlp(ft_goal, q_cur, obj_pose = obj_pose, npz_filepath = npz_filepath)
    ft_pos = nlp.ft_pos_soln
    ft_vel = nlp.ft_vel_soln
    return ft_pos, ft_vel

##############################################################################
# Flip mode functions
##############################################################################

"""
Determine face that is closest to ground
"""
def get_closest_ground_face(obj_pose):
    min_z = np.inf
    min_face = None
    for i in range(1,7):
        c = OBJ_FACES_INFO[i]["center_param"].copy()
        c_wf = get_wf_from_of(c, obj_pose)
        if c_wf[2] < min_z:
            min_z = c_wf[2]
            min_face = i

    return min_face

"""
Get flipping contact points
"""
def get_flipping_cp_params(
                           init_pose,
                           goal_pose,
                          ):
    # Get goal face
    init_face = get_closest_ground_face(init_pose)
    #print("Init face: {}".format(init_face))
    # Get goal face
    goal_face = get_closest_ground_face(goal_pose)
    #print("Goal face: {}".format(goal_face))
    
    if goal_face not in OBJ_FACES_INFO[init_face]["adjacent_faces"]:
        #print("Goal face not adjacent to initial face")
        goal_face = OBJ_FACES_INFO[init_face]["adjacent_faces"][0]
        #print("Intermmediate goal face: {}".format(goal_face))

    # Common adjacent faces to init_face and goal_face
    common_adjacent_faces = list(set(OBJ_FACES_INFO[init_face]["adjacent_faces"]). intersection(OBJ_FACES_INFO[goal_face]["adjacent_faces"]))

    opposite_goal_face = OBJ_FACES_INFO[goal_face]["opposite_face"]
    
    #print("place fingers on faces {}, towards face {}".format(common_adjacent_faces, opposite_goal_face))

    # Find closest fingers to each of the common_adjacent_faces
    # Transform finger tip positions to object frame
    finger_base_of = []
    for f_wf in FINGER_BASE_POSITIONS:
        f_of = get_of_from_wf(f_wf, init_pose)
        #f_of = np.squeeze(get_of_from_wf(f_wf, init_pose))
        finger_base_of.append(f_of)

    # Object frame axis corresponding to plane parallel to ground plane
    x_ind, y_ind = __get_parallel_ground_plane_xy(init_face)
    # Find distance from x axis and y axis, and store in xy_distances
    x_axis = np.array([1,0])
    y_axis = np.array([0,1])
        
    xy_distances = np.zeros((3, 2)) # Row corresponds to a finger, columns are x and y axis distances
    for f_i, f_of in enumerate(finger_base_of):
        point_in_plane = np.array([f_of[0,x_ind], f_of[0,y_ind]]) # Ignore dimension of point that's not in the plane
        x_dist = __get_distance_from_pt_2_line(x_axis, np.array([0,0]), point_in_plane)
        y_dist = __get_distance_from_pt_2_line(y_axis, np.array([0,0]), point_in_plane)
        
        xy_distances[f_i, 0] = np.sign(f_of[0,y_ind]) * x_dist
        xy_distances[f_i, 1] = np.sign(f_of[0,x_ind]) * y_dist

    finger_assignments = {}
    for face in common_adjacent_faces:
        face_ind = OBJ_FACES_INFO[init_face]["adjacent_faces"].index(face)
        if face_ind in [2,3]:
            # Check y_ind column for finger that is furthest away
            if OBJ_FACES_INFO[face]["center_param"][x_ind] < 0: 
                # Want most negative value
                f_i = np.nanargmin(xy_distances[:,1])
            else:
                # Want most positive value
                f_i = np.nanargmax(xy_distances[:,1])
        else:
            # Check x_ind column for finger that is furthest away
            if OBJ_FACES_INFO[face]["center_param"][y_ind] < 0: 
                f_i = np.nanargmin(xy_distances[:,0])
            else:
                f_i = np.nanargmax(xy_distances[:,0])
        finger_assignments[face] = f_i
        xy_distances[f_i, :] = np.nan

    cp_params = [None, None, None]
    # TODO Hardcoded
    height_param = -0.65 # Always want cps to be at this height
    width_param = 0.65 
    for face in common_adjacent_faces:
        param = OBJ_FACES_INFO[face]["center_param"].copy()
        param += OBJ_FACES_INFO[OBJ_FACES_INFO[init_face]["opposite_face"]]["center_param"] * height_param
        param += OBJ_FACES_INFO[opposite_goal_face]["center_param"] * width_param
        cp_params[finger_assignments[face]] = param
        #cp_params.append(param)
    #print("Assignments: {}".format(finger_assignments))
    return cp_params, init_face, goal_face

##############################################################################
# Private functions
##############################################################################

"""
Given a ground face id, get the axes that are parallel to the floor
"""
def __get_parallel_ground_plane_xy(ground_face):
    if ground_face in [1,2]:
        x_ind = 0
        y_ind = 2
    if ground_face in [3,5]:
        x_ind = 2
        y_ind = 1
    if ground_face in [4,6]:
        x_ind = 0
        y_ind = 1
    return x_ind, y_ind

"""
Get distance from point to line (in 2D)
Inputs:
a, b: points on line
p: standalone point, for which we want to compute its distance to line
"""
def __get_distance_from_pt_2_line(a, b, p):
    a = np.squeeze(a)
    b = np.squeeze(b)
    p = np.squeeze(p)

    ba = b - a
    ap = a - p
    c = ba * (np.dot(ap,ba) / np.dot(ba,ba))
    d = ap - c
    
    return np.sqrt(np.dot(d,d))

