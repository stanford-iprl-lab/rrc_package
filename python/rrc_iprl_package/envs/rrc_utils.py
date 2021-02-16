import gym
import numpy as np
from gym import wrappers
import functools
from gym.envs.registration import register
from scipy.spatial.transform import Rotation

# SET THIS TO DETERMINE RRC_UTILS IMPORTS
phase = 2

if phase == 1:
    from rrc_simulation.gym_wrapper.envs import cube_env, custom_env
elif phase == 2:
    from rrc_iprl_package.envs import cube_env
    from rrc_iprl_package.envs import custom_env
    from rrc_iprl_package.envs import env_wrappers, initializers


registered_envs = [spec.id for spec in gym.envs.registry.all()]

FRAMESKIP = 10
EPLEN = 15 * 1000 // FRAMESKIP  # 15 seconds
EPLEN_SHORT = 5 * 1000 // FRAMESKIP  # 5 seconds, 500 total timesteps

if phase == 1:
    if "real_robot_challenge_phase_1-v2" not in registered_envs:
        register(
            id="real_robot_challenge_phase_1-v2",
            entry_point=custom_env.PushCubeEnv
            )
    if "real_robot_challenge_phase_1-v3" not in registered_envs:
        register(
            id="real_robot_challenge_phase_1-v3",
            entry_point=custom_env.PushReorientCubeEnv
            )
    if "real_robot_challenge_phase_1-v4" not in registered_envs:
        register(
            id="real_robot_challenge_phase_1-v4",
            entry_point=custom_env.SparseCubeEnv
            )
elif phase == 2:
    if "real_robot_challenge_phase_2-v1" not in registered_envs:
        register(
            id="real_robot_challenge_phase_2-v1",
            entry_point=cube_env.RealRobotCubeEnv
            )
    if "real_robot_challenge_phase_2-v2" not in registered_envs:
        register(
            id="real_robot_challenge_phase_2-v2",
            entry_point=cube_env.PushCubeEnv
            )

total_steps = 5e6
step_rates = np.linspace(0, 0.6, 10)

def success_rate_early_stopping(steps, success_rate):
    return steps/total_steps > 0.25 and step_rates[min(9, int(steps/total_steps * 10))] > success_rate

def make_env_fn(env_str, wrapper_params=[], **make_kwargs):
    """Returns env_fn to pass to spinningup alg"""

    def env_fn(visualization=False, **mod_kwargs):
        make_kwargs.update(mod_kwargs)
        env = gym.make(env_str, visualization=visualization, **make_kwargs)
        for w in wrapper_params:
            if isinstance(w, dict):
                env = w['cls'](env, *w.get('args', []), **w.get('kwargs', {}))
            else:
                env = w(env)
        return env
    return env_fn

def build_env_fn(difficulty=1,ep_len=EPLEN, frameskip=FRAMESKIP,
                 action_type='pos', rew_fn='sigmoid', goal_env=False,
                 dist_thresh=0.02, ori_thresh=np.pi/6,
                 pos_coef=.1, ori_coef=.1, fingertip_coef=0, ac_norm_pen=0.,
                 scaled_ac=False, sa_relative=False, lim_pen=0.,
                 task_space=False, ts_relative=False,
                 goal_relative=True, keep_goal=False, use_quat=False,
                 residual=False, res_torque=True,
                 framestack=1, sparse=False, initializer=None,
                 flatten=True, single_finger=False):
    if goal_env or residual:
        env_str = 'real_robot_challenge_phase_2-v1'
    else:
        env_str = 'real_robot_challenge_phase_2-v2'

    custom_env.DIST_THRESH = dist_thresh
    custom_env.ORI_THRESH = ori_thresh
    env_wrappers.DIST_THRESH = dist_thresh
    env_wrappers.ORI_THRESH = ori_thresh
    initializers.DIST_THRESH = dist_thresh
    if action_type == 'pos' and not residual:
        action_type = cube_env.ActionType.POSITION
    else:
        action_type = cube_env.ActionType.TORQUE

    final_wrappers = []
    if single_finger:
        final_wrappers = [env_wrappers.SingleFingerWrapper]
    else:
        # Action wrappers (scaled actions, task space, relative goal)
        if residual:
            final_wrappers.append(functools.partial(custom_env.ResidualPolicyWrapper,
                                  rl_torque=res_torque))
        else:
            if scaled_ac:
                final_wrappers.append(
                    functools.partial(env_wrappers.ScaledActionWrapper,
                                      goal_env=goal_env, relative=sa_relative,
                                      lim_penalty=lim_pen))
            if goal_relative and not goal_env:
                final_wrappers.append(functools.partial(
                    env_wrappers.RelativeGoalWrapper, keep_goal=keep_goal,
                    use_quat=use_quat))

    # Adds time limit, logging, action clipping, and flattens observation 
    final_wrappers.append(functools.partial(wrappers.TimeLimit,
            max_episode_steps=ep_len))

    p2_info_keys = ['is_success', 'is_success_ori', 'final_dist', 'final_score',
                    'final_ori_dist', 'final_ori_scaled']
    p2_log_info_wrapper = functools.partial(env_wrappers.LogInfoWrapper,
                                            info_keys=p2_info_keys)
    if ((action_type == cube_env.ActionType.TORQUE
                and (not residual or res_torque))
            or (scaled_ac and not sa_relative)
            or not scaled_ac):
        final_wrappers.append(
                functools.partial(wrappers.RescaleAction, a=-1, b=1))
    final_wrappers +=  [p2_log_info_wrapper, wrappers.ClipAction]

    if flatten:  # set to false to debug with obs dict
        if not goal_env:
            final_wrappers.append(wrappers.FlattenObservation)
        else:
            final_wrappers.append(env_wrappers.FlattenGoalWrapper)

    if framestack > 1:
        final_wrappers.append(functools.partial(wrappers.FrameStack,
                                                num_stack=framestack))
        if flatten:
            final_wrappers.append(wrappers.FlattenObservation)

    ewrappers = []
    if task_space:
        assert not scaled_ac, 'Can only use TaskSpaceWrapper OR ' \
                              'ScaledActionWrapper'
        ewrappers.append(functools.partial(env_wrappers.TaskSpaceWrapper,
                                           relative=ts_relative))
        action_type = cube_env.ActionType.POSITION
    ewrappers += final_wrappers

    if initializer == 'random':
        initializer = initializers.RandomInitializer(difficulty=difficulty)
    elif initializer == 'fixed':
        init_pose = initializers.FixedInitializer.def_initial_pose
        goal_pose = initializers.FixedInitializer.def_goal_pose
        ori = Rotation.from_quat(init_pose.orientation).as_euler('xyz')
        ori += np.array([0,0,np.pi/2])
        ori = Rotation.from_euler('xyz', ori).as_quat()

        init_pose.position += np.array([0,0.05,0])
        init_pose.orientation = ori
        goal_pose.orientation = ori
        initializer = initializers.FixedInitializer(
		    difficulty, initial_state=init_pose, goal=goal_pose)
    elif initializer == 'reorient':
        initializer = initializers.ReorientInitializer(difficulty, 0.09)
    elif initializer == 'curriculum':
        initializer = initializers.CurriculumInitializer(difficulty)
    else:
        initializer = initializers.RandomGoalOrientationInitializer(difficulty)

    if goal_env or residual:
        ret = make_env_fn(env_str, ewrappers,
                          initializer=initializer,
                          action_type=action_type,
                          frameskip=frameskip, goal_difficulty=difficulty,
                          sparse=sparse)
    else:
        if sparse:
            rew_fn = 'sparse'
        ret = make_env_fn(env_str, ewrappers,
                          initializer=initializer,
                          action_type=action_type,
                          frameskip=frameskip,
                          pos_coef=pos_coef,
                          ori_coef=ori_coef,
                          fingertip_coef=fingertip_coef,
                          ac_norm_pen=ac_norm_pen,
                          rew_fn=rew_fn
                          )

    return ret



if phase == 1:
    push_random_initializer = cube_env.RandomInitializer(difficulty=1)

    fixed_reorient_initializer = custom_env.RandomGoalOrientationInitializer()

    push_curr_initializer = custom_env.CurriculumInitializer(initial_dist=0.,
                                                             num_levels=5)
    push_fixed_initializer = custom_env.CurriculumInitializer(initial_dist=0.,
                                                              num_levels=2)
    reorient_initializer = reorient_curr_initializer = custom_env.CurriculumInitializer(
            initial_dist=0.06, num_levels=3, difficulty=4,
            fixed_goal=custom_env.RandomOrientationInitializer.goal)
    recenter_initializer = custom_env.ReorientInitializer(1, 0.09)

    push_initializer = push_fixed_initializer

    lift_initializer = cube_env.RandomInitializer(difficulty=2)
    ori_initializer = cube_env.RandomInitializer(difficulty=3)
# Val in info string calls logger.log_tabular() with_min_and_max to False
    push_info_kwargs = {'is_success': 'SuccessRateVal', 'final_dist': 'FinalDist',
        'final_score': 'FinalScore', 'init_sample_radius': 'InitSampleDistVal',
        'goal_sample_radius': 'GoalSampleDistVal'}
    reorient_info_kwargs = {'is_success': 'SuccessRateVal',
            'is_success_ori': 'OriSuccessRateVal',
            'final_dist': 'FinalDist', 'final_ori_dist': 'FinalOriDist',
            'final_ori_scaled': 'FinalOriScaledDist',
            'final_score': 'FinalScore'}

    info_keys = ['is_success', 'is_success_ori', 'final_ori_dist', 'final_dist',
                 'final_score']
    curr_info_keys = info_keys + ['goal_sample_radius', 'init_sample_radius']
    reorient_info_keys = ['is_success', 'is_success_ori', 'final_dist', 'final_score',
                          'final_ori_dist', 'final_ori_scaled']
    action_type = cube_env.ActionType.POSITION


    log_info_wrapper = functools.partial(custom_env.LogInfoWrapper,
                                         info_keys=info_keys)
    reorient_log_info_wrapper = functools.partial(custom_env.LogInfoWrapper,
                                                  info_keys=reorient_info_keys)

    final_wrappers = [functools.partial(wrappers.TimeLimit, max_episode_steps=EPLEN),
                      log_info_wrapper,
                      wrappers.ClipAction, wrappers.FlattenObservation]
    final_wrappers = final_wrappers_short = [
           functools.partial(wrappers.TimeLimit, max_episode_steps=EPLEN_SHORT),
           log_info_wrapper,
           wrappers.FlattenObservation]

    final_wrappers_reorient = [
            functools.partial(wrappers.TimeLimit, max_episode_steps=EPLEN_SHORT),
            reorient_log_info_wrapper,
            wrappers.FlattenObservation]

    final_wrappers_reorient_abs = [
            functools.partial(wrappers.TimeLimit, max_episode_steps=EPLEN_SHORT),
            reorient_log_info_wrapper,
            wrappers.FlattenObservation]

    final_wrappers_relgoal = [
            functools.partial(custom_env.RelativeGoalWrapper,
                keep_goal=False),
            functools.partial(wrappers.TimeLimit, max_episode_steps=EPLEN_SHORT),
            reorient_log_info_wrapper,
            wrappers.FlattenObservation]

    final_wrappers_vds = [functools.partial(wrappers.TimeLimit, max_episode_steps=EPLEN),
            custom_env.FlattenGoalWrapper]

    abs_scaled_wrapper = functools.partial(custom_env.ScaledActionWrapper,
                                           goal_env=False, relative=False)
    rel_scaled_wrapper = functools.partial(custom_env.ScaledActionWrapper,
                                           goal_env=False, relative=True)

    abs_task_wrapper = functools.partial(custom_env.TaskSpaceWrapper, relative=False)
    rel_task_wrapper = functools.partial(custom_env.TaskSpaceWrapper, relative=True)
    rew_wrappers_step = [functools.partial(custom_env.CubeRewardWrapper,
                                           pos_coef=1., ori_coef=1.,
                                           ac_norm_pen=0.2, fingertip_coef=1.,
                                           rew_fn='exp', augment_reward=True),
                         custom_env.StepRewardWrapper,
                         functools.partial(custom_env.ReorientWrapper,
                                           goal_env=False, dist_thresh=0.075,
                                           ori_thresh=np.pi/6)]
    rew_wrappers = [functools.partial(custom_env.CubeRewardWrapper,
                                      pos_coef=.1, ori_coef=.1,
                                      ac_norm_pen=.1, fingertip_coef=.1,
                                      rew_fn='exp', augment_reward=True),
                    functools.partial(custom_env.ReorientWrapper,
                                      goal_env=False, dist_thresh=0.075,
                                      ori_thresh=np.pi)]

    recenter_rew_wrappers = [functools.partial(custom_env.CubeRewardWrapper, pos_coef=1.,
                                 ori_coef=.5, ac_norm_pen=0., augment_reward=True, rew_fn='exp'),
                             functools.partial(custom_env.ReorientWrapper, goal_env=False,
                                 dist_thresh=0.05,
                                 ori_thresh=np.pi),
                             rel_scaled_wrapper]

    rrc_wrappers = [rel_scaled_wrapper] + final_wrappers
    rrc_vds_wrappers = [rel_scaled_wrapper] + final_wrappers_vds

    push_wrappers = [functools.partial(custom_env.CubeRewardWrapper, pos_coef=1.,
                              ac_norm_pen=0.2, rew_fn='exp'),
                     rel_scaled_wrapper]
    push_wrappers = push_wrappers + final_wrappers

    recenter_wrappers_rel = recenter_rew_wrappers + final_wrappers_reorient
    reorient_wrappers_relgoal = recenter_rew_wrappers + final_wrappers_relgoal
    reorient_wrappers_relgoaltask = [rel_task_wrapper] + rew_wrappers + final_wrappers_relgoal


    recenter_wrappers_abs = recenter_rew_wrappers + final_wrappers_reorient

    abs_task_wrappers =  [abs_task_wrapper] + rew_wrappers + final_wrappers_reorient + [wrappers.ClipAction]
    rel_task_wrappers =  [rel_task_wrapper] + rew_wrappers + final_wrappers_reorient + [wrappers.ClipAction]

    abs_task_step_wrappers =  [abs_task_wrapper] + rew_wrappers_step + final_wrappers_reorient + [wrappers.ClipAction]
    rel_task_step_wrappers =  [rel_task_wrapper] + rew_wrappers_step + final_wrappers_reorient + [wrappers.ClipAction]

    rrc_env_str = 'rrc_simulation.gym_wrapper:real_robot_challenge_phase_1-v1'
    rrc_env_fn = make_env_fn(rrc_env_str, rrc_wrappers,
                                 initializer=push_initializer,
                                 action_type=action_type,
                                 frameskip=FRAMESKIP)

    rrc_vds_env_fn = make_env_fn(rrc_env_str, rrc_vds_wrappers,
                                 initializer=push_initializer,
                                 action_type=action_type,
                                 frameskip=FRAMESKIP)


    push_env_str = 'real_robot_challenge_phase_1-v2'
    push_env_fn = make_env_fn(push_env_str, push_wrappers,
                                  initializer=push_initializer,
                                  action_type=action_type,
                                  frameskip=FRAMESKIP)

    reorient_env_str = 'real_robot_challenge_phase_1-v3'
    abs_task_env_fn = make_env_fn(reorient_env_str, abs_task_wrappers,
                                  initializer=reorient_initializer,
                                  action_type=cube_env.ActionType.TORQUE,
                                  frameskip=FRAMESKIP)
    rel_task_env_fn = make_env_fn(reorient_env_str, rel_task_wrappers,
                                  initializer=reorient_initializer,
                                  action_type=cube_env.ActionType.TORQUE,
                                  frameskip=FRAMESKIP)


    abs_task_step_env_fn = make_env_fn(reorient_env_str, abs_task_step_wrappers,
                                  initializer=reorient_initializer,
                                  action_type=cube_env.ActionType.TORQUE,
                                  frameskip=FRAMESKIP)
    rel_task_step_env_fn = make_env_fn(reorient_env_str, rel_task_step_wrappers,
                                  initializer=reorient_initializer,
                                  action_type=cube_env.ActionType.TORQUE,
                                  frameskip=FRAMESKIP)

    recenter_rel_env_fn = make_env_fn(reorient_env_str, recenter_wrappers_rel,
                                  initializer=recenter_initializer,
                                  action_type=action_type,
                                  frameskip=FRAMESKIP)

    recenter_env_fn = make_env_fn(reorient_env_str, recenter_wrappers_abs,
                                  initializer=recenter_initializer,
                                  action_type=action_type,
                                  frameskip=FRAMESKIP)

    reorient_env_fn = make_env_fn(reorient_env_str, reorient_wrappers_relgoal,
                                  initializer=fixed_reorient_initializer,
                                  action_type=action_type,
                                  frameskip=FRAMESKIP)

    reorient_task_env_fn = make_env_fn(reorient_env_str, reorient_wrappers_relgoaltask,
                                  initializer=fixed_reorient_initializer,
                                  action_type=cube_env.ActionType.TORQUE,
                                  frameskip=FRAMESKIP)

    eval_keys = ['is_success', 'is_success_ori', 'final_ori_dist', 'final_dist',
                 'final_score']


# PHASE 2
if phase == 2:
    # ENV STR
    p2_goalenv_str = "real_robot_challenge_phase_2-v0"
    p2_env_str = "real_robot_challenge_phase_2-v2"

    # INITIALIZERS
    p2_fixed_reorient = initializers.RandomGoalOrientationInitializer()
    p2_push_curr = initializers.CurriculumInitializer(initial_dist=0.,
                                                      num_levels=5)
    p2_push_fixed = initializers.CurriculumInitializer(initial_dist=0.,
                                                       num_levels=2)
    p2_reorient_curr = initializers.CurriculumInitializer(
            initial_dist=0.06, num_levels=3, difficulty=4,
            fixed_goal=initializers.RandomOrientationInitializer.def_goal_pose)
    p2_recenter = initializers.ReorientInitializer(1, 0.09)

    p2_info_keys = ['is_success', 'is_success_ori', 'final_dist', 'final_score',
                    'final_ori_dist', 'final_ori_scaled']
    p2_info_kwargs = {'is_success': 'SuccessRateVal',
            'is_success_ori': 'OriSuccessRateVal',
            'final_dist': 'FinalDist', 'final_ori_dist': 'FinalOriDist',
            'final_ori_scaled': 'FinalOriScaledDist'}


    p2_log_info_wrapper = functools.partial(env_wrappers.LogInfoWrapper,
                                            info_keys=p2_info_keys)

    p2_final_wrappers = [functools.partial(wrappers.TimeLimit, max_episode_steps=EPLEN),
                      p2_log_info_wrapper,
                      wrappers.ClipAction, wrappers.FlattenObservation]
    p2_final_wrappers_reorient = [
            functools.partial(wrappers.TimeLimit, max_episode_steps=EPLEN_SHORT),
            p2_log_info_wrapper,
            wrappers.FlattenObservation]

    p2_final_wrappers_relgoal = [functools.partial(env_wrappers.RelativeGoalWrapper,
                keep_goal=False)] + p2_final_wrappers_reorient


    p2_rew_wrappers = [functools.partial(env_wrappers.CubeRewardWrapper,
                                      pos_coef=.1, ori_coef=.1,
                                      ac_norm_pen=.1, fingertip_coef=.1,
                                      rew_fn='exp', augment_reward=True),
                    functools.partial(env_wrappers.ReorientWrapper,
                                      goal_env=False, dist_thresh=0.075,
                                      ori_thresh=np.pi)]

    p2_rel_scaled_wrapper = functools.partial(env_wrappers.ScaledActionWrapper,
                                           goal_env=False, relative=True)

    p2_recenter_rew_wrappers = [functools.partial(env_wrappers.CubeRewardWrapper,
                                    pos_coef=1.,  ori_coef=.5, ac_norm_pen=0.,
                                    augment_reward=True, rew_fn='exp'),
                                functools.partial(env_wrappers.ReorientWrapper,
                                     goal_env=False, dist_thresh=0.05,
                                     ori_thresh=0.2),
                                p2_rel_scaled_wrapper]

    p2_rrc_wrappers = [p2_rel_scaled_wrapper] + p2_final_wrappers_relgoal

    p2_reorient_env_fn = make_env_fn(
            p2_env_str,
            p2_rrc_wrappers,
            initializer=p2_fixed_reorient,
            action_type=cube_env.ActionType.POSITION,
            frameskip=FRAMESKIP)

