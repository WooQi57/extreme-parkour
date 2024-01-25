# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class Go1GBRoughCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]

            'l_finger_joint': 0.0,    # [m]
            'r_finger_joint': 0.0,    # [m]
        }

    class init_state_slope( LeggedRobotCfg.init_state ):
        pos = [0.56, 0.0, 0.24] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.03,   # [rad]
            'RL_hip_joint': 0.03,   # [rad]
            'FR_hip_joint': -0.03,  # [rad]
            'RR_hip_joint': -0.03,   # [rad]

            'FL_thigh_joint': 1.0,     # [rad]
            'RL_thigh_joint': 1.9,   # [rad]1.8
            'FR_thigh_joint': 1.0,     # [rad]
            'RR_thigh_joint': 1.9,   # [rad]

            'FL_calf_joint': -2.2,   # [rad]
            'RL_calf_joint': -0.9,    # [rad]
            'FR_calf_joint': -2.2,  # [rad]
            'RR_calf_joint': -0.9,    # [rad]

            'l_finger_joint': 0.0,    # [m]
            'r_finger_joint': 0.0,    # [m]
        }
    class env(LeggedRobotCfg.env):
        num_envs = 3072 # 6144
        num_actions = 5
        # num_lowlevel_actions = 13
        num_dummy_dof = 1
        
        n_scan = 132
        n_priv = 3+3 +3
        n_priv_latent = 4 + 1 + 14 +14
        n_proprio = 6+5 #3 + 2 + 2 + 4 + 2 + 13*3 + 4
        history_len = 10

        num_observations = n_proprio + n_scan + history_len*n_proprio + n_priv_latent + n_priv #n_scan + n_proprio + n_priv #187 + 47 + 5 + 12 

        next_goal_threshold = 0.1

        episode_length_s = 10 # episode length in seconds 20


    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        # stiffness = {'joint': 40.}  # [N*m/rad]
        # damping = {'joint': 0.5}     # [N*m*s/rad]
        stiffness = {'joint': 30.}  # [N*m/rad]
        damping = {'joint': 0.6}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go1/urdf/go1_new.urdf'
        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go1g/urdf/go1_gripper.urdf'
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2g_description_v4.urdf'
        foot_name = "foot"
        finger_name = "finger"
        penalize_contacts_on = ["thigh", "calf", "finger", "gripper"]
        terminate_after_contacts_on = ["base"]#, "thigh", "calf"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter

        gripper_pitch_offset = 0

        collapse_fixed_joints = False # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">


    class rewards( LeggedRobotCfg.rewards ):
        class scales:
            # tracking rewards
            # tracking_goal_vel = 1.5
            tracking_goal_pos = 1.5
            # tracking_yaw = 0.5  # 0.5
            # tracking_pitch = 0.5  # 0.5
            # tracking_gripper = 0.5
            # tracking_gripper_open = 0.5  # 0.5 for approach else 0.0
            grasp = 1
            tracking_goal_pos_frac = 0.15  # 0.05

            # pickup rewards no applicable for approaching
            pickup_box = 6 # 3
            box_height = 10 # 2 10

            # regularization rewards
            fit_truth = 2.5  # 2.5 for approach else 0.5 fit ground truth data: keep yaw and pitch reasonable
            between_fingers = 0.05  # 0.05# lin_vel_z = -1.0
            hold_still = 10
            # ang_vel_xy = -0.05
            # orientation = -1.
            # dof_acc = -2.5e-7
            collision = -1.*0
            # action_rate = -0.1
            # delta_torques = -1.0e-7
            # torques = -0.00001
            # hip_pos = -0.5
            # dof_error = -0.04
            # feet_stumble = -1
            # feet_edge = -1

        tracking_sigma = 0.2 # tracking reward = exp(-error^2/sigma)  0.2
        pick_sigma = 0.075
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25
        box_max_height = 0.25
        # class scales( LeggedRobotCfg.rewards.scales ):
            # torques = -0.0002
            # dof_pos_limits = -10.0

    class terrain( LeggedRobotCfg.terrain):
        terrain_dict = {"smooth slope": 0., 
                "rough slope up": 0.0,
                "rough slope down": 0.0,
                "rough stairs up": 0., 
                "rough stairs down": 0., 
                "discrete": 0., 
                "stepping stones": 0.0,
                "gaps": 0., 
                "smooth flat": 0,
                "pit": 0.0,
                "wall": 0.0,
                "platform": 0.,
                "large stairs up": 0.,
                "large stairs down": 0.,
                "parkour": 0.0,
                "parkour_hurdle": 0.0,
                "parkour_flat": 0.0,
                "parkour_step": 1.0,
                "parkour_gap": 0.0,
                "demo": 0.0,}
        terrain_proportions = list(terrain_dict.values())
        y_range = [-0.4, 0.4]
        num_goals = 1

    class commands( LeggedRobotCfg.commands):
        resampling_time = 4. # time before command are changed[s]  # 6
        num_commands = 5 # default: lin_vel_x, lin_vel_y, yaw, pitch, gripper close
        class max_ranges:
            lin_vel_x = [0., 1.5] # min max [m/s]
            lin_vel_y = [-0.5, 0.5]   # min max [m/s]
            yaw = [-1, 1]    # min max [rad]
            pitch = [-0.7, 0.7]  # min max [rad]
            
            target_x = [0.5, 1.] # min max [m]  1 1.5
            target_y = [-0.25, 0.25]   # min max [m] -0.5 0.5
            target_z = [0, 0.6]    # min max [m]  higher
            
        lin_vel_clip = 0.2
        ang_clip = 0.05
        vx_max = 0.5

class Go1GBRoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01  # 0
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'rough_a1'
        resume = True
        max_iterations = 3000 # number of policy updates 50000

    class estimator( LeggedRobotCfgPPO.estimator):
        priv_states_dim = Go1GBRoughCfg.env.n_priv
        num_prop = Go1GBRoughCfg.env.n_proprio
  
