from time import time
from warnings import WarningMessage
import numpy as np
import os
# os.environ['CUDA_LAUNCH_BLOCKING']='1'

import torch, torchvision
from torch import Tensor
from typing import Tuple, Dict

from tqdm import tqdm
import matplotlib.pyplot as plt

LEG_DOF=12

class HardwareCfg():
    class init_state():
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

    class env:
        num_envs = 1
        num_actions = 13
        num_dummy_dof = 1
        
        n_scan = 132
        n_priv = 3+3 +3
        n_priv_latent = 4 + 1 + 14 +14
        n_proprio = 3 + 2 + 1 + 5  + 13*3 + 4
        history_len = 10

        num_observations = n_proprio + n_scan + history_len*n_proprio + n_priv_latent + n_priv #n_scan + n_proprio + n_priv #187 + 47 + 5 + 12 
        num_privileged_obs = None
        next_goal_threshold = 0.1

        episode_length_s = 20 # episode length in seconds

    class control:
        # PD Drive parameters:
        control_type = 'P'
        # stiffness = {'joint': 40.}  # [N*m/rad]
        # damping = {'joint': 0.5}     # [N*m*s/rad]
        stiffness = {'joint': 30.}  # [N*m/rad] 30
        damping = {'joint': 1.}     # [N*m*s/rad] 0.6
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        
    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 1.2

    class commands:
        num_commands = 5 # default: lin_vel_x, lin_vel_y, yaw, pitch, gripper close
        class max_ranges:
            lin_vel_x = [0., 1.5] # min max [m/s]
            lin_vel_y = [-0.5, 0.5]   # min max [m/s]
            yaw = [-1, 1]    # min max [rad]
            pitch = [-0.7, 0.7]  # min max [rad]
            
class Hardware():
    def __init__(self):
        self.cfg = HardwareCfg()
        self.num_envs = 1 #cfg.env.num_envs
        self.num_obs = self.cfg.env.num_observations
        self.num_privileged_obs = self.cfg.env.num_privileged_obs
        self.num_actions = self.cfg.env.num_actions
        self.obs_scales = self.cfg.normalization.obs_scales
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # prepare action deployment joint positions offsets and PD gains
        # urdf_dof_names =  ['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint', 'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',\
        #                     'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint', 'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint']
        self.dof_names =  ['FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint', 'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',\
                            'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint', 'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint']        
        self.p_gains = torch.zeros(LEG_DOF, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(LEG_DOF, dtype=torch.float, device=self.device, requires_grad=False)
        self.default_dof_pos_all = torch.zeros(1, LEG_DOF, dtype=torch.float, device=self.device, requires_grad=False)
        self.default_dof_pos = torch.zeros(LEG_DOF, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(LEG_DOF):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        self.default_dof_pos_all[:] = self.default_dof_pos[0]
        self.default_dof_pos_np = self.default_dof_pos_all[0].cpu().numpy()

        # prepare osbervations buffer
        self.obs_buf_dim = self.cfg.env.n_proprio+ self.cfg.env.n_scan + self.cfg.env.n_priv + self.cfg.env.n_priv_latent+ self.cfg.env.history_len*self.cfg.env.n_proprio 
        self.obs_buf = torch.zeros(self.num_envs, self.obs_buf_dim, dtype=torch.float, device=self.device, requires_grad=False)
        self.scan_buf = torch.zeros(self.num_envs, self.cfg.env.n_scan, dtype=torch.float, device=self.device, requires_grad=False)
        self.priv_explicit = torch.zeros(self.num_envs, self.cfg.env.n_priv, dtype=torch.float, device=self.device, requires_grad=False)
        self.priv_latent = torch.zeros(self.num_envs, self.cfg.env.n_priv_latent, dtype=torch.float, device=self.device, requires_grad=False)
        self.obs_history_buf = torch.zeros(self.num_envs, self.cfg.env.history_len, self.cfg.env.n_proprio, dtype=torch.float, device=self.device, requires_grad=False)
        self.obs_init = True

    def compute_angle(self,actions):
        # action output
        clip_lowlevel_actions = self.cfg.normalization.clip_actions / self.cfg.control.action_scale
        lowlevel_actions = torch.clip(actions, -clip_lowlevel_actions, clip_lowlevel_actions).to(self.device)
        lowlevel_actions[:,-1] = torch.clip(lowlevel_actions[:,-1], -1, 1).to(self.device)

        # TODO: joint pos limits? torque limits?
        actions_scaled = lowlevel_actions[:,:-1] * self.cfg.control.action_scale + self.default_dof_pos_all
        return actions_scaled
    
        # torques = self.p_gains*(actions_scaled + self.default_dof_pos_all - self.dof_pos) - self.d_gains*self.dof_vel
        # output_torques = torch.clip(torques, -self.torque_limits, self.torque_limits)
    
    def compute_observations(self, obs_proprio):
        lowlevel_obs_buf = torch.tensor(obs_proprio,dtype=torch.float, device=self.device).unsqueeze(0)
        self.obs_buf = torch.cat([lowlevel_obs_buf, self.scan_buf, self.priv_explicit, self.priv_latent, self.obs_history_buf.view(self.num_envs, -1)], dim=-1)
        # if self.obs_init:
        #     self.obs_history_buf = torch.stack([lowlevel_obs_buf] * self.cfg.env.history_len, dim=1)
        # else:
        #     self.obs_history_buf = torch.cat([self.obs_history_buf[:, 1:], lowlevel_obs_buf.unsqueeze(1)], dim=1)
        clip_obs = self.cfg.normalization.clip_observations
        obs = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        return obs
        # obs_jit = torch.cat((obs.detach()[:, :env_cfg.env.n_proprio+env_cfg.env.n_priv], obs.detach()[:, -env_cfg.env.history_len*env_cfg.env.n_proprio:]), dim=1)
