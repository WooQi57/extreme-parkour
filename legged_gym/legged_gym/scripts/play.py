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

from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import code

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
from isaacgym import gymtorch, gymapi, gymutil
import numpy as np
import torch
import cv2
from collections import deque
import statistics
import faulthandler
from copy import deepcopy
import matplotlib.pyplot as plt
from time import time, sleep
from legged_gym.utils import webviewer

def get_load_path(root, load_run=-1, checkpoint=-1, model_name_include="model"):
    if checkpoint==-1:
        models = [file for file in os.listdir(root) if model_name_include in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
        checkpoint = model.split("_")[-1].split(".")[0]
    return model, checkpoint

def play(args):
    if args.web:
        web_viewer = webviewer.WebViewer(output_video_file="../figs/output.mp4")
    faulthandler.enable()
    exptid = args.exptid
    log_pth = "../../logs/{}/".format(args.proj_name) + args.exptid

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    if args.nodelay:
        env_cfg.domain_rand.action_delay_view = 0
    env_cfg.env.num_envs = 1 if not args.save else 64  # 2
    env_cfg.env.episode_length_s = 30 # 60 30
    env_cfg.commands.resampling_time = 10 # 60 10
    env_cfg.terrain.num_rows = 2
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.height = [0.02, 0.02]
    env_cfg.terrain.terrain_dict = {"smooth slope": 0., 
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
                                    "parkour": 0.2*0,
                                    "parkour_hurdle": 0.2*0,
                                    "parkour_flat": 0.,
                                    "parkour_step": 1.0,
                                    "parkour_gap": 0.2*0, 
                                    "demo": 0.2*0}
    
    env_cfg.terrain.terrain_proportions = list(env_cfg.terrain.terrain_dict.values())
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.max_difficulty = True
    
    env_cfg.depth.angle = [0, 1]
    env_cfg.noise.add_noise = True
    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.push_interval_s = 6
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_base_com = False

    depth_latent_buffer = []
    # prepare environment
    env: LeggedRobot
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    # prepare plot data
    time_hist = []
    cmd_hist = []
    state_hist = []
    ref_hist = []
    finger_force_hist = []
    if args.web:
        web_viewer.setup(env)

    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg, log_pth = task_registry.make_alg_runner(log_root = log_pth, env=env, name=args.task, args=args, train_cfg=train_cfg, model_name_include="model", return_log_dir=True)
    
    if args.use_jit:
        path = os.path.join(log_pth, "traced")
        model, checkpoint = get_load_path(root=path, checkpoint=args.checkpoint)
        path = os.path.join(path, model)
        print("Loading jit for policy: ", path)
        policy_jit = torch.jit.load(path, map_location=env.device)
    else:
        policy = ppo_runner.get_inference_policy(device=env.device)
    estimator = ppo_runner.get_estimator_inference_policy(device=env.device)
    if env.cfg.depth.use_camera:
        depth_encoder = ppo_runner.get_depth_encoder_inference_policy(device=env.device)

    actions = torch.zeros(env.num_envs, 12, device=env.device, requires_grad=False)
    infos = {}
    infos["depth"] = env.depth_buffer.clone().to(ppo_runner.device)[:, -1] if ppo_runner.if_depth else None

    for i in range(3*int(env.max_episode_length)):
        if args.use_jit:
            if env.cfg.depth.use_camera:
                if infos["depth"] is not None:
                    depth_latent = torch.ones((env_cfg.env.num_envs, 32), device=env.device)
                    actions, depth_latent = policy_jit(obs.detach(), True, infos["depth"], depth_latent)
                else:
                    depth_buffer = torch.ones((env_cfg.env.num_envs, 58, 87), device=env.device)
                    actions, depth_latent = policy_jit(obs.detach(), False, depth_buffer, depth_latent)
            else:
                obs_jit = torch.cat((obs.detach()[:, :env_cfg.env.n_proprio+env_cfg.env.n_priv], obs.detach()[:, -env_cfg.env.history_len*env_cfg.env.n_proprio:]), dim=1)
                actions = policy(obs_jit)
        else:
            if env.cfg.depth.use_camera:
                if infos["depth"] is not None:
                    obs_student = obs[:, :env.cfg.env.n_proprio].clone()
                    obs_student[:, 6:8] = 0
                    depth_latent_and_yaw = depth_encoder(infos["depth"], obs_student)
                    depth_latent = depth_latent_and_yaw[:, :-2]
                    yaw = depth_latent_and_yaw[:, -2:]
                obs[:, 6:8] = 1.5*yaw
                    
            else:
                depth_latent = None
            
            if hasattr(ppo_runner.alg, "depth_actor"):
                actions = ppo_runner.alg.depth_actor(obs.detach(), hist_encoding=True, scandots_latent=depth_latent)
            else:
                actions = policy(obs.detach(), hist_encoding=True, scandots_latent=depth_latent)
        # print(actions[:,-1])
        obs, _, rews, dones, infos = env.step(actions.detach())
        if args.web:
            web_viewer.render(fetch_results=True,
                        step_graphics=True,
                        render_all_camera_sensors=True,
                        wait_for_page_load=True)
            web_viewer.write_vid()
        # store data for plot
        cur_time = env.episode_length_buf[env.lookat_id].item() / 50
        time_hist.append(cur_time)
        cmd_hist.append((env.target_position[env.lookat_id, :]).tolist())
        cur_state = env.ee_pos[env.lookat_id, :].tolist()
        cur_state.append(env.yaw[env.lookat_id].tolist())
        cur_state.append(env.pitch[env.lookat_id].tolist())
        state_hist.append(cur_state)
        ref = [env.target_yaw[env.lookat_id].tolist(),env.target_pitch[env.lookat_id].tolist()]
        ref_hist.append(ref)


        real_delta_yaw = env.target_yaw[env.lookat_id].tolist() - env.yaw[env.lookat_id].tolist()
        real_delta_pitch = env.target_pitch[env.lookat_id].tolist() - env.pitch[env.lookat_id].tolist()
        finger_force = torch.norm(env.contact_forces[env.lookat_id, env.finger_indices, :],dim=1).tolist()
        finger_force_hist.append(finger_force)
        print("----------\ntime:", cur_time, 
              "\nreal_delta_pos:", env.base_target_pos[env.lookat_id, :].tolist(),
              "\nhighlevel_pos:", env.actions[env.lookat_id, 2:5].tolist(),
              "\nreal_delta_yaw:", real_delta_yaw, 
              "\nhighlevel_yaw:", env.actions[env.lookat_id, 0].tolist(),
              "\nreal_delta_pitch:", real_delta_pitch,
              "\nhighlevel_pitch:", env.actions[env.lookat_id, 1].tolist(),
              "\nhighlevel_gripper open:", env.actions[env.lookat_id, -1]<0,
              "\nee_pos:", env.ee_pos[env.lookat_id, :].tolist(),
              "\nfinger_contact_force:",finger_force,
              "\nfinger_position",[[round(x,2) for x in sublist] for sublist in env.rigid_body_states[env.lookat_id, env.finger_indices, :3].tolist()]
              )
            #   "\ndof_pos:",env.dof_pos,
            #   "\nbox_position:",[round(x,2) for x in env.box_states[env.lookat_id,:3].tolist()],
        
        id = env.lookat_id
        if cur_time == 0 or i == 3*int(env.max_episode_length)-1:  #or (cur_time % env_cfg.commands.resampling_time)==0 
            time_hist = np.array(time_hist[:-3])
            cmd_hist = np.array(cmd_hist[:-3])
            state_hist = np.array(state_hist[:-3])
            ref_hist = np.array(ref_hist[:-3])
            finger_force_hist = np.array(finger_force_hist[:-3])
            fig,axs = plt.subplots(5,1,sharex=True)
            axs[0].plot(time_hist,cmd_hist[:,0],linestyle='--',label='target_x')
            axs[0].plot(time_hist,state_hist[:,0],label='x')
            axs[0].legend()
            axs[0].set_ylabel('m')
            # axs[0].set_ylim((-0.5,1.5))

            axs[1].plot(time_hist,cmd_hist[:,1],linestyle='--',label='target_y')
            axs[1].plot(time_hist,state_hist[:,1],label='y')    
            axs[1].legend()
            axs[1].set_ylabel('m')
            # axs[1].set_ylim((-1,1))

            axs[2].plot(time_hist,cmd_hist[:,2],linestyle='--',label='target_z')
            axs[2].plot(time_hist,state_hist[:,2],label='z') 
            axs[2].legend()
            axs[2].set_ylabel('m')  
            # axs[2].set_ylim((-1,1))

            axs[3].plot(time_hist,ref_hist[:,0],linestyle='--',label='ref_yaw')
            axs[3].plot(time_hist,state_hist[:,3],label='yaw')
            axs[3].legend()
            axs[3].set_ylabel('rad')
            # axs[3].set_ylim((-0.7,0.7))

            axs[4].plot(time_hist,ref_hist[:,1],linestyle='--',label='ref_pitch')
            axs[4].plot(time_hist,state_hist[:,4],label='pitch')
            axs[4].legend()
            axs[4].set_ylabel('rad')
            # axs[4].set_ylim((-0.7,0.7))

            plt.ylabel('force/N')
            plt.xlabel('time/s')
            # fig.suptitle(f"targetx,vy,yaw,pitch,grasp(>0)):{np.round(cmd_hist[0,:], decimals=2)}")
            plt.tight_layout() 
            plt.savefig(f'../figs/cmd_following_{i}.png')
            # plt.savefig(f'../figs/force_{i}_{cur_time}.png')

            time_hist = []
            cmd_hist = []
            state_hist = []
            ref_hist = []
            finger_force_hist = []

if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
