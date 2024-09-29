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
import sys

def get_load_path(root, exptid, load_run=-1, checkpoint=-1, model_name_include="model"):
    if checkpoint==-1:
        models = [file for file in os.listdir(root) if model_name_include in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
        checkpoint = model.split("_")[-1].split(".")[0]
    else:
        model = "{}-{}-base_jit.pt".format(exptid,checkpoint) 
    return model, checkpoint

def sweep(args):
    if args.web:
        web_viewer = webviewer.WebViewer(output_video_file="../figs/output.mp4")
    faulthandler.enable()

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    if args.nodelay:
        env_cfg.domain_rand.action_delay_view = 1
    env_cfg.env.num_envs = 2 if not args.save else 64  # 2
    env_cfg.env.episode_length_s = 20 # 60 30  8
    env_cfg.commands.resampling_time = 6 # 60 10  2
    env_cfg.terrain.num_rows = 2
    env_cfg.terrain.num_cols = 2
    env_cfg.terrain.height = [0.02, 0.02]
    env_cfg.terrain.terrain_dict = {
                                    "parkour_flat": 0.5,
                                    "parkour_step": 0.5,}
    
    env_cfg.terrain.terrain_proportions = list(env_cfg.terrain.terrain_dict.values())
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.max_difficulty = True
    
    env_cfg.depth.angle = [29-0.1, 29+0.1]
    env_cfg.noise.add_noise = True
    env_cfg.domain_rand.randomize_friction = True # False will use minimal friction
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_motor = False
    env_cfg.domain_rand.push_interval_s = 2
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.asset.fix_base_link = True
    env_cfg.control.stiffness = {'joint': 30.}
    env_cfg.control.damping = {'joint': 0.6}

    depth_latent_buffer = []
    # prepare environment
    env: LeggedRobot
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    if args.web:
        web_viewer.setup(env)

    actions = torch.zeros(env.num_envs, 13, device=env.device, requires_grad=False)
    joint_id = 8 #2
    total_length = 75
    r_dof3006_8 = np.load("../figs/sysid/real_data3006_8.npz")['dof_hist']
    sweep_log = []

    for kp in np.arange(10,42,4):
        for kd in np.arange(0.2, 2, 0.3):
            # prepare plot data
            obs_time_hist = []
            time_hist = []
            cmd_hist = []
            state_hist = []
            ref_hist = []
            finger_force_hist = []
            action_hist = []
            angle_hist = []
            dof_hist = []
            total_rew = []
            rew_log = {}
            
            env.p_gains = kp
            env.d_gains = kd
            env.reset_idx(torch.arange(env.num_envs, device=env.device))

            for i in range(total_length):
                if i > 50:
                    actions[:, joint_id] = -3# -3
                else:
                    actions *= 0
                obs, _, rews, dones, infos = env.step(actions.detach())
                if args.web:
                    web_viewer.render(fetch_results=True,
                                step_graphics=True,
                                render_all_camera_sensors=True,
                                wait_for_page_load=True)
                    web_viewer.write_vid()


                # store data for plot
                time_hist.append(i/50)
                dof_hist.append(env.reindex(env.dof_pos)[0].tolist())
                # print(f"{i}: time={time_hist[-1]}, dof={dof_hist[-1][joint_id]},action={actions[0][joint_id]}")
                

            time_hist = np.array(time_hist)
            angle_hist = np.array(angle_hist)
            dof_hist = np.array(dof_hist)
            err = np.linalg.norm(dof_hist[50:50+18, joint_id]-r_dof3006_8[25:25+18, joint_id])
            print(f"kp={kp}, kd={kd}, err={err}")
            sweep_log.append((kp, kd, err))

            plt.figure()
            plt.plot(dof_hist[50:50+18, joint_id], label='sim_8')
            plt.plot(r_dof3006_8[25:25+18, joint_id], label='real3006')
            # plt.plot(r_dof3204_5[20:25+18, joint_id], label='real3204')
            plt.legend()
            plt.grid(True)
            plt.savefig(f'../figs/sysid/sweep/q{joint_id}-kp{kp}-kd{kd}-err{err}.png')
            plt.close()

    sweep_log = np.array(sweep_log)
    min_id = np.argmin(sweep_log[:,2])
    kp, kd, err = sweep_log[min_id]
    print(f"best kp={kp}, kd={kd}, err={err}")


def play(args):
    if args.web:
        web_viewer = webviewer.WebViewer(output_video_file="../figs/output.mp4")
    faulthandler.enable()

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    if args.nodelay:
        env_cfg.domain_rand.action_delay_view = 1
    env_cfg.env.num_envs = 2 if not args.save else 64  # 2
    env_cfg.env.episode_length_s = 20 # 60 30  8
    env_cfg.commands.resampling_time = 6 # 60 10  2
    env_cfg.terrain.num_rows = 2
    env_cfg.terrain.num_cols = 2
    env_cfg.terrain.height = [0.02, 0.02]
    env_cfg.terrain.terrain_dict = {
                                    "parkour_flat": 0.5,
                                    "parkour_step": 0.5,}
    
    env_cfg.terrain.terrain_proportions = list(env_cfg.terrain.terrain_dict.values())
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.max_difficulty = True
    
    env_cfg.depth.angle = [29-0.1, 29+0.1]
    env_cfg.noise.add_noise = True
    env_cfg.domain_rand.randomize_friction = True # False will use minimal friction
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_motor = False
    env_cfg.domain_rand.push_interval_s = 2
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.asset.fix_base_link = True
    env_cfg.control.stiffness = {'joint': 10.}
    env_cfg.control.damping = {'joint': 0.2}

    depth_latent_buffer = []
    # prepare environment
    env: LeggedRobot
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    # prepare plot data
    obs_time_hist = []
    time_hist = []
    cmd_hist = []
    state_hist = []
    ref_hist = []
    finger_force_hist = []
    action_hist = []
    angle_hist = []
    dof_hist = []
    total_rew = []
    rew_log = {}

    if args.web:
        web_viewer.setup(env)

    actions = torch.zeros(env.num_envs, 13, device=env.device, requires_grad=False)
    print(f"{env.dof_names=}")
    joint_id = 8 #2
    total_length = 75
    for i in range(total_length):
        if i > 50:
            actions[:, joint_id] = -3# -3

        obs, _, rews, dones, infos = env.step(actions.detach())
        if args.web:
            web_viewer.render(fetch_results=True,
                        step_graphics=True,
                        render_all_camera_sensors=True,
                        wait_for_page_load=True)
            web_viewer.write_vid()


        # store data for plot
        time_hist.append(i/50)
        dof_hist.append(env.reindex(env.dof_pos)[0].tolist())
        print(f"{i}: time={time_hist[-1]}, dof={dof_hist[-1][joint_id]},action={actions[0][joint_id]}")
        
    time_hist = np.array(time_hist)
    angle_hist = np.array(angle_hist)
    dof_hist = np.array(dof_hist)
    plt.figure()
    plt.plot(time_hist[40:], dof_hist[40:, joint_id])
    plt.xticks(np.arange(40*0.02,total_length*0.02 ,0.05))
    plt.grid(True)
    plt.savefig(f'../figs/sysid/q{joint_id}.png')
    np.savez(f'../figs/sysid/q{joint_id}_sim.npz', dof_hist=dof_hist, time_hist=time_hist)

if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = True
    args = get_args()
    sweep(args)
    # play(args)
