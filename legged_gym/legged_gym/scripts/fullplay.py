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

def get_load_path(root, exptid, load_run=-1, checkpoint=-1, model_name_include="model"):
    if checkpoint==-1:
        models = [file for file in os.listdir(root) if model_name_include in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
        checkpoint = model.split("_")[-1].split(".")[0]
    else:
        model = "{}-{}-base_jit.pt".format(exptid,checkpoint) 
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
        env_cfg.domain_rand.action_delay_view = 1
    env_cfg.env.num_envs = 40 if not args.save else 64  # 2
    env_cfg.env.episode_length_s = 20 # 60 30  8
    env_cfg.commands.resampling_time = 6 # 60 10  2
    env_cfg.terrain.num_rows = 2
    env_cfg.terrain.num_cols = 40
    env_cfg.terrain.height = [0.02, 0.02]
    env_cfg.terrain.terrain_dict = {
                                    "parkour_flat": 0.5*0,
                                    "parkour_step": 0.5,}
    
    env_cfg.terrain.terrain_proportions = list(env_cfg.terrain.terrain_dict.values())
    env_cfg.terrain.curriculum = False
    # env_cfg.terrain.max_difficulty = True
    
    env_cfg.depth.angle = [27-0, 27+1]
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
    obs_time_hist = []
    time_hist = []
    cmd_hist = []
    state_hist = []
    ref_hist = []
    finger_force_hist = []
    action_hist = []
    angle_hist = []
    dof_hist = []

    if args.web:
        web_viewer.setup(env)

    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg, log_pth = task_registry.make_alg_runner(log_root = log_pth, env=env, name=args.task, args=args, train_cfg=train_cfg, model_name_include="model", return_log_dir=True)
    if args.use_jit:
        path = os.path.join(log_pth, "traced")
        model, checkpoint = get_load_path(root=path, exptid=args.exptid, checkpoint=args.checkpoint)
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

    for i in range(2*int(env.max_episode_length)):
        if args.use_jit:
            if env.cfg.depth.use_camera:
                if infos["depth"] is not None:
                    depth_latent = torch.ones((env_cfg.env.num_envs, 32), device=env.device)
                    
                    # depth_latent[0]=torch.tensor([-0.0893, -0.8894,  0.2022,  0.9225, -0.2467,  0.1751, -0.0564,  0.1091,
                    # -0.2128,  0.7953,  0.3242,  0.2088,  0.2267,  0.1037,  0.2565, -0.0339,
                    # -0.9887,  0.2621,  0.0845, -0.7242, -0.3818,  0.0784,  0.1277, -0.7304,
                    # 0.0136,  0.1669,  0.0916, -0.1428, -0.9189,  0.8611,  0.1624, -0.2422],device=env.device)
                    # actions = policy_jit(obs.detach(), depth_latent)
                    actions = policy_jit(obs.detach(), infos["depth"])
                else:
                    # depth_buffer = torch.ones((env_cfg.env.num_envs, 32), device=env.device)
                    # depth_latent[0]=torch.tensor([-0.0893, -0.8894,  0.2022,  0.9225, -0.2467,  0.1751, -0.0564,  0.1091,
                    # -0.2128,  0.7953,  0.3242,  0.2088,  0.2267,  0.1037,  0.2565, -0.0339,
                    # -0.9887,  0.2621,  0.0845, -0.7242, -0.3818,  0.0784,  0.1277, -0.7304,
                    # 0.0136,  0.1669,  0.0916, -0.1428, -0.9189,  0.8611,  0.1624, -0.2422],device=env.device)
                    
                    depth_buffer = torch.ones((env_cfg.env.num_envs, 58, 87), device=env.device)
                    actions = policy_jit(obs.detach(), depth_buffer)
            else:
                obs_jit = torch.cat((obs.detach()[:, :env_cfg.env.n_proprio+env_cfg.env.n_priv], obs.detach()[:, -env_cfg.env.history_len*env_cfg.env.n_proprio:]), dim=1)
                actions = policy_jit(obs.detach())
        else:
            if env.cfg.depth.use_camera:
                if infos["depth"] is not None:
                    obs_student = obs[:, :env.cfg.env.n_proprio].clone()
                    obs_student[:, 5] = 0
                    depth_latent = depth_encoder(infos["depth"], obs_student)
                    # depth_latent = depth_latent_and_yaw[:, :-2]
                    # yaw = depth_latent_and_yaw[:, -2:]
                # obs[:, 6:8] = 1.5*yaw
                    
            else:
                depth_latent = None

        #     depth_latent[0]=torch.tensor([-0.0893, -0.8894,  0.2022,  0.9225, -0.2467,  0.1751, -0.0564,  0.1091,
        #  -0.2128,  0.7953,  0.3242,  0.2088,  0.2267,  0.1037,  0.2565, -0.0339,
        #  -0.9887,  0.2621,  0.0845, -0.7242, -0.3818,  0.0784,  0.1277, -0.7304,
        #   0.0136,  0.1669,  0.0916, -0.1428, -0.9189,  0.8611,  0.1624, -0.2422],device=env.device)
       
            if hasattr(ppo_runner.alg, "depth_actor"):
                actions = ppo_runner.alg.depth_actor(obs.detach(), hist_encoding=True, scandots_latent=depth_latent)
            else:
                # obs[0,env_cfg.env.n_proprio+env_cfg.env.n_scan:env_cfg.env.n_proprio+env_cfg.env.n_scan+env_cfg.env.n_priv] = estimator(obs[:, :env_cfg.env.n_proprio])
                # print((obs==0).sum().item())
                actions = policy(obs.detach(), hist_encoding=True, scandots_latent=depth_latent)
        # print(env.commands[:,1])
        obs, _, rews, dones, infos = env.step(actions.detach())
        if args.web:
            web_viewer.render(fetch_results=True,
                        step_graphics=True,
                        render_all_camera_sensors=True,
                        wait_for_page_load=True)
            web_viewer.write_vid()

        # # store data for plot
        # cur_time = env.episode_length_buf[env.lookat_id].item() / 50
        # time_hist.append(cur_time)
        # angle_hist.append(env.target_angles[env.lookat_id].tolist())
        # dof_hist.append(env.dof_pos[env.lookat_id].tolist())
        # # action_hist.append(actions[env.lookat_id].tolist())

        # id = env.lookat_id
        # if cur_time == 0 or i == 2*int(env.max_episode_length)-1:  #or (cur_time % env_cfg.commands.resampling_time)==0 
        #     time_hist = np.array(time_hist[:-3])
        #     angle_hist = np.array(angle_hist[:-3])
        #     dof_hist = np.array(dof_hist[:-3])
        #     # action_hist = np.array(action_hist[:-3])
        #     for f in range(4):
        #         fig,axs = plt.subplots(3,1,sharex=True)
        #         for j in range(3):
        #             axs[j].plot(time_hist, angle_hist[:,j+3*f], label=f'cmd{j}')
        #             axs[j].plot(time_hist[:-1], dof_hist[1:,j+3*f], '--', label=f'real{j}')  # dof observe is actually one step earlier
        #             axs[j].legend()
        #             axs[j].grid(True, which='both', axis='both')
        #             axs[j].minorticks_on()
        #         plt.tight_layout()
                # plt.savefig(f'../figs/cmd_following_{i}_{f}.png')

            # time_hist = []
            # angle_hist = []
            # dof_hist = []

if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = True
    args = get_args()
    play(args)