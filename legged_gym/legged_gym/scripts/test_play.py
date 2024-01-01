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
    env_cfg.env.episode_length_s = 12 # 60
    env_cfg.commands.resampling_time = 4 # 60
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
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    # prepare plot data
    time_hist = []
    cmd_hist = []
    state_hist = []
    ref_hist = []
    if args.web:
        web_viewer.setup(env)

    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg, log_pth = task_registry.make_alg_runner(log_root = log_pth, env=env, name=args.task, args=args, train_cfg=train_cfg, model_name_include="model", return_log_dir=True)
    policy = ppo_runner.get_inference_policy(device=env.device)

    actions = torch.zeros(env.num_envs, 13, device=env.device, requires_grad=False)
    infos = {}
    infos["depth"] = env.depth_buffer.clone().to(ppo_runner.device)[:, -1] if ppo_runner.if_depth else None

    for i in range(3*int(env.max_episode_length)):
        depth_latent = None
        # actions = policy(obs.detach(), hist_encoding=True, scandots_latent=depth_latent)

        obs, _, rews, dones, infos = env.step(actions.detach())
        if args.web:
            web_viewer.render(fetch_results=True,
                        step_graphics=True,
                        render_all_camera_sensors=True,
                        wait_for_page_load=True)
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
        print("----------\ntime:", cur_time, 
              "\ntarget_pos:", env.target_position[env.lookat_id, :].tolist(),
              "\nee_pos:", env.ee_pos[env.lookat_id, :].tolist(),
              "\nbase_pos:", env.root_states[env.lookat_id, :3].tolist(),
              "\nyaw:", env.yaw[env.lookat_id].tolist(), 
              "\ntarget_yaw:", env.target_yaw[env.lookat_id].tolist(), 
              "\npitch:", env.pitch[env.lookat_id].tolist(),
              "\ntarget_pitch:", env.target_pitch[env.lookat_id].tolist(),
              "\ngripper open:", env.commands[env.lookat_id,-1]<0)
        


if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
