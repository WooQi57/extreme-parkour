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
    env_cfg.env.num_envs = 8 if not args.save else 64  # 2
    env_cfg.env.episode_length_s = 20 # 60 30  8
    env_cfg.commands.resampling_time = 6 # 60 10  2
    env_cfg.terrain.num_rows = 2
    env_cfg.terrain.num_cols = 4
    env_cfg.terrain.height = [0.02, 0.02]
    env_cfg.terrain.terrain_dict = {
                                    "parkour_flat": 0.5,
                                    "parkour_step": 0.5,}
    
    env_cfg.terrain.terrain_proportions = list(env_cfg.terrain.terrain_dict.values())
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.max_difficulty = True
    
    env_cfg.depth.angle = [29-0.1, 29+0.1]
    env_cfg.position = [0.3, 0, 0.147]
    env_cfg.position_rand = 0.01*0
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False # False will use minimal friction
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.push_interval_s = 2
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.randomize_motor = False
    env_cfg.rewards.print_rewards = False
    
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
    torque_hist = []
    rew_log = {}

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

    actions = torch.zeros(env.num_envs, 13, device=env.device, requires_grad=False)
    infos = {}
    infos["depth"] = env.depth_buffer.clone().to(ppo_runner.device)[:, -1] if ppo_runner.if_depth else None
    print(f"{env.friction_coeffs_tensor=}")
    
    # # 203-30
    # env.motor_strength[0,:]=torch.tensor([0.9929, 1.0525, 1.0009, 0.8281, 0.8838, 1.1968, 0.8290, 1.1138, 1.1148, 1.0191, 0.9686, 1.1285, 1.1913, 0.8139], device=env.device)
    # env.motor_strength[1,:]=torch.tensor([1.0830, 0.9201, 0.9799, 1.1462, 1.1100, 0.8942, 0.9411, 0.8063, 1.1274, 1.1894, 0.9046, 0.9850, 1.0292, 1.0455], device=env.device)
    # 207-31
    # env.motor_strength[0,:]=torch.tensor([0.8317, 1.0933, 0.9149, 0.9054, 0.9159, 0.9923, 0.8742, 0.9675, 1.1260, 0.9508, 1.1286, 1.1584, 1.0495, 0.8562], device=env.device)
    # env.motor_strength[1,:]=torch.tensor([0.9130, 1.0389, 0.9475, 1.1827, 0.9942, 0.8441, 1.0007, 1.1131, 1.0615, 1.1286, 0.9774, 0.9170, 1.1695, 0.9929], device=env.device)
    # 202-30
    # env.motor_strength[0,:]=torch.tensor([1.1969, 0.8113, 1.1885, 1.0147, 0.9823, 1.1997, 0.8627, 1.0189, 1.1024, 0.9395, 0.9926, 0.8427, 1.1424, 1.0296], device=env.device)
    # env.motor_strength[1,:]=torch.tensor([1.1661, 1.0459, 1.1494, 1.0219, 1.1172, 1.1486, 0.8885, 1.1164, 1.0610, 1.1412, 0.9780, 0.9431, 0.8005, 0.9595], device=env.device)
    # 205-30
    # env.motor_strength[0,:]=torch.tensor([1.1858, 1.1568, 1.0145, 0.8776, 0.9550, 0.9836, 0.9865, 0.9090, 0.9550, 0.8003, 1.1019, 1.1277, 0.9863, 1.1425], device=env.device)
    # env.motor_strength[1,:]=torch.tensor([1.1224, 1.0190, 0.9427, 0.9467, 0.8892, 0.8726, 0.8324, 0.8467, 1.0163, 0.8173, 0.8674, 0.9425, 1.1454, 1.1143], device=env.device)
    # 204-30 can't find a good one
    # 202-34
    # env.motor_strength[0,:]=torch.tensor([0.8852, 1.0392, 0.8399, 1.1666, 1.1056, 0.9707, 0.8283, 1.0034, 1.1107, 0.8862, 0.9946, 0.9798, 1.1137, 0.8723], device=env.device)
    # env.motor_strength[1,:]=torch.tensor([0.8600, 0.9529, 0.9313, 1.1716, 1.1772, 0.9846, 0.9400, 1.0100, 1.1292, 1.0291, 0.8349, 0.9301, 0.8496, 1.0705], device=env.device)

    print(f"{env.motor_strength=}") 
    # print(f"{env.dof_pos_limits=}")
    for i in range(2*int(env.max_episode_length)):
        if args.use_jit:
            if env.cfg.depth.use_camera:
                if infos["depth"] is not None:
                    obs_student = obs[:, :env.cfg.env.n_proprio].clone()
                    obs_student[:,0] = -1
                    obs_student[:, 6] = 0
                    depth_latent = depth_encoder(infos["depth"], obs_student)
                # else:
                #     depth_buffer = torch.ones((env_cfg.env.num_envs, 58, 87), device=env.device)
                #     actions = policy_jit(obs.detach(), torch.ones(env.num_envs, 32, device=env.device))
                obs[:, 0] = -1
                obs[:, 6] = 0
                # depth_latent[0]=torch.tensor([-0.8619,  0.9112,  0.8675,  0.9987,  1.0000,  0.9995, -0.6052, -0.1257,
                #                             -0.5195, -0.9969,  0.0310, -0.7561, -0.9255, -0.7634,  0.3352, -1.0000,
                #                             0.8715, -0.2612, -0.9900,  0.8102, -0.5170,  0.9999, -0.9357, -0.6454,
                #                             -0.9972, -0.8571,  0.9534, -0.9999, -0.9883,  0.7779, -0.9103, -0.9840],
                #                         device='cuda:0')
                
                actions = policy_jit(obs.detach(), depth_latent)

                # with np.printoptions(threshold=np.inf):
                    # print(depth_latent)
                # if i == 150:
                #     np.save(f"depth_image.npy", infos["depth"][0].cpu().numpy())
                #     import matplotlib.pyplot as plt
                #     plt.imsave("first_frame.png", infos["depth"][0].cpu().numpy(), vmin=-2, vmax=2, cmap='gray')
                # actions = ppo_runner.alg.depth_actor(obs.detach(), hist_encoding=True, scandots_latent=depth_latent)
                # print(f"jit actions:{actions}\noriginal actions:{original_actions}")
                # print(f"1 diff:{actions-original_actions}")
            else:
                obs_jit = torch.cat((obs.detach()[:, :env_cfg.env.n_proprio+env_cfg.env.n_priv], obs.detach()[:, -env_cfg.env.history_len*env_cfg.env.n_proprio:]), dim=1)
                actions = policy_jit(obs.detach())



        else:
            if env.cfg.depth.use_camera:
                if infos["depth"] is not None:
                    obs_student = obs[:, :env.cfg.env.n_proprio].clone()
                    obs_student[:,0] = -1
                    obs_student[:, 6] = 0
                    depth_latent = depth_encoder(infos["depth"], obs_student)
                    # depth_latent = depth_latent_and_yaw[:, :-2]
                    # yaw = depth_latent_and_yaw[:, -2:]
                obs[:, 0] = -1
                obs[:, 6] = 0
            else:
                depth_latent = None

            if hasattr(ppo_runner.alg, "depth_actor"):
                actions = ppo_runner.alg.depth_actor(obs.detach(), hist_encoding=True, scandots_latent=depth_latent)
            else:
                # obs[0,env_cfg.env.n_proprio+env_cfg.env.n_scan:env_cfg.env.n_proprio+env_cfg.env.n_scan+env_cfg.env.n_priv] = estimator(obs[:, :env_cfg.env.n_proprio])
                actions = policy(obs.detach(), hist_encoding=True, scandots_latent=depth_latent)



            # state_dict = ppo_runner.alg.depth_actor.state_dict()
            # for param_tensor in state_dict:
            #     print(f"Layer: {param_tensor} \nShape: {state_dict[param_tensor].size()} \nValues: {state_dict[param_tensor]}\n")
            # raise NotImplementedError
        obs, _, rews, dones, infos = env.step(actions.detach())
        if args.web:
            web_viewer.render(fetch_results=True,
                        step_graphics=True,
                        render_all_camera_sensors=True,
                        wait_for_page_load=True)
            web_viewer.write_vid()
        # if i == 0:
            # print("\n"*len(env.reward_functions))
        # else:
            # # sys.stdout.write("\033[F" * len(env.reward_functions))
            # for r in range(len(env.reward_functions)):
            #     name = env.reward_names[r]
            #     rew = env.reward_functions[r]() * env.reward_scales[name]
            #     sys.stdout.write(f"{r}_{name}:{rew[-1]}\n")
            # sys.stdout.flush()



        # store data for plot
        cur_time = env.episode_length_buf[env.lookat_id].item() / 50
        time_hist.append(cur_time)
        angle_hist.append(env.target_angles[env.lookat_id].tolist())
        dof_hist.append(env.dof_pos[env.lookat_id].tolist())
        torque_hist.append(env.torques[-1].cpu().tolist())
        if env_cfg.rewards.print_rewards:
            if len(rew_log) == 0:
                for name in env.reward_names:
                    rew_log[name] = []
                    rew_log["total_rew"] = []
            for name in env.reward_names:
                rew_log[name].append(env.rew_log[name].tolist())
            rew_log["total_rew"].append(env.rew_log["total_rew"].tolist())
        
        action_hist.append(actions[-1].cpu().tolist())

        id = env.lookat_id
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
        #         # plt.tight_layout()
        #         # plt.savefig(f'../figs/cmd_following_{i}_{f}.png')

        #     time_hist = []
        #     angle_hist = []
        #     dof_hist = []

        if i == 400:
            np.save("../figs/torques.npy",np.array(torque_hist))
            # action_hist_np = np.array(action_hist)
            # np.save("../figs/replay_data.npy", action_hist_np)#280
            # print("saving replay data for env -1")

        if i == 500 and env_cfg.rewards.print_rewards:
            for k,name in enumerate(rew_log):
                plt.figure()
                plt.plot(np.arange(len(rew_log[name]))*0.02, rew_log[name])
                plt.xticks(np.arange(0,len(rew_log[name])*0.02 ,0.5))
                if name == 'total_rew':
                    plt.savefig('../figs/reward/total_rew.png')
                else:
                    plt.savefig(f'../figs/reward/{k}_{name}.png')
                plt.close()
    if env_cfg.rewards.print_rewards:       
        print("---------------- mean reward ----------------")
        for k,name in enumerate(rew_log):
            print(f"{k}_{name}: {statistics.mean(rew_log[name])}")
        print(f"total rew mean: {statistics.mean(rew_log['total_rew'])}")

if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = True
    args = get_args()
    play(args)
