
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
    env_cfg.env.num_envs = 20 if not args.save else 64  # 2
    env_cfg.env.episode_length_s = 20 # 60 30  8
    env_cfg.commands.resampling_time = 6 # 60 10  2
    env_cfg.terrain.num_rows = 2
    env_cfg.terrain.num_cols = 10
    env_cfg.terrain.height = [0.02, 0.02]
    env_cfg.terrain.terrain_dict = {
                                    "parkour_flat": 0.5*0,
                                    "parkour_step": 0.5,}
    
    env_cfg.terrain.terrain_proportions = list(env_cfg.terrain.terrain_dict.values())
    env_cfg.terrain.curriculum = False
    # env_cfg.terrain.max_difficulty = True
    
    env_cfg.depth.angle = [27-5, 27+5]
    env_cfg.noise.add_noise = True
    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.push_interval_s = 6
    env_cfg.domain_rand.randomize_base_mass = True
    env_cfg.domain_rand.randomize_base_com = True

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
        if env.cfg.depth.use_camera:
            # train stuff
            if infos["depth"] != None:
                with torch.no_grad():
                    scandots_latent = ppo_runner.alg.actor_critic.actor.infer_scandots_latent(obs)
                obs_prop_depth = obs[:, :env.cfg.env.n_proprio].clone()
                obs_prop_depth[:, 6] = 0
                obs_prop_depth[:, 0] = -1
                depth_latent = ppo_runner.alg.depth_encoder(infos["depth"].clone(), obs_prop_depth)  # clone is crucial to avoid in-place operation
                depth_latent_buffer.append(depth_latent)
            
            with torch.no_grad():
                actions_teacher = ppo_runner.alg.actor_critic.act_inference(obs, hist_encoding=True, scandots_latent=None)

            obs_student = obs.clone()
            obs_student[:, 6] = 0  # mask delta_z to be 0
            obs_student[:, 0] = -1  # mask terrain class to be -1
            actions_student = ppo_runner.alg.depth_actor(obs_student, hist_encoding=True, scandots_latent=depth_latent)

            # obs, privileged_obs, rewards, dones, infos = env.step(actions_teacher.detach())  # obs has changed to next_obs !! if done obs has been reset
            obs, privileged_obs, rewards, dones, infos = env.step(actions_student.detach())  # obs has changed to next_obs !! if done obs has been reset

        if args.web:
            web_viewer.render(fetch_results=True,
                        step_graphics=True,
                        render_all_camera_sensors=True,
                        wait_for_page_load=True)
            web_viewer.write_vid()


if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = True
    args = get_args()
    play(args)
