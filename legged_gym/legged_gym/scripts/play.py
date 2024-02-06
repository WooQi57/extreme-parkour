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

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from crc_module import get_crc

from legged_gym.envs.go1g.deploy import *
from legged_gym.envs.go1g.deploy_config import *
from legged_gym.utils import get_args, task_registry
import numpy as np
import torch
import faulthandler
import matplotlib.pyplot as plt

import rclpy
from rclpy.node import Node
from unitree_go.msg import (
    WirelessController,
    LowState,
    MotorState,
    IMUState,
    LowCmd,
    MotorCmd,
)
import time

POS_STOP_F = 2.146e9
VEL_STOP_F = 16000.0
LEG_DOF = 12
SDK_DOF = 20

# TODO: add emergency stop


class DeployNode(Node):
    def __init__(self, policy_args):
        super().__init__("deploy_node")  # type: ignore

        # init subcribers
        self.joy_stick_sub = self.create_subscription(
            WirelessController, "wirelesscontroller", self.joy_stick_cb, 10
        )
        self.joy_stick_sub  # prevent unused variable warning
        self.lowlevel_state_sub = self.create_subscription(
            LowState, "lowstate", self.lowlevel_state_cb, 10
        )  # "/lowcmd" or  "lf/lowstate" (low frequencies)
        self.lowlevel_state_sub  # prevent unused variable warning

        self.low_state = LowState()
        self.joint_pos = np.zeros(LEG_DOF)
        self.joint_vel = np.zeros(LEG_DOF)

        # init publishers
        self.motor_pub = self.create_publisher(LowCmd, "lowcmd", 10)
        self.motor_pub_freq = 200
        self.motor_timer = self.create_timer(
            1.0 / self.motor_pub_freq, self.motor_timer_callback
        )
        self.cmd_msg = LowCmd()

        # init motor command
        self.motor_cmd = [
            MotorCmd(q=POS_STOP_F, dq=VEL_STOP_F, tau=0.0, kp=0.0, kd=0.0, mode=0x01)
            for _ in range(SDK_DOF)
        ]
        self.cmd_msg.motor_cmd = self.motor_cmd.copy()
        self.kp = 0
        self.kd = 0

        # init policy
        self.init_policy(policy_args)
        self.policy_freq = 50
        self.policy_timer = self.create_timer(
            1.0 / self.policy_freq, self.policy_timer_callback
        )

        self.start_time = time.monotonic()

    ##############################
    # subscriber callbacks
    ##############################

    def joy_stick_cb(self, msg):
        self.get_logger().info(
            "Wireless controller -- lx: %f; ly: %f; rx: %f; ry: %f; key value: %d"
            % (msg.lx, msg.ly, msg.rx, msg.ry, msg.keys)
        )

    def lowlevel_state_cb(self, msg: LowState):
        # imu data
        imu_data = msg.imu_state

        self.get_logger().info(
            "Euler angle -- roll: %f; pitch: %f; yaw: %f"
            % (imu_data.rpy[0], imu_data.rpy[1], imu_data.rpy[2])
        )
        self.get_logger().info(
            "Quaternion -- qw: %f; qx: %f; qy: %f; qz: %f"
            % (
                imu_data.quaternion[0],
                imu_data.quaternion[1],
                imu_data.quaternion[2],
                imu_data.quaternion[3],
            )
        )
        self.get_logger().info(
            "Gyroscope -- wx: %f; wy: %f; wz: %f"
            % (imu_data.gyroscope[0], imu_data.gyroscope[1], imu_data.gyroscope[2])
        )
        self.get_logger().info(
            "Accelerometer -- ax: %f; ay: %f; az: %f"
            % (
                imu_data.accelerometer[0],
                imu_data.accelerometer[1],
                imu_data.accelerometer[2],
            )
        )

        # motor data
        for motor_id in range(LEG_DOF):
            motor_data = msg.motor_state[motor_id]  # type: ignore
            self.get_logger().info(
                "Motor state -- num: %d; q: %f; dq: %f; ddq: %f; tau: %f"
                % (
                    motor_id,
                    motor_data.q,
                    motor_data.dq,
                    motor_data.ddq,
                    motor_data.tau_est,
                )
            )

        # foot force data
        foot_force = []
        foot_force_est = []
        for foot_id in range(4):
            foot_force.append(msg.foot_force[foot_id])
            foot_force_est.append(msg.foot_force_est[foot_id])

        self.get_logger().info(
            "Foot force -- foot0: %d; foot1: %d; foot2: %d; foot3: %d"
            % (foot_force[0], foot_force[1], foot_force[2], foot_force[3])
        )
        self.get_logger().info(
            "Estimated foot force -- foot0: %d; foot1: %d; foot2: %d; foot3: %d"
            % (
                foot_force_est[0],
                foot_force_est[1],
                foot_force_est[2],
                foot_force_est[3],
            )
        )

        # battery data
        battery_current = msg.power_a
        battery_voltage = msg.power_v

        self.joint_pos = np.array([motor_data.q for motor_data in msg.motor_state])
        self.joint_vel = np.array([motor_data.dq for motor_data in msg.motor_state])
        # self.get_logger().info("Battery state -- current: %f; voltage: %f" %(battery_current, battery_voltage))

    ##############################
    # motor commands
    ##############################

    def set_gains(self, kp: float, kd: float):
        self.kp = kp
        self.kd = kd
        for i in range(LEG_DOF):
            self.motor_cmd[i].kp = kp
            self.motor_cmd[i].kd = kd

    def motor_timer_callback(self):
        self.cmd_msg.crc = get_crc(self.cmd_msg)
        self.motor_pub.publish(self.cmd_msg)

    def set_motor_position(
        self,
        q: np.ndarray,
    ):
        for i in range(LEG_DOF):
            self.motor_cmd[i].q = float(q[i])
        self.cmd_msg.motor_cmd = self.motor_cmd.copy()

    def emergency_stop(self):
        self.motor_cmd = [
            MotorCmd(q=POS_STOP_F, dq=VEL_STOP_F, tau=0.0, kp=0.0, kd=0.0, mode=0x01)
            for _ in range(SDK_DOF)
        ]
        self.cmd_msg.motor_cmd = self.motor_cmd.copy()
        self.motor_timer_callback()

    ##############################
    # policy inference
    ##############################

    def policy_timer_callback(self):
        target_kp = 40
        target_kd = 0.6
        stand_up_time = 3.0
        if time.monotonic() - self.start_time < stand_up_time:
            time_ratio = (time.monotonic() - self.start_time) / stand_up_time
            self.set_gains(kp=time_ratio * target_kp, kd=time_ratio * target_kd)
            self.set_motor_position(
                q=self.env.default_dof_pos[0].cpu().detach().numpy()
            )
        elif time.monotonic() - self.start_time < 2 * stand_up_time:
            pass
        else:
            actions = self.policy(
                self.obs.detach(), hist_encoding=True, scandots_latent=None
            )
            angles = self.env.compute_angle(actions)
            self.get_logger().info(f"angles: {angles}")
            # self.set_motor_position(angles.cpu().detach().numpy()[0])

        # apply actions
        # obs = get_observations()

    def init_policy(self, args):
        self.get_logger().info("Preparing policy")
        faulthandler.enable()
        log_pth = os.path.dirname(os.path.realpath(__file__))
        env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
        env_cfg.env.num_envs = 1

        # prepare environment
        self.env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
        # obs = get_observations()
        self.obs = torch.zeros(1, env_cfg.env.num_observations, device="cuda")

        # load policy
        train_cfg.runner.resume = True
        ppo_runner, train_cfg, log_pth = task_registry.make_alg_runner(
            log_root=log_pth,
            env=self.env,
            name=args.task,
            args=args,
            train_cfg=train_cfg,
            model_name_include="lowlevel_pos.pt",
            return_log_dir=True,
        )
        self.policy = ppo_runner.get_inference_policy(device=self.env.device)
        actions = torch.zeros(1, 13, device=self.env.device, requires_grad=False)
        actions = self.policy(self.obs.detach(), hist_encoding=True, scandots_latent=None)

        # init p_gains, d_gains, torque_limits, default_dof_pos_all
        for i in range(LEG_DOF):
            self.motor_cmd[i].q = self.env.default_dof_pos[0][i].item()
            self.motor_cmd[i].dq = 0.0
            self.motor_cmd[i].tau = 0.0
            self.motor_cmd[i].kp = 0.0  # self.env.p_gains[i]  # 30
            self.motor_cmd[i].kd = 0.0  # self.env.d_gains[i]  # 0.6
        self.cmd_msg.motor_cmd = self.motor_cmd.copy()

        self.get_logger().info("starting to play policy")
        input("Press Enter to start...")
        # init angles:([-0.1000,  0.8000, -1.5000,  0.1000,  0.8000, -1.5000, -0.1000,  1.0000,
        # -1.5000,  0.1000,  1.0000, -1.5000], device='cuda:0')

if __name__ == "__main__":
    rclpy.init(args=None)
    args = get_args()
    dp_node = DeployNode(args)
    dp_node.get_logger().info("Deploy node started")
    rclpy.spin(dp_node)
    rclpy.shutdown()

    # # prepare plot data
    # time_hist = []
    # cmd_hist = []
    # state_hist = []
    # ref_hist = []
    # finger_force_hist = []

    # # plot results

    # # store data for plot
    # cur_time = env.episode_length_buf[env.lookat_id].item() / 50
    # time_hist.append(cur_time)
    # cmd_hist.append((env.target_position[env.lookat_id, :]).tolist())
    # cur_state = env.ee_pos[env.lookat_id, :].tolist()
    # cur_state.append(env.yaw[env.lookat_id].tolist())
    # cur_state.append(env.pitch[env.lookat_id].tolist())
    # state_hist.append(cur_state)
    # ref = [env.target_yaw[env.lookat_id].tolist(),env.target_pitch[env.lookat_id].tolist()]
    # ref_hist.append(ref)

    # real_delta_yaw = env.target_yaw[env.lookat_id].tolist() - env.yaw[env.lookat_id].tolist()
    # real_delta_pitch = env.target_pitch[env.lookat_id].tolist() - env.pitch[env.lookat_id].tolist()
    # finger_force = torch.norm(env.contact_forces[env.lookat_id, env.finger_indices, :],dim=1).tolist()
    # finger_force = env.contact_forces[env.lookat_id, env.finger_indices, :].tolist()
    # # finger_force_hist.append(finger_force)
    # # print("----------\ntime:", cur_time,
    # #       "\nbase_target_pos:", env.base_target_pos[env.lookat_id, :].tolist(),
    # #       "\nreal_delta_pos:", env.target_pos_rel[env.lookat_id, :].tolist(),
    # #       "\nhighlevel_vel:", env.actions[env.lookat_id, :2].tolist(),
    # #       "\nreal_target_yaw:", env.target_yaw[env.lookat_id].tolist(),
    # #       "\nhighlevel_yaw:", env.actions[env.lookat_id, 2].tolist(),
    # #       "\nreal_target_pitch:", env.target_pitch[env.lookat_id].tolist(),
    # #       "\nhighlevel_pitch:", env.actions[env.lookat_id, 3].tolist(),
    # #       "\nhighlevel_gripper open:", env.actions[env.lookat_id, -1]<0,
    # #       "\nee_pos:", env.ee_pos[env.lookat_id, :].tolist(),
    # #       "\nfinger_contact_force:",finger_force,
    # #       "\nfinger_position",[[round(x,2) for x in sublist] for sublist in env.rigid_body_states[env.lookat_id, env.finger_indices, :3].tolist()]
    # #       )
    #     #   "\ndof_pos:",env.dof_pos,
    #     #   "\nbox_position:",[round(x,2) for x in env.box_states[env.lookat_id,:3].tolist()],

    # id = env.lookat_id
    # # if cur_time == 0 or i == 3*int(env.max_episode_length)-1:  #or (cur_time % env_cfg.commands.resampling_time)==0
    # #     time_hist = np.array(time_hist[:-3])
    # #     cmd_hist = np.array(cmd_hist[:-3])
    # #     state_hist = np.array(state_hist[:-3])
    # #     ref_hist = np.array(ref_hist[:-3])
    # #     finger_force_hist = np.array(finger_force_hist[:-3])
    # #     fig,axs = plt.subplots(5,1,sharex=True)
    # #     axs[0].plot(time_hist,cmd_hist[:,0],linestyle='--',label='target_x')
    # #     axs[0].plot(time_hist,state_hist[:,0],label='x')
    # #     axs[0].legend()
    # #     axs[0].set_ylabel('m')
    # #     # axs[0].set_ylim((-0.5,1.5))

    # #     axs[1].plot(time_hist,cmd_hist[:,1],linestyle='--',label='target_y')
    # #     axs[1].plot(time_hist,state_hist[:,1],label='y')
    # #     axs[1].legend()
    # #     axs[1].set_ylabel('m')
    # #     # axs[1].set_ylim((-1,1))

    # #     axs[2].plot(time_hist,cmd_hist[:,2],linestyle='--',label='target_z')
    # #     axs[2].plot(time_hist,state_hist[:,2],label='z')
    # #     axs[2].legend()
    # #     axs[2].set_ylabel('m')
    # #     # axs[2].set_ylim((-1,1))

    # #     axs[3].plot(time_hist,ref_hist[:,0],linestyle='--',label='ref_yaw')
    # #     axs[3].plot(time_hist,state_hist[:,3],label='yaw')
    # #     axs[3].legend()
    # #     axs[3].set_ylabel('rad')
    # #     # axs[3].set_ylim((-0.7,0.7))

    # #     axs[4].plot(time_hist,ref_hist[:,1],linestyle='--',label='ref_pitch')
    # #     axs[4].plot(time_hist,state_hist[:,4],label='pitch')
    # #     axs[4].legend()
    # #     axs[4].set_ylabel('rad')
    # #     # axs[4].set_ylim((-0.7,0.7))

    # #     plt.ylabel('force/N')
    # #     plt.xlabel('time/s')
    # #     # fig.suptitle(f"targetx,vy,yaw,pitch,grasp(>0)):{np.round(cmd_hist[0,:], decimals=2)}")
    # #     plt.tight_layout()
    # #     plt.savefig(f'../figs/cmd_following_{i}.png')
    # #     # plt.savefig(f'../figs/force_{i}_{cur_time}.png')

    # #     time_hist = []
    # #     cmd_hist = []
    # #     state_hist = []
    # #     ref_hist = []
    # #     finger_force_hist = []
