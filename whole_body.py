import os
import sys
import multiprocessing

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from crc_module import get_crc

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
import requests
import threading
from web_hand import *
from dynamixel_sdk_custom_interfaces.msg import SetPosition
import atexit

POS_STOP_F = 2.146e9
VEL_STOP_F = 16000.0
HW_DOF = 20

WALK_STRAIGHT = False
PLOT_DATA = False
USE_GRIPPPER = False

SERVER = '172.24.68.171'
se = requests.Session()


def wrap_to_pi(angles):
    angles %= 2*np.pi
    angles -= 2*np.pi * (angles > np.pi)
    return angles

def euler_from_quat(quat_angle):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    x = quat_angle[:,0]
    y = quat_angle[:,1]
    z = quat_angle[:,2]
    w = quat_angle[:,3]
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll_x = torch.atan2(t0, t1)

    t2 = 2.0 * (w * y - z * x)
    t2 = torch.clip(t2, -1, 1)
    pitch_y = torch.asin(t2)

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw_z = torch.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z # in radians

def isWeak(motor_index):
    return motor_index == 10 or motor_index == 11 or motor_index == 12 or motor_index == 13 or \
        motor_index == 14 or motor_index == 15 or motor_index == 16 or motor_index == 17 or \
        motor_index == 18 or motor_index == 19

def cleanup(ser_l,ser_r):
    print("closing ports")
    ser_r.flush()
    ser_r.reset_input_buffer()
    ser_r.reset_output_buffer()
    ser_l.flush()
    ser_l.reset_input_buffer()
    ser_l.reset_output_buffer()
    ser_r.close()
    ser_l.close()

class H1():
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.num_envs = 1 
        self.num_observations = 62 + 19 + 5
        self.num_privileged_obs = None
        self.num_actions = 19
        self.obs_context_len=8

        self.scale_lin_vel = 1.0
        self.scale_ang_vel = 1.0
        self.scale_orn = 1.0
        self.scale_dof_pos = 1.0
        self.scale_dof_vel = 1.0
        self.scale_action = 1.0

        # prepare action deployment joint positions offsets and PD gains
        # self.p_gains = np.array([200,200,300,200,200,300,300,200,200,0,40,40,100,100,100,100,100,100,100,100],dtype=np.float64)
        self.p_gains = np.array([200,200,200,200,200,200,200,200,200,0,80,80,80*0.2,80*0.2,80*0.2,80*0.2,80*0.2,80*0.2,80*0.2,80*0.2],dtype=np.float64)
        self.d_gains = np.array([5,5,5,5,5,5,5,5,5,0,2,2,2,2,2,2,2,2,2,2],dtype=np.float64)
        self.joint_limit_lo = [-0.43,-1.57,-0.26,-0.43,-1.57,-0.26,-2.35,-0.43,-0.43,0,-0.87,-0.87,-2.87,-3.11,-4.45,-1.25,-2.87,-0.34,-1.3,-1.25]
        self.joint_limit_hi = [0.43,1.57,2.05,0.43,1.57,2.05,2.35,0.43,0.43,0,0.52,0.52,2.87,0.34,1.3,2.61,2.87,3.11,4.45,2.61]
        self.default_dof_pos_np = np.array([0.0,-10/180*np.pi,20/180*np.pi,0.0,-10/180*np.pi,\
                                            20/180*np.pi,0.0,0.42,-0.42,0.0,\
                                                -10/180*np.pi, -10/180*np.pi,0.0,0.0,0.0,\
                                                    0.0,0.0,0.0,0.0,0.0])
        default_dof_pos = torch.tensor(self.default_dof_pos_np, dtype=torch.float, device=self.device, requires_grad=False)
        self.default_dof_pos = default_dof_pos.unsqueeze(0)


        # prepare osbervations buffer
        self.obs_buf = torch.zeros(1, self.num_observations, dtype=torch.float, device=self.device, requires_grad=False)
        self.obs_history_buf = torch.zeros(1, self.obs_context_len, self.num_observations, dtype=torch.float, device=self.device, requires_grad=False)


class DeployNode(Node):
    def __init__(self):
        super().__init__("deploy_node")  # type: ignore

        # init subcribers & publishers
        self.joy_stick_sub = self.create_subscription(WirelessController, "wirelesscontroller", self.joy_stick_cb, 1)
        self.joy_stick_sub  # prevent unused variable warning
        self.lowlevel_state_sub = self.create_subscription(LowState, "lowstate", self.lowlevel_state_cb, 1)  # "/lowcmd" or  "lf/lowstate" (low frequencies)
        self.lowlevel_state_sub  # prevent unused variable warning

        self.low_state = LowState()
        self.joint_pos = np.zeros(HW_DOF)
        self.joint_vel = np.zeros(HW_DOF)

        self.motor_pub = self.create_publisher(LowCmd, "lowcmd", 1)
        self.motor_pub_freq = 50
        self.cmd_msg = LowCmd()
        
        self.wrist_pub = self.create_publisher(SetPosition, "/set_position", 1)
        self.wrist_timer = self.create_timer(1.0 / 60, self.wrist_timer_callback)
        self.wrist_msg_l = SetPosition()
        self.wrist_msg_l.id = 1
        self.wrist_msg_r = SetPosition()
        self.wrist_msg_r.id = 2
        self.wrist_l = multiprocessing.Value('i', 2048)
        self.wrist_r = multiprocessing.Value('i', 2048)
        self.wrist_i = 0

        # init motor command
        self.motor_cmd = []
        for id in range(HW_DOF):
            if isWeak(id):
                mode = 0x01
            else:
                mode = 0x0A
            cmd=MotorCmd(q=0.0, dq=0.0, tau=0.0, kp=0.0, kd=0.0, mode=mode, reserve=[0,0,0])
            self.motor_cmd.append(cmd)
        self.cmd_msg.motor_cmd = self.motor_cmd.copy()

        # init policy
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.init_policy()
        self.prev_action = np.zeros(self.env.num_actions)
        self.start_policy = False

        # standing up
        self.get_logger().info("Standing up")
        self.stand_up = False
        self.stand_up = True

        # start
        self.start_time = time.monotonic()
        self.get_logger().info("Press L2 to start policy")
        self.get_logger().info("Press L1 for emergent stop")
        self.init_buffer = 0
        self.foot_contact_buffer = []
        self.time_hist = []
        self.obs_time_hist = []
        self.angle_hist = []
        self.action_hist = []
        self.dof_hist = []
        self.imu_hist = []
        self.foot_contact_hist = []

    def reindex_urdf2hw(self, vec):
        vec = np.array(vec)
        assert len(vec)==19, "wrong dim for reindex"
        hw_vec = vec[[6, 7, 8, 1, 2, 3, 10, 0, 5, 0, 4, 9, 15, 16, 17, 18, 11, 12, 13, 14]]
        hw_vec[9] = 0
        return hw_vec
        # return vec[[1, 2, 3, 6, 7, 8, 10, 5, 0, 0, 9, 4, 11, 12, 13, 14, 15, 16, 17, 18]] # online doc is wrong!!!

    def reindex_hw2urdf(self, vec):
        vec = np.array(vec)
        assert len(vec)==20, "wrong dim for reindex"
        return vec[[7, 3, 4, 5, 10, 8, 0, 1, 2, 11, 6, 16, 17, 18, 19, 12, 13, 14, 15]]
        # return vec[[8, 0, 1, 2, 11, 7, 3, 4, 5, 10, 6, 12, 13, 14, 15, 16, 17, 18, 19]]
    
    def wrist_timer_callback(self):
        self.wrist_msg_l.position = self.wrist_l.value
        self.wrist_msg_r.position = self.wrist_r.value
        if self.wrist_i % 2 == 1:
            self.wrist_pub.publish(self.wrist_msg_l)
        else:
            self.wrist_pub.publish(self.wrist_msg_r)
        self.wrist_i+=1
        
    ##############################
    # subscriber callbacks
    ##############################

    def joy_stick_cb(self, msg):
        if msg.keys == 2:  # L1: emergency stop
            self.get_logger().info("Emergency stop")
            self.set_gains(np.array([0.0]*HW_DOF),self.env.d_gains)
            self.set_motor_position(q=self.env.default_dof_pos_np)
            # if PLOT_DATA:
            #     ## plots
            #     time_hist = np.array(self.time_hist)
            #     obs_time_hist = np.array(self.obs_time_hist)
            #     print("Plotting angles")
            #     angle_hist = np.array(self.angle_hist)
            #     dof_hist = np.array(self.dof_hist)
            #     for i in range(4):
            #         fig,axs = plt.subplots(3,1,sharex=True)
            #         for j in range(3):
            #             axs[j].plot(time_hist, angle_hist[:,j+3*i], label=f'cmd{j}')
            #             axs[j].plot(obs_time_hist, dof_hist[:,j+3*i], '--', label=f'real{j}')
            #             axs[j].legend()
            #             axs[j].grid(True, which='both', axis='both')
            #             axs[j].minorticks_on()
            #         plt.tight_layout()
            #         plt.savefig(f'../fig/angle_{i}.png')

            #     print("Plotting actions")
            #     for i in range(4):
            #         fig,axs = plt.subplots(3,1,sharex=True)
            #         action_hist = np.array(self.action_hist)
            #         axs[0].plot(time_hist, action_hist[:,0+3*i], label='0')
            #         axs[0].legend()
            #         axs[1].plot(time_hist, action_hist[:,1+3*i], label='1')
            #         axs[1].legend()
            #         axs[2].plot(time_hist, action_hist[:,2+3*i], label='2')
            #         axs[2].legend()
            #         plt.tight_layout()
            #         plt.savefig(f'../fig/action_{i}.png')

            #     print("Plotting imu")
            #     plt.figure()
            #     imu_hist = np.array(self.imu_hist)
            #     plt.plot(obs_time_hist, imu_hist[:,0], label='roll')
            #     plt.plot(obs_time_hist, imu_hist[:,1], label='pitch')
            #     plt.legend()
            #     plt.savefig('../fig/imu.png')

            #     print("Plotting foot contact")
            #     plt.figure()
            #     foot_contact_hist = np.array(self.foot_contact_hist)
            #     for i in range(4):
            #         plt.plot(obs_time_hist, foot_contact_hist[:,i], label=f'foot_{i}')
            #     plt.legend()
            #     plt.savefig('../fig/foot_contact.png')

            #     print("Saving data")
            #     np.savez('../fig/real_data.npz', time_hist=time_hist, obs_time_hist=obs_time_hist, angle_hist=angle_hist, \
            #             action_hist=action_hist, dof_hist=dof_hist, imu_hist=imu_hist, foot_contact_hist=foot_contact_hist)

            raise SystemExit
        if msg.keys == 32:  # L2: start policy
            if self.stand_up:
                self.get_logger().info("Start policy")
                self.start_policy = True
                self.policy_start_time = time.monotonic()
            else:
                self.get_logger().info("Wait for standing up first")

    def lowlevel_state_cb(self, msg: LowState):
        # imu data
        imu_data = msg.imu_state
        print(f"msg tick:{msg.tick/1000}")
        self.msg_tick = msg.tick/1000
        self.roll, self.pitch, self.yaw = imu_data.rpy
        self.obs_ang_vel = np.array(imu_data.gyroscope)*self.env.scale_ang_vel
        self.obs_imu = np.array([self.roll, self.pitch])*self.env.scale_orn

        # termination condition
        r_threshold = abs(self.roll) > 0.5
        p_threshold = abs(self.pitch) > 0.5
        if r_threshold or p_threshold:
            self.get_logger().warning("Roll or pitch threshold reached")
            self.set_gains(np.array([0.0]*HW_DOF),self.env.d_gains)
            self.set_motor_position(q=self.env.default_dof_pos_np)
            raise SystemExit

        # motor data
        self.joint_pos = [msg.motor_state[i].q for i in range(HW_DOF)]
        self.obs_joint_pos = (np.array(self.joint_pos) - self.env.default_dof_pos_np) * self.env.scale_dof_pos
        # self.obs_joint_pos = self.reindex_hw2urdf(obs_joint_pos)

        joint_vel = [msg.motor_state[i].dq for i in range(HW_DOF)]
        self.obs_joint_vel = np.array(joint_vel) * self.env.scale_dof_vel
        # self.obs_joint_vel = self.reindex_hw2urdf(obs_joint_vel)

        # # fetch proprioceptive data
        # obs_buf = np.concatenate((self.obs_imu, self.obs_ang_vel, self.obs_joint_pos, self.obs_joint_vel, self.prev_action, self.reindex_hw2urdf(self.env.default_dof_pos_np)*self.env.scale_dof_pos, np.zeros(5)))  # add reference input TODO
        # obs_buf = torch.tensor(obs_buf,dtype=torch.float, device=self.device).unsqueeze(0)
        # self.env.obs_history_buf = torch.cat([
        #     self.env.obs_history_buf[:, 1:],
        #     obs_buf.unsqueeze(1)
        # ], dim=1)
        

        # if self.start_policy:
        #     self.imu_hist.append(self.obs_imu)
        #     self.foot_contact_hist.append(self.obs_foot_contact)
        #     self.dof_hist.append(self.joint_pos)
        #     self.obs_time_hist.append(time.monotonic()-self.start_time)
        
    ##############################
    # motor commands
    ##############################

    def set_gains(self, kp: np.ndarray, kd: np.ndarray):
        self.kp = kp
        self.kd = kd
        for i in range(HW_DOF):
            self.motor_cmd[i].kp = kp[i]  #*0.5
            self.motor_cmd[i].kd = kd[i]  #*3

    def set_motor_position(
        self,
        q: np.ndarray,
    ):
        for i in range(HW_DOF):
            self.motor_cmd[i].q = q[i]
        self.cmd_msg.motor_cmd = self.motor_cmd.copy()
        self.cmd_msg.crc = get_crc(self.cmd_msg)

    ##############################
    # deploy policy
    ##############################
    def init_policy(self):
        self.get_logger().info("Preparing policy")
        faulthandler.enable()

        # prepare environment
        self.env = H1()

        # load policy
        file_pth = os.path.dirname(os.path.realpath(__file__))
        self.policy = torch.jit.load(os.path.join(file_pth, "0293_policy.pt"), map_location=self.env.device)  #0253
        self.policy.to(self.env.device)
        actions = self.policy(self.env.obs_history_buf.detach())  # first inference takes longer time

        # init p_gains, d_gains, torque_limits
        for i in range(HW_DOF):
            self.motor_cmd[i].q = self.env.default_dof_pos[0][i].item()
            self.motor_cmd[i].dq = 0.0
            self.motor_cmd[i].tau = 0.0
            self.motor_cmd[i].kp = 0.0  # self.env.p_gains[i]  # 30
            self.motor_cmd[i].kd = 0.0  # float(self.env.d_gains[i])  # 0.6
        self.cmd_msg.motor_cmd = self.motor_cmd.copy()
        self.angles = self.env.default_dof_pos


    @torch.no_grad()
    def main_loop(self):
        # keep stand up pose first
        while self.stand_up and not self.start_policy:
            stand_up_time = 2.0
            if time.monotonic() - self.start_time < stand_up_time:
                time_ratio = (time.monotonic() - self.start_time) / stand_up_time
                self.set_gains(kp=time_ratio * self.env.p_gains, kd=time_ratio * self.env.d_gains)
                self.set_motor_position(q=self.env.default_dof_pos_np)
            self.motor_pub.publish(self.cmd_msg)
            rclpy.spin_once(self)
            
            if hasattr(self,'msg_tick'):
                print(f"obs tick:{self.msg_tick}")
        
        cnt = 0
        fps_ck = time.monotonic()
        while rclpy.ok():
            loop_start_time = time.monotonic()
            rclpy.spin_once(self)
            rclpy.spin_once(self)
            if hasattr(self,'msg_tick'):
                print(f"obs tick:{self.msg_tick}")
            if self.start_policy:
                # policy inference
                # warm_up_time = 1.0
                # if time.monotonic() - self.policy_start_time < warm_up_time:
                #     time_ratio = (time.monotonic() - self.policy_start_time) / warm_up_time
                #     self.set_gains(kp=time_ratio * self.env.p_gains, kd=self.env.d_gains)

                # target_jt_hw = self.get_retarget()

                # target_xy_vel = requested_motion['root_velocity']
                # quat = requested_motion['root_rotation']
                # r, p, _ = euler_from_quat(quat)
                # y_vel = requested_motion['root_angularv'][2]
                # target_rpy = np.array([wrap_to_pi(r), wrap_to_pi(p), y_vel])

                # fetch proprioceptive data
                self.obs_joint_vel_ = self.reindex_hw2urdf(self.obs_joint_vel)
                self.obs_joint_pos_ = self.reindex_hw2urdf(self.obs_joint_pos)

                obs_buf = np.concatenate((self.obs_imu, self.obs_ang_vel, self.obs_joint_pos_, self.obs_joint_vel_, self.prev_action, self.reindex_hw2urdf(self.env.default_dof_pos_np)*self.env.scale_dof_pos, np.zeros(5)))  # add reference input TODO
                # obs_buf = np.concatenate((self.obs_imu, self.obs_ang_vel, self.obs_joint_pos_, self.obs_joint_vel_, self.prev_action, self.reindex_hw2urdf(target_jt_hw)*self.env.scale_dof_pos, np.zeros(5)))  # add reference input TODO
                obs_buf = torch.tensor(obs_buf,dtype=torch.float, device=self.device).unsqueeze(0)
                self.env.obs_history_buf = torch.cat([
                    self.env.obs_history_buf[:, 1:],
                    obs_buf.unsqueeze(1)
                ], dim=1)

                if self.init_buffer < 10:
                    self.init_buffer += 1
                    raw_actions = self.policy(self.env.obs_history_buf.detach())
                    self.set_motor_position(q=self.env.default_dof_pos_np)
                    self.motor_pub.publish(self.cmd_msg)
                else:
                    raw_actions = self.policy(self.env.obs_history_buf.detach())
                    # raw_actions[0, -8:]=0
                    self.prev_action = raw_actions.clone().detach().cpu().numpy().squeeze(0)
                    angles = self.reindex_urdf2hw(raw_actions.detach().cpu()[0]) * self.env.scale_action + self.env.default_dof_pos_np
                    angles_out_bound = np.any((angles < self.env.joint_limit_lo) | (angles > self.env.joint_limit_hi))
                    if angles_out_bound:
                        self.get_logger().warning("Output angles out of bound")
                    # self.get_logger().info(f"{self.obs_imu=}")
                    # self.get_logger().info(f"{angles=}")
                    self.angles = np.clip(angles, self.env.joint_limit_lo, self.env.joint_limit_hi)

                    self.get_logger().info(f"inference time: {time.monotonic()-loop_start_time}")
                    # while 0.010-time.monotonic()+loop_start_time > 0:
                    #     pass
                    # time.sleep(max(0.010-time.monotonic()+loop_start_time,0))
                    self.set_motor_position(self.angles)
                    self.motor_pub.publish(self.cmd_msg)
                
            self.get_logger().info(f"loop time: {time.monotonic()-loop_start_time}")
            while 0.020-time.monotonic()+loop_start_time>0:
                pass
            # time.sleep(max(0.0124-time.monotonic()+loop_start_time,0))
            cnt+=1
            if cnt == 100:
                dt = (time.monotonic()-fps_ck)/cnt
                cnt = 0
                fps_ck = time.monotonic()
                print(f"current fps:{dt}")


if __name__ == "__main__":
    rclpy.init(args=None)
    dp_node = DeployNode()
    dp_node.get_logger().info("Deploy node started")
    serial_process = multiprocessing.Process(target=hand_process,args=(dp_node.wrist_l,dp_node.wrist_r,))
    serial_process.daemon = True 
    # serial_process.start()

    dp_node.main_loop()
    rclpy.shutdown()
    # serial_process.join()