import os
import sys
import multiprocessing

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from crc_module import get_crc

import numpy as np
import torch
import faulthandler
import matplotlib.pyplot as plt
from hardware import Hardware

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
import subprocess
import socket
from multiprocessing import Process, Array, Value

POS_STOP_F = 2.146e9
VEL_STOP_F = 16000.0
LEG_DOF = 12
SDK_DOF = 20

CONTACT_THRESHOLD = [26,29,25,27] #[16,19,15,17]+5
WALK_STRAIGHT = False
USE_TIMER = False
PLOT_DATA = False
USE_GRIPPPER = True
NO_MOTOR = False

if USE_GRIPPPER:
    from dynamixel_sdk_custom_interfaces.msg import SetPosition

class DepthSocket():
    def __init__(self, fake_depth_latent):
        self.send_addr = "127.0.0.1"
        self.send_port = 5701

        self.sock_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_recv.bind(("127.0.0.1", 5702))
        self._depth_latent = Array('f', fake_depth_latent)
        self.receive_thread = Process(target=self.recv, args=(self._depth_latent,))
        self.receive_thread.daemon = True
        self.receive_thread.start()

    def recv(self, _depth_latent):
        while True:
            data, addr = self.sock_recv.recvfrom(4096)
            data = data.decode("utf-8")
            data_list = data.split(",")
            _depth_latent[:] = np.array(list(map(float, data_list)))[1:]
            
    @property
    def depth_latent(self):
        return np.array(self._depth_latent)

    def send_prop(self, prop_np):
        assert prop_np.shape[0] == 55
        string = np.array2string(prop_np, precision=4, separator=',', suppress_small=False, max_line_width=100000)
        string = string[1:-1] # remove []
        msg = string.encode('utf-8')
        self.sock_recv.sendto(msg, (self.send_addr, self.send_port))  


class DeployNode(Node):
    def __init__(self):
        super().__init__("deploy_node")  # type: ignore

        # init udp
        self.depth_sock = DepthSocket(np.zeros(32))

        # init subcribers
        self.joy_stick_sub = self.create_subscription(
            WirelessController, "wirelesscontroller", self.joy_stick_cb, 1
        )
        self.joy_stick_sub  # prevent unused variable warning
        self.lowlevel_state_sub = self.create_subscription(
            LowState, "lowstate", self.lowlevel_state_cb, 1
        )  # "/lowcmd" or  "lf/lowstate" (low frequencies)
        self.lowlevel_state_sub  # prevent unused variable warning

        self.low_state = LowState()
        self.joint_pos = np.zeros(LEG_DOF)
        self.joint_vel = np.zeros(LEG_DOF)

        # init publishers
        if USE_GRIPPPER:
            self.gripper_pub = self.create_publisher(SetPosition, "/set_position", 1)
            self.gripper_timer = self.create_timer(1.0 / 100, self.gripper_timer_callback)
            self.gripper_msg = SetPosition()
            self.gripper_msg.id = 1
            self.gripper_msg.position = 1500

        self.motor_pub = self.create_publisher(LowCmd, "lowcmd", 1)
        self.motor_pub_freq = 50
        if USE_TIMER:
            self.motor_timer = self.create_timer(1.0 / self.motor_pub_freq, self.motor_timer_callback)
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
        self.init_policy()
        self.policy_freq = 50
        if USE_TIMER:
            self.policy_timer = self.create_timer( 1.0 / self.policy_freq, self.policy_timer_callback)
        self.obs_proprio = np.zeros(self.env.cfg.env.n_proprio)
        self.prev_action = np.zeros(self.env.cfg.env.num_actions)
        self.command = np.zeros(self.env.cfg.commands.num_commands)
        if WALK_STRAIGHT:
            self.command[0] = 0.6
            self.command[2] = 0.25*0

        # init gripper
        self.gripper_cmd = 1.0  # TODO: gripper cmd 1:close; -1:open
        self.start_policy = False

        # standing up
        self.get_logger().info("Standing up")
        self.stand_up = False
        # subprocess.run(["./stand_up_go2 eth0"], shell=True)
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


    ##############################
    # subscriber callbacks
    ##############################

    def joy_stick_cb(self, msg):
        if msg.keys == 2:  # L1: emergency stop
            self.get_logger().info("Emergency stop")
            self.set_gains(0.0,0.6)
            if PLOT_DATA:
                ## plots
                time_hist = np.array(self.time_hist)
                obs_time_hist = np.array(self.obs_time_hist)
                print("Plotting angles")
                angle_hist = np.array(self.angle_hist)
                dof_hist = np.array(self.dof_hist)
                for i in range(4):
                    fig,axs = plt.subplots(3,1,sharex=True)
                    for j in range(3):
                        axs[j].plot(time_hist, angle_hist[:,j+3*i], label=f'cmd{j}')
                        axs[j].plot(obs_time_hist, dof_hist[:,j+3*i], '--', label=f'real{j}')
                        axs[j].legend()
                        axs[j].grid(True, which='both', axis='both')
                        axs[j].minorticks_on()
                    plt.tight_layout()
                    plt.savefig(f'../fig/angle_{i}.png')

                print("Plotting actions")
                for i in range(4):
                    fig,axs = plt.subplots(3,1,sharex=True)
                    action_hist = np.array(self.action_hist)
                    axs[0].plot(time_hist, action_hist[:,0+3*i], label='0')
                    axs[0].legend()
                    axs[1].plot(time_hist, action_hist[:,1+3*i], label='1')
                    axs[1].legend()
                    axs[2].plot(time_hist, action_hist[:,2+3*i], label='2')
                    axs[2].legend()
                    plt.tight_layout()
                    plt.savefig(f'../fig/action_{i}.png')

                print("Plotting imu")
                plt.figure()
                imu_hist = np.array(self.imu_hist)
                plt.plot(obs_time_hist, imu_hist[:,0], label='roll')
                plt.plot(obs_time_hist, imu_hist[:,1], label='pitch')
                plt.legend()
                plt.savefig('../fig/imu.png')

                print("Plotting foot contact")
                plt.figure()
                foot_contact_hist = np.array(self.foot_contact_hist)
                for i in range(4):
                    plt.plot(obs_time_hist, foot_contact_hist[:,i], label=f'foot_{i}')
                plt.legend()
                plt.savefig('../fig/foot_contact.png')

                print("Saving data")
                np.savez('../fig/real_data.npz', time_hist=time_hist, obs_time_hist=obs_time_hist, angle_hist=angle_hist, \
                        action_hist=action_hist, dof_hist=dof_hist, imu_hist=imu_hist, foot_contact_hist=foot_contact_hist)

            raise SystemExit
        if msg.keys == 32:  # L2: start policy
            if self.stand_up:
                self.get_logger().info("Start policy")
                self.start_policy = True
            else:
                self.get_logger().info("Wait for standing up first")
        

        if msg.keys == 1 and USE_GRIPPPER:  # R1 close gripper
            self.gripper_msg.position = 1000

        if msg.keys == 16 and USE_GRIPPPER: # R2 open gripper
            self.gripper_msg.position = 1500

        cmd_vx = msg.ly * 0.8 if msg.ly > 0 else msg.ly * 0.3
        cmd_vy = msg.lx * -0.5  # 0.5
        cmd_delta_yaw = msg.rx * -0.8  # 0.5 1  0.6
        cmd_pitch = msg.ry * 0.7  # 0.7
        if not WALK_STRAIGHT:
            self.command = np.array([cmd_vx, cmd_vy, cmd_delta_yaw, cmd_pitch])


    def lowlevel_state_cb(self, msg: LowState):
        # imu data
        imu_data = msg.imu_state
        # print(f"msg tick:{msg.tick/1000}")
        self.msg_tick = msg.tick/1000
        self.roll, self.pitch, self.yaw = imu_data.rpy
        self.obs_ang_vel = np.array(imu_data.gyroscope)*self.env.obs_scales.ang_vel
        self.obs_imu = np.array([self.roll, self.pitch])

        # motor data
        self.joint_pos = [msg.motor_state[i].q for i in range(LEG_DOF)]
        obs_joint_pos = (np.array(self.joint_pos) - self.env.default_dof_pos_np) * self.env.obs_scales.dof_pos
        self.obs_joint_pos = np.append(obs_joint_pos, -0.02*(self.gripper_cmd+1) * self.env.obs_scales.dof_pos)  # -0.04 close; 0 open

        joint_vel = [msg.motor_state[i].dq for i in range(LEG_DOF)]
        obs_joint_vel = np.array(joint_vel) * self.env.obs_scales.dof_vel
        self.obs_joint_vel = np.append(obs_joint_vel, 0.0)
        # foot force data
        # policy feet names:['FR_foot', 'FL_foot', 'RR_foot', 'RL_foot']
        # robot feet names:['FR_foot', 'FL_foot', 'RR_foot', 'RL_foot']

        foot_force = [msg.foot_force[foot_id] for foot_id in range(4)]
        if len(self.foot_contact_buffer ) < 10:
            self.foot_contact_buffer.append(foot_force)
        else:
            self.foot_contact_buffer.pop(0)
            self.foot_contact_buffer.append(foot_force)
        
        foot_force = np.sum((np.array(self.foot_contact_buffer) > np.array(CONTACT_THRESHOLD)).astype(float), axis=0)
        self.obs_foot_contact = (foot_force > 5) - 0.5
        # self.get_logger().info(f"{self.obs_foot_contact=}, {foot_force=}")
        if self.start_policy:
            self.imu_hist.append(self.obs_imu)
            self.foot_contact_hist.append(self.obs_foot_contact)
            self.dof_hist.append(self.joint_pos)
            self.obs_time_hist.append(time.monotonic()-self.start_time)

        
    ##############################
    # motor commands
    ##############################

    def set_gains(self, kp: float, kd: float):
        self.kp = kp
        self.kd = kd
        for i in range(LEG_DOF):
            self.motor_cmd[i].kp = kp
            self.motor_cmd[i].kd = kd

    def gripper_timer_callback(self):
        self.gripper_pub.publish(self.gripper_msg)

    def motor_timer_callback(self):
        if self.stand_up:
            self.cmd_msg.crc = get_crc(self.cmd_msg)
            if not NO_MOTOR:
                self.motor_pub.publish(self.cmd_msg)

    def set_motor_position(
        self,
        q: np.ndarray,
    ):
        for i in range(LEG_DOF):
            self.motor_cmd[i].q = float(q[i])
        self.cmd_msg.motor_cmd = self.motor_cmd.copy()

    ##############################
    # deploy policy
    ##############################
    def init_policy(self):
        self.get_logger().info("Preparing policy")
        faulthandler.enable()
        # rec_data = np.load("../fig/rec_data.npz")
        # self.rec_cmd = rec_data["angle_hist"]
        self.replay_i = 0

        # prepare environment
        self.env = Hardware()

        # load policy
        self.obs = torch.zeros(1, self.env.obs_buf_dim, device=self.env.device)
        file_pth = os.path.dirname(os.path.realpath(__file__))
        self.policy = torch.jit.load(os.path.join(file_pth, "./ckpt/100-33-18500-base_jit.pt"), map_location=self.env.device)#101-95 000-94-8000 100-92-19500- 100-91-7000 100-03 
        self.policy.to(self.env.device)
        actions = self.policy(self.obs, torch.zeros(1,32, device=self.env.device))  # first inference takes longer time

        # init p_gains, d_gains, torque_limits, default_dof_pos_all
        for i in range(LEG_DOF):
            self.motor_cmd[i].q = self.env.default_dof_pos[0][i].item()
            self.motor_cmd[i].dq = 0.0
            self.motor_cmd[i].tau = 0.0
            self.motor_cmd[i].kp = 0.0  # self.env.p_gains[i]  # 30
            self.motor_cmd[i].kd = 0.0  # float(self.env.d_gains[i])  # 0.6
        self.cmd_msg.motor_cmd = self.motor_cmd.copy()

        # init angles:([-0.1000,  0.8000, -1.5000,  0.1000,  0.8000, -1.5000, -0.1000,  1.0000,
        # -1.5000,  0.1000,  1.0000, -1.5000], device='cuda:0')
        self.angles = self.env.default_dof_pos[0].clone().unsqueeze(0)


    @torch.no_grad()
    def main_loop(self):
        # keep stand up pose first
        while self.stand_up and not self.start_policy:
            # self.set_motor_position(q=self.env.default_dof_pos[0].cpu().detach().numpy())
            # self.set_gains(kp=float(self.env.p_gains[0]), kd=float(self.env.d_gains[0]))

            stand_kp = 40
            stand_kd = 0.6
            stand_up_time = 2.0

            if time.monotonic() - self.start_time < stand_up_time:
                time_ratio = (time.monotonic() - self.start_time) / stand_up_time
                self.set_gains(kp=time_ratio * stand_kp, kd=time_ratio * stand_kd)
                self.set_motor_position(
                    q=self.env.default_dof_pos[0].cpu().detach().numpy()
                )
            elif time.monotonic() - self.start_time < stand_up_time * 2:
                pass
            elif time.monotonic() - self.start_time < stand_up_time * 3:
                time_ratio = (
                    time.monotonic() - self.start_time - stand_up_time * 2
                ) / stand_up_time
                kp = (1 - time_ratio) * stand_kp + time_ratio * float(self.env.p_gains[0])
                kd = (1 - time_ratio) * stand_kd + time_ratio * float(self.env.d_gains[0])
                self.set_gains(kp=kp, kd=kd)
            elif time.monotonic() - self.start_time < stand_up_time * 3 + 0.005:
                self.get_logger().info(f"ready to start policy")

            self.cmd_msg.crc = get_crc(self.cmd_msg)
            if not NO_MOTOR:
                self.motor_pub.publish(self.cmd_msg)
            rclpy.spin_once(self)
            
            # if hasattr(self,'msg_tick'):
            #     print(f"obs tick:{self.msg_tick}")
        
        cnt = 0
        fps_ck = time.monotonic()
        self.obs_tick = self.msg_tick
        while rclpy.ok():
            start_time = time.monotonic()
            rclpy.spin_once(self)
            while self.msg_tick == self.obs_tick:
                print(f"respin. Only found:{self.msg_tick}")
                rclpy.spin_once(self)

            self.obs_tick = self.msg_tick
            print(f"obs tick:{self.obs_tick}")

            if self.start_policy:
                # policy inference
                self.delta_yaw = np.array([self.command[2]])
                self.delta_pitch = np.array([self.command[3] - self.pitch]) #np.array([0])
                self.obs_proprio = np.concatenate(([-1], self.obs_ang_vel, self.obs_imu, [0], self.delta_pitch, self.command, \
                                                self.obs_joint_pos, self.obs_joint_vel, self.prev_action, 0*self.obs_foot_contact))
                # self.obs_proprio = np.array([-1.00000000e+00, -1.59789645e-03, -1.59789645e-03, -4.26105736e-03,
                #     -1.46267600e-02,  1.57248974e-02,  0.00000000e+00, -1.57248974e-02,
                #         8.00000000e-01,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                #         4.34341207e-02,  4.78153825e-02, -1.23036981e-01, -3.49564403e-02,
                #     -4.35274839e-03, -1.28865957e-01,  6.48352653e-02,  2.95356512e-02,
                #     -5.55679798e-02, -6.63188472e-02,  1.14903450e-02, -1.15091085e-01,
                #     -4.00000000e-02,  5.81328571e-04, -1.16265714e-03,  1.01100618e-04,
                #         1.35643333e-03, -7.75104761e-04,  1.01100625e-03, -2.32531428e-03,
                #     -1.55020952e-03,  4.04402474e-04, -1.35643333e-03, -3.87552381e-04,
                #     -7.07704341e-04,  0.00000000e+00,  2.94919276e+00,  3.70014608e-01,
                #         8.49897194e+00,  7.31491148e-02,  3.46814066e-01, -9.27761197e-02,
                #     -6.81187987e-01, -8.99115682e-01, -2.48833990e+00, -1.18636012e-01,
                #         2.56737685e+00,  1.33327454e-01, -5.43424034e+01,  0.00000000e+00,
                #     -0.00000000e+00,  0.00000000e+00,  0.00000000e+00])
                self.obs = self.env.compute_observations(self.obs_proprio)

                # depth_latent = torch.tensor([[ 0.5534,  0.9367, -0.5928, -0.9057,  0.9920,  0.9834,  0.9989,  0.9913,
                #     -0.6760,  0.9490,  0.6417, -0.9701, -0.7825,  0.8949, -0.9638, -0.9998,
                #     0.9811, -0.5608, -0.8833,  0.5882,  0.0769,  0.9788,  0.3382, -0.9965,
                #     0.9914, -0.3694,  0.0344,  0.8892, -0.9260,  0.9578,  0.8858, -0.6454]],device=self.env.device)
                depth_latent = torch.from_numpy(self.depth_sock.depth_latent).unsqueeze(0).float()
                depth_latent = depth_latent.to(self.env.device)

                if self.init_buffer < 10:
                    self.init_buffer += 1
                    actions = self.policy(self.obs.detach(), depth_latent) # warmup for the policy
                    self.set_motor_position(q=self.env.default_dof_pos_np)
                    if not NO_MOTOR:
                        self.motor_pub.publish(self.cmd_msg)
                else:
                    actions = self.policy(self.obs.detach(), depth_latent) #self.policy(self.obs.detach())
                    self.depth_sock.send_prop(self.obs_proprio)
                    self.prev_action = actions.clone().detach().cpu().numpy().squeeze(0)
                    self.angles = self.env.compute_angle(actions)
                    self.time_hist.append(time.monotonic()-self.start_time)
                    self.angle_hist.append(self.angles[0].tolist())
                    self.action_hist.append(actions[0].tolist())
                    self.get_logger().info(f"inference time: {time.monotonic()-start_time}")
                    if time.monotonic()-start_time > 0.1:
                        self.get_logger().info("Stop due to long inference time or disconnection")
                        self.set_gains(0.0,0.6)
                        raise SystemExit
                    
                    while 0.010-time.monotonic()+start_time>0:
                        pass

                    self.set_motor_position(self.angles.cpu().detach().numpy()[0])
                    self.cmd_msg.crc = get_crc(self.cmd_msg)
                    if not NO_MOTOR:
                        self.motor_pub.publish(self.cmd_msg)

            
            self.get_logger().info(f"loop time: {time.monotonic()-start_time}")
            while 0.020-time.monotonic()+start_time>0:
                pass
            # time.sleep(max(0.0124-time.monotonic()+loop_start_time,0))
            cnt+=1
            if cnt == 100:
                dt = (time.monotonic()-fps_ck)/cnt
                cnt = 0
                fps_ck = time.monotonic()
                print(f"current fps:{dt}")

    @torch.no_grad()
    def policy_timer_callback(self):
        # keep stand up pose first
        if self.stand_up and not self.start_policy:
            # self.set_motor_position(q=self.env.default_dof_pos[0].cpu().detach().numpy())
            # self.set_gains(kp=float(self.env.p_gains[0]), kd=float(self.env.d_gains[0]))

            stand_kp = 40
            stand_kd = 0.6
            stand_up_time = 2.0

            if time.monotonic() - self.start_time < stand_up_time:
                time_ratio = (time.monotonic() - self.start_time) / stand_up_time
                self.set_gains(kp=time_ratio * stand_kp, kd=time_ratio * stand_kd)
                self.set_motor_position(
                    q=self.env.default_dof_pos[0].cpu().detach().numpy()
                )
            elif time.monotonic() - self.start_time < stand_up_time * 2:
                pass
            elif time.monotonic() - self.start_time < stand_up_time * 3:
                time_ratio = (
                    time.monotonic() - self.start_time - stand_up_time * 2
                ) / stand_up_time
                kp = (1 - time_ratio) * stand_kp + time_ratio * float(self.env.p_gains[0])
                kd = (1 - time_ratio) * stand_kd + time_ratio * float(self.env.d_gains[0])
                self.set_gains(kp=kp, kd=kd)
            
        elif self.start_policy:
            # self.set_motor_position(self.angles.cpu().detach().numpy()[0])
            self.get_logger().info(f"policy timer cb")
            # policy inference
            start_time = time.monotonic()
            self.delta_yaw = np.array([self.command[2]])
            self.delta_pitch = np.array([self.command[3] - self.pitch]) #np.array([0])
            self.obs_proprio = np.concatenate((self.obs_ang_vel, self.obs_imu, 0, self.delta_pitch, self.command, \
                                              self.obs_joint_pos, self.obs_joint_vel, self.prev_action, 0*self.obs_foot_contact))
            
            self.obs = self.env.compute_observations(self.obs_proprio)
            # self.get_logger().info(f"obs_proprio: {self.obs_proprio}")

            if self.init_buffer < 10:
                self.init_buffer += 1
            else:
                actions = self.policy(self.obs.detach())
                self.prev_action = actions.clone().detach().cpu().numpy().squeeze(0)
                self.angles = self.env.compute_angle(actions)
                self.time_hist.append(time.monotonic()-self.start_time)
                self.angle_hist.append(self.angles[0].tolist())
                self.action_hist.append(actions[0].tolist())
                self.get_logger().info(f"inference time: {time.monotonic()-start_time}")
                # self.get_logger().info(f"angles: {self.angles[0]}")
                self.set_motor_position(self.angles.cpu().detach().numpy()[0])

    def test_response_time(self):
        stand_up_time = 2.0
        if time.monotonic() - self.start_time < stand_up_time:
            self.set_gains(kp=float(self.env.p_gains[0]), kd=float(self.env.d_gains[0]))
            self.set_motor_position(q=self.env.default_dof_pos[0].cpu().detach().numpy())
        elif self.start_policy:
            self.set_motor_position(self.rec_cmd[self.replay_i])
            self.cmd_msg.crc = get_crc(self.cmd_msg)
            if not NO_MOTOR:
                self.motor_pub.publish(self.cmd_msg)

            self.time_hist.append(time.monotonic()-self.start_time)
            self.angle_hist.append(self.rec_cmd[self.replay_i])
            self.replay_i = (self.replay_i + 1) % len(self.rec_cmd)

if __name__ == "__main__":
    # subprocess.run(["./stand_up_go2 eth0"], shell=True)

    rclpy.init(args=None)
    dp_node = DeployNode()
    dp_node.get_logger().info("Deploy node started")
    if USE_TIMER:
        rclpy.spin(dp_node)
    else:
        dp_node.main_loop()
    rclpy.shutdown()