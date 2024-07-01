# First import the library
import pyrealsense2.pyrealsense2 as rs
import time
from collections import deque
import socket
import numpy as np
import struct
import torch
import sys
import cv2
from utils import *
import os
import matplotlib.pyplot as plt
import multiprocessing as mp
from multiprocessing import Process, Array, Value
from depth_backbone import HardwareVisionNN
from tqdm import tqdm

class DepthSocket():
    def __init__(self):
        self.n_proprio = 55
        self.send_addr = "127.0.0.1"
        self.send_port = 5702
        self.count = 0
        
        self.sock_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_recv.bind(("127.0.0.1", 5701))
        self._prop = Array('f', np.zeros((self.n_proprio, )))
        self.receive_thread = Process(target=self.recv, args=(self._prop,))
        self.receive_thread.daemon = True
        self.receive_thread.start()

    def recv(self, _prop):
        while True:
            data, addr = self.sock_recv.recvfrom(4096)
            data = data.decode("utf-8")
            data_list = data.split(",")
            _prop[:] = np.array(list(map(float, data_list)))[:]
        self.sock_recv.close()
            
    @property
    def proprio(self):
        return np.array(self._prop)

    def send_latent(self, latent_np):
        assert latent_np.shape[0] == 32
        # new_float = float(str(time.time()).split(".")[0][-3:]+"."+str(time.time()).split(".")[1][:3])
        # latent_np[0] = new_float
        string = np.array2string(latent_np, precision=5, separator=',', suppress_small=False, max_line_width=100000)
        string = string[1:-1] # remove []
        string = str(self.count) + "," + string
        msg = string.encode('utf-8')
        self.sock_recv.sendto(msg, (self.send_addr, self.send_port))  
        self.count += 1

depth_sock = DepthSocket()
backbone_model = HardwareVisionNN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_path = "./ckpt/100-33-18500-vision_weight.pt"

ac_state_dict = torch.load(load_path, map_location=device)
backbone_model.depth_encoder.load_state_dict(ac_state_dict['depth_encoder_state_dict'])

# compression_model.to(device)
backbone_model.to(device)

DEPTH_UPDATE_INTERVAL = 0.1
# Create a context object. This object owns the handles to all connected realsense devices
pipeline = rs.pipeline()
config = rs.config()
depth_to_disparity = rs.disparity_transform(True)
disparity_to_depth = rs.disparity_transform(False)
decimation = rs.decimation_filter()
decimation.set_option(rs.option.filter_magnitude, 4)
spatial = rs.spatial_filter()
spatial.set_option(rs.option.filter_magnitude, 5)
spatial.set_option(rs.option.filter_smooth_alpha, 1)
spatial.set_option(rs.option.filter_smooth_delta, 50)
spatial.set_option(rs.option.holes_fill, 3)
hole_filling = rs.hole_filling_filter()
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
profile = pipeline.start(config)
depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
prop_q = deque(maxlen=50)
depth_q = deque(maxlen=15)
depth_output = np.zeros(100)
latent_output = np.zeros(8)
new_data = True
prop_i = np.zeros(47)

# depth_hvec = maintain_depth(compression_model, device)
prop_hvec = maintain_prop(device)

print("starting loop")
frame_received_time = time.time()
os.system("rm -rf frames")
os.system("rm -rf frames_npy")
os.system("rm -rf depth_output")

os.mkdir("frames")
os.mkdir("frames_npy")
os.mkdir("depth_output")
frame_timestamp = -10000
ctr = 0
# depth_output_list = os.listdir("depth_output_sim")
# depth_output_list.sort()
frame = None

filename = "depth_latent.txt"
if os.path.exists(filename):
    os.remove(filename)

depth_images = []

for i in tqdm(range(10000)):
    # Create a pipeline object. This object configures the streaming camera and owns it's handle
    start = time.time()
    ctr += 1
    
    try:
        depth_frame = pipeline.wait_for_frames(timeout_ms = 1)
        frame = depth_frame.get_depth_frame()
        frame_timestamp = time.time()
    except:
        pass
    
    if time.time() - frame_received_time >= DEPTH_UPDATE_INTERVAL:
        if time.time() - frame_timestamp >= 5e-2: 
            print("Stale frame ", time.time() - frame_timestamp)
            if frame != None:
                plt.imsave("frames/stale_{}.png".format("{:0>5d}".format(ctr)), process_and_resize(np.asanyarray(frame.get_data()) * depth_scale)+0.5)
        else:
            start = time.time()
            # print("Frame_received_time, ", time.time() - frame_received_time)
            frame_received_time = time.time()
            
            frame = decimation.process(frame)
            frame = depth_to_disparity.process(frame)
            frame = spatial.process(frame)
            #frame = temporal.process(frame)
            frame = disparity_to_depth.process(frame)
            frame = hole_filling.process(frame)

            np_frame = (np.asanyarray(frame.get_data()) * depth_scale)
            low_res = process_and_resize(np_frame)
            depth_images.append(low_res)
            depth_image = torch.from_numpy(np.array(low_res)).float().to(device)
            # fake_depth_image = np.load("frames_npy/np_frame_50.npy")
            # print(fake_depth_image.shape)
            # depth_image = torch.from_numpy(fake_depth_image).float().to(device)
            obs = torch.from_numpy(depth_sock.proprio[None, :]).float().to(device)
            # print(obs)
            depth_latent_np = backbone_model(obs, depth_image[None,:,:]).detach().cpu().numpy().squeeze()
            
            frame_send_time = time.time()
            duration = frame_send_time - start
            if duration <= 0.09:
                time.sleep(0.09 - duration)
            # print("Image duration, ", time.time()-start)
            depth_sock.send_latent(depth_latent_np)

            # cv2.namedWindow("name", cv2.WINDOW_NORMAL)
            # cv2.imshow("name", low_res+0.5)
            # cv2.waitKey(1)
            # plt.imsave("debug.png", low_res+0.5)
            
            # plt.imsave("frames/np_frame_{}.png".format("{:0>5d}".format(ctr)), low_res+0.5, vmin=0, vmax=1)
            # np.save("frames_npy/depth_output_{}.txt".format("{:0>5d}".format(ctr)), low_res)
            # print(f"img processing time: {time.time()-start}")
            
            # with open('depth_latent.txt', 'a') as f:
            #     data = np.concatenate(((depth_sock.count-1,), depth_latent_np)).reshape(1, 35)
            #     np.savetxt(f, data, fmt='%f')
            


plt.imsave("first_frame.png", depth_images[0], vmin=-2, vmax=2, cmap='gray')

# Linear transformation to convert values to 0-255 range
frames = [(frame + 0.5) * 255 for frame in depth_images]

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 10.0, (frames[0].shape[1], frames[0].shape[0]), isColor=False)

# Write each frame to the video
for frame in frames:
    out.write(np.uint8(frame))

# Release the VideoWriter object
out.release()