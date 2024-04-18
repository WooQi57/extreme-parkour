# First import the library
import pyrealsense2.pyrealsense2 as rs
# import pyrealsense2 as rs
import time
from collections import deque
import socket
import numpy as np
import struct
import sys
import matplotlib.pyplot as plt
import cv2
import os

def process_and_resize(frame):
    frame = np.copy(frame)
    
    # Remove 30 pixels from the left and 20 from the top
    frame = frame[:, 30:-30]
    frame = frame[11:-21, :]
    
    frame = cv2.resize(frame, dsize=(87, 58), interpolation=cv2.INTER_CUBIC)
    frame = np.clip(frame, 0., 2)
    frame = (frame - 0.) / (2 - 0.) - 0.5

    return frame

# Create a context object. This object owns the handles to all connected realsense devices
pipeline = rs.pipeline()
config = rs.config()
# filter stuff
depth_to_disparity = rs.disparity_transform(True)
disparity_to_depth = rs.disparity_transform(False)
decimation = rs.decimation_filter()
decimation.set_option(rs.option.filter_magnitude, 4)
spatial = rs.spatial_filter()
spatial.set_option(rs.option.filter_magnitude, 5)
spatial.set_option(rs.option.filter_smooth_alpha, 1)
spatial.set_option(rs.option.filter_smooth_delta, 50)
spatial.set_option(rs.option.holes_fill, 3)
temporal = rs.temporal_filter()
hole_filling = rs.hole_filling_filter()
# start
config.enable_stream(rs.stream.depth, 640, 360, rs.format.z16, 30)
profile = pipeline.start(config)
depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

sensor_dep = profile.get_device().first_depth_sensor()
print("Trying to set Exposure")
exp = sensor_dep.get_option(rs.option.exposure)
print("exposure =  ", exp)
print("Setting exposure to new value")
exp = sensor_dep.set_option(rs.option.exposure, 10000)
exp = sensor_dep.get_option(rs.option.exposure)
print("New exposure = ", exp)

# plt.figure(figsize=(24, 12), dpi=80)

# ax1 = plt.subplot(1,2,1)
# ax2 = plt.subplot(1,2,2)
# ax3 = plt.subplot(1,3,3)

# im1 = None
# im2 = None
# im3 = None

# plt.ion()
os.system("rm -rf frames")
os.system("rm -rf frames_npy")
# os.system("rm -rf depth_output")

os.mkdir("frames")
os.mkdir("frames_npy")
# os.mkdir("depth_output")

for gitr in range(10000):
    # Create a pipeline object. This object configures the streaming camera and owns it's handle
    start = time.time()
    try:
        depth_frame = pipeline.wait_for_frames(timeout_ms = 100)
        frame = depth_frame.get_depth_frame()
    except:
        frame = None

    # try:
    depth_frame = pipeline.wait_for_frames(timeout_ms = 500)
    frame = depth_frame.get_depth_frame()
    
    frame = decimation.process(frame)
    frame = depth_to_disparity.process(frame)
    frame = spatial.process(frame)
    frame = disparity_to_depth.process(frame)
    frame = hole_filling.process(frame)
    frame = temporal.process(frame)
    np_frame = (np.asanyarray(frame.get_data()) * depth_scale)
    low_res = process_and_resize(np_frame)
    print(low_res.min(), low_res.max())
    np.save("frames_npy/np_frame_{}.npy".format(gitr), low_res)
    # plt.imsave("frames/np_frame_{}.png".format(gitr), np_frame, vmin=-2, vmax=2)
    plt.imsave("frames/np_frame_{}.png".format(gitr), low_res, vmin=-2, vmax=2, cmap='gray')
    # plt.imsave('gray_image.png', gray_image, cmap='gray')
    # cv2.imwrite("frames/np_frame_{}.png".format(gitr), low_res)


        # color_frame = depth_frame.get_color_frame()
        # color_frame = np.asanyarray(color_frame.get_data())
        
        # import pdb; pdb.set_trace()
        # if im1 is None:
        #    ax1.imshow(np_frame, cmap="inferno", vmin=0, vmax=1)
        # else:
        #    ax1.set_data(np_frame, cmap="inferno", vmin=0, vmax=1)

        # if im2 is None:
        #    ax2.imshow(low_res, cmap="inferno", vmin=-3, vmax=2)
        # else:
        #    ax2.set_data(low_res, cmap="inferno", vmin=-3, vmax=2)
           
        # if im3 is None:
        #    ax3.imshow(color_frame)
        # else:
        #   ax3.set_data(color_frame)
        
        # plt.pause(0.001)
        
        # ctr += 1
    # except:
    #     print("Frame not found")
    #     continue

pipeline.stop()
