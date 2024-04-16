import torch
from PIL import Image
import numpy as np
import cv2
import torchvision

COMPRESSED_DEPTH_DIM = 128
resize_transform = torchvision.transforms.Resize((58, 87), interpolation=torchvision.transforms.InterpolationMode.BICUBIC)

class maintain_depth:
    def __init__(self, compression_model, device):
        self.compression_model  = compression_model
        self.depth_vec = torch.zeros(1, COMPRESSED_DEPTH_DIM * 15).to(device)

        for i in range(15):
            self.depth_vec[:, COMPRESSED_DEPTH_DIM * i: COMPRESSED_DEPTH_DIM * (i+1)] = self.compression_model(torch.zeros(1, 1, 58, 87))

    def update_depth_vec(self, nn_depth_input):
        with torch.no_grad():
            doutput = self.compression_model(nn_depth_input[None, None, :, :])
        self.depth_vec = torch.cat([self.depth_vec[:, COMPRESSED_DEPTH_DIM:], doutput], dim = 1) 

class maintain_prop:
    def __init__(self, device):
        self.prop_vec = torch.zeros(1, 32 * 15).to(device)

    def update_prop_vec(self, prop_input):
        assert(prop_input.shape[0] == 47)
        update_vector = torch.cat([prop_input[:8].unsqueeze(0), prop_input[11:11+24].unsqueeze(0)], dim=1)
        print("Updating prop_q: ", update_vector)
        self.prop_vec = torch.cat([self.prop_vec[:, 32:], update_vector], dim = 1) 
        assert(self.prop_vec.shape[1] == 32 * 15)

def process_udp_data(inp_data):
    float_list = inp_data.decode("utf-8").split(" ")
    decoded_vector = np.array([float(j) for j in float_list])

    return decoded_vector

def encode_depth_extrinsics(depth_output):
    str_depth = " ".join([str("{:.4}".format(j.item())) for j in depth_output])
    str_depth = "START " + str_depth + " END " # This prevents random numbers from coming in
    return str_depth

def process_and_resize(frame):
    frame = np.copy(frame)
    
    # Remove 30 pixels from the left and 20 from the top
    frame = frame[:, 30:-30]
    frame = frame[:-20, :]
    
    frame = cv2.resize(frame, dsize=(87, 58), interpolation=cv2.INTER_CUBIC)
    # frame -= 0.02
    frame = np.clip(frame, 0., 2)
    frame = (frame - 0.) / (2 - 0.) - 0.5
    # frame = np.clip(frame, 0.2, 6)
    # frame = (frame - 0.2) / (6 - 0.2) - 0.5

    return frame

def rescale(x, reference):
    # Match 90th percentiles
    ref_max = np.percentile(reference, 90)
    ref_min = np.percentile(reference, 10)
    x_max = np.percentile(reference, 90)
    x_min = np.percentile(reference, 10)

    x = (x - x_min) / (x_max - x_min)
    x = x * (ref_max - ref_min) + ref_min

    return x 

def crop_depth_image(frame, args):
    if args.crop_left is not None:
        frame = frame[:, args.crop_left:]

    if args.crop_right is not None:
        frame = frame[:, :-args.crop_right]

    if args.crop_top is not None:
        frame = frame[args.crop_top:, :]

    if args.crop_bottom is not None:
        frame = frame[:-args.crop_bottom, :]

    return frame

def normalize_depth_image(frame, args):
    frame = frame - abs(args.normalize_mean) # The abs is for backward compatibility
    frame = frame / args.normalize_std

    return frame

def process_depth_image(frame, args):
    frame = torch.from_numpy(frame).float()

    frame = crop_depth_image(frame, args)
    frame = torch.clip(frame, -args.clip, args.clip)
    frame = resize_transform(frame[None, :]).squeeze()
    frame = normalize_depth_image(frame, args)

    return frame.cuda()
