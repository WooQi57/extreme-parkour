# Hardware deployment #

### Installation ###
- install unitree_sdk: 
    https://github.com/unitreerobotics/unitree_sdk2
    
- install unitree_ros2: 
    https://support.unitree.com/home/en/developer/ROS2_service

- install robotis sdk for the gripper(set USE_GRIPPPER=True in play_hardware.py): 
    https://www.youtube.com/watch?v=E8XPqDjof4U 

```bash
conda create -n doggy python=3.8
conda activate doggy
```

- install nvidia-jetpack

- install torch==1.11.0 torchvision==0.12.0: 
    https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048 
    https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html

```bash
conda activate doggy
pip install "numpy<1.24" pydelatin wandb tqdm opencv-python ipdb pyfqmr flask
```

### Usage ###

1. turn off sport mode: (needed only once after turning on the robot)
```bash
./stand_up_go2 eth0
```

2. run policy:  
```bash
python play_hardware.py
```

joystick commands:
- L1: emergency stop
- L2: start playing policy
- R1: gripper close
- R2: gripper open
- left stick: linear movement
- right stick: yaw & pitch