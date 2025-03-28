import argparse
import copy
import datetime
import io
import os
import threading
import time
from typing import Any, Dict

import cv2
import numpy as np
import pyrealsense2 as rs
import yaml
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import YamlConfig
from deoxys.utils.config_utils import get_default_controller_config
from deoxys.utils.log_utils import get_deoxys_example_logger
from robot.controllers.oculus import BimanualVRController
from scipy.spatial.transform import Rotation as R

logger = get_deoxys_example_logger()

# Constants
CONTROL_HZ = 20
MAX_GRIPPER_WIDTH = 0.08  # Maximum width of the gripper (in meters)

def flatten_dict(d: Dict, separator: str = ".") -> Dict:
    """Flatten a nested dictionary structure."""
    flat_dict = {}
    _flatten_dict_helper(flat_dict, d, "", separator=separator)
    return flat_dict

def _flatten_dict_helper(flat_dict: Dict, value: Any, prefix: str, separator: str = ".") -> None:
    """Helper function for flatten_dict."""
    if isinstance(value, dict):
        for k in value.keys():
            assert isinstance(k, str), "Can only flatten dicts with str keys"
            _flatten_dict_helper(flat_dict, value[k], prefix + separator + k, separator=separator)
    else:
        flat_dict[prefix[1:]] = value

def nest_dict(d: Dict, separator: str = ".") -> Dict:
    """Convert a flattened dictionary back to a nested structure."""
    nested_d = dict()
    for key in d.keys():
        key_parts = key.split(separator)
        current_d = nested_d
        while len(key_parts) > 1:
            if key_parts[0] not in current_d:
                current_d[key_parts[0]] = dict()
            current_d = current_d[key_parts[0]]
            key_parts.pop(0)
        current_d[key_parts[0]] = d[key]  # Set the value
    return nested_d

def append(lst, item):
    """Append items to a nested list structure."""
    if isinstance(lst, dict):
        assert isinstance(item, dict)
        for k in item.keys():
            append(lst[k], item[k])
    else:
        lst.append(item)

def save_episode(data: Dict, path: str, enforce_length: bool = True) -> None:
    """Save episode data to a compressed numpy file."""
    # Flatten the dict for saving as a numpy array
    data = flatten_dict(data)
    
    # Format everything into numpy in case it was saved as a list
    for k in data.keys():
        if isinstance(data[k], np.ndarray) and not data[k].dtype == np.float64:
            continue
        elif isinstance(data[k], list):
            # Check if the list is empty
            if not data[k]:
                # Handle empty list - convert to empty numpy array
                # You can choose an appropriate dtype based on what you expect
                data[k] = np.array([], dtype=np.float32)
                continue
                
            first_value = data[k][0]
            if isinstance(first_value, (np.float64, float)):
                dtype = np.float32  # Detect and convert out float64
            elif isinstance(first_value, (np.ndarray, np.generic)):
                dtype = first_value.dtype
            elif isinstance(first_value, bool):
                dtype = np.bool_
            elif isinstance(first_value, int):
                dtype = np.int64
            else:
                dtype = None
            data[k] = np.array(data[k], dtype=dtype)
        else:
            raise ValueError("Unsupported type passed to `save_data`.")
    
    if enforce_length:
        print(list(map(len, data.values())))
    
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **data)
        bs.seek(0)
        with open(path, "wb") as f:
            f.write(bs.read())


def get_ee_pose(robot_interface):
    """Get end effector pose from robot interface."""
    eef_rot, eef_pos = robot_interface.last_eef_rot_and_pos
    if eef_rot is None or eef_pos is None:
        logger.warning("End effector rotation or position is None, skipping iteration.")
        return None
    
    r = R.from_matrix(eef_rot)
    eef_euler = r.as_euler('xyz', degrees=False)
    pose = np.concatenate((eef_pos.flatten(), eef_euler))
    return pose

def get_gripper_position(robot_interface):
    """Get gripper position from robot interface."""
    last_gripper = robot_interface.last_gripper_q
    if last_gripper is None:
        logger.warning("Last gripper state is None, skipping iteration.")
        return None
    
    gripper = 1 - (last_gripper.item() / MAX_GRIPPER_WIDTH)  # 0 ~ 0.08 -> 0 ~ 1.0
    return gripper

def reset_joint(robot_interface):
    """Reset robot to a default joint configuration."""
    controller_type = "JOINT_POSITION"
    
    # Golden resetting joints
    reset_joint_positions = [
        0.09162008114028396,
        -0.24826458111314524,
        -0.01990020486871322,
        -2.1732269941140346,
        -0.01307073642274261,
        2.00396583422025,
        0.8480939705504309,
    ]

    # Add small random variation to joint positions
    reset_joint_positions = [
        e + np.clip(np.random.randn() * 0.005, -0.005, 0.005)
        for e in reset_joint_positions
    ]
    
    action = reset_joint_positions + [-1.0]
    
    while True:
        if len(robot_interface._state_buffer) > 0:
            difference = np.max(
                np.abs(
                    np.array(robot_interface._state_buffer[-1].q)
                    - np.array(reset_joint_positions)
                )
            )
            if difference < 1e-3:
                break
        
        robot_interface.control(
            controller_type=controller_type,
            action=action,
            controller_cfg=YamlConfig("/home/prior/deoxys_control/deoxys/config/joint-position-controller.yml").as_easydict(),
        )

class RealSenseCamera:
    """Class to handle RealSense camera operations."""
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(self.config)
    
    def get_frame(self):
        """Get the latest color frame from the camera."""
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return None
        return np.asanyarray(color_frame.get_data())
    
    def stop(self):
        """Stop the camera pipeline."""
        self.pipeline.stop()

class MultiRealSenseCamera:
    """Class to handle multiple RealSense cameras."""
    def __init__(self, serial_numbers=None):
        """
        Initialize multiple RealSense cameras.
        
        Args:
            serial_numbers: List of camera serial numbers to use specific cameras.
                           If None, will use all available cameras (up to 3).
        """
        self.pipelines = []
        self.configs = []
        self.serial_numbers = serial_numbers
        
        # Find all available RealSense devices
        ctx = rs.context()
        devices = [device for device in ctx.devices]
        
        if len(devices) == 0:
            raise RuntimeError("No RealSense cameras detected")
        
        if len(devices) < 3:
            logger.warning(f"Only {len(devices)} RealSense cameras detected, but 3 are requested")
        
        # Initialize cameras (use specific ones if serial numbers provided, otherwise use all)
        cam_count = 0
        for device in devices:
            device_serial = device.get_info(rs.camera_info.serial_number)
            
            if serial_numbers is not None and device_serial not in serial_numbers:
                continue
                
            # Set up pipeline and config for this camera
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(device_serial)
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            
            # Start pipeline
            pipeline.start(config)
            
            # Store pipeline and config
            self.pipelines.append(pipeline)
            self.configs.append(config)
            
            logger.info(f"Initialized RealSense camera {cam_count} with serial: {device_serial}")
            cam_count += 1
            
            # Limit to 3 cameras
            if cam_count >= 3:
                break
        
        self.cam_count = cam_count
        logger.info(f"Initialized {self.cam_count} RealSense cameras")
    
    def get_frames(self):
        """Get the latest color frames from all cameras."""
        frames = []
        for i, pipeline in enumerate(self.pipelines):
            try:
                frameset = pipeline.wait_for_frames(timeout_ms=1000)
                color_frame = frameset.get_color_frame()
                if color_frame:
                    frames.append(np.asanyarray(color_frame.get_data()))
                else:
                    logger.warning(f"No color frame from camera {i}")
                    frames.append(None)
            except Exception as e:
                logger.error(f"Error getting frame from camera {i}: {e}")
                frames.append(None)
        
        return frames
    
    def stop(self):
        """Stop all camera pipelines."""
        for pipeline in self.pipelines:
            pipeline.stop()

def step(robot_interface, oculus, pose, gripper, info, controller_id, controller_type, controller_cfg, control_hz=CONTROL_HZ):
    """Execute a single control step for one robot arm."""
    if pose is None or gripper is None:
        logger.warning(f"[{controller_id}] Pose or gripper state is None, skipping iteration.")
        return None
    
    state = {
        "robot_state": {
            "cartesian_position": pose,
            "gripper_position": gripper,
        }
    }
    
    vel_act, info_dict = oculus.forward(state, controller_id=controller_id, include_info=True)
    
    if not info["movement_enabled"][controller_id]:
        return None
    
    action = info_dict['delta_action']
    vr_gripper = oculus.vr_state[controller_id]["gripper"]
    
    if vr_gripper < 0.5:
        action[-1] = -1
    else:
        action[-1] = 1.0
    
    robot_interface.control(
        controller_type=controller_type,
        action=action,
        controller_cfg=controller_cfg,
    )
    
    time.sleep(1/control_hz)
    return action

def record_episode(left_robot_interface, right_robot_interface, oculus, cameras, episode_num, args):
    """Record a complete episode of robot teleoperation."""
    done = False
    
    # Initialize episode data structure
    episode = dict(
        timestamp=[],
        left_joint_positions=[],
        right_joint_positions=[],
        left_delta_action=[],
        right_delta_action=[],
        left_gripper_position=[],
        right_gripper_position=[],
        camera_image_1=[],
        camera_image_2=[],
        camera_image_3=[],
        language_instruction=[],
    )
    
    # See if we want to use language instruction
    lang = args.instr if args.instr is not None else input("Language instruction? ")
    lang = "" if lang == None else lang
    episode["language_instruction"] = [lang]
    
    print("Start episode. Press Ctrl+C to stop recording.")
    
    try:
        while not done:
            # Get current timestamp
            current_time = datetime.datetime.now().timestamp()
            
            # Get robot states
            left_pose = get_ee_pose(left_robot_interface)
            left_gripper = get_gripper_position(left_robot_interface)
            right_pose = get_ee_pose(right_robot_interface)
            right_gripper = get_gripper_position(right_robot_interface)
            
            # Get camera image
            # camera_image = camera.get_frame()
            camera_images = cameras.get_frames()
            
            # Get VR controller info
            info = oculus.get_info()
            
            # Execute control steps for both arms
            left_action = step(
                left_robot_interface, oculus, left_pose, left_gripper, info,
                controller_id="l", controller_type=args.controller_type,
                controller_cfg=args.controller_cfg, control_hz=CONTROL_HZ
            )
            
            right_action = step(
                right_robot_interface, oculus, right_pose, right_gripper, info,
                controller_id="r", controller_type=args.controller_type,
                controller_cfg=args.controller_cfg, control_hz=CONTROL_HZ
            )
            
            # Record data if both control steps were successful
            if (left_action is not None or right_action is not None) and any(img is not None for img in camera_images):
                # Get absolute joint positions
                print("break")
                left_joints = np.array(left_robot_interface.last_q) if left_robot_interface.last_q is not None else np.zeros(7)
                right_joints = np.array(right_robot_interface.last_q) if right_robot_interface.last_q is not None else np.zeros(7)
                
                # Store data in episode dictionary
                if (left_action is None):
                    left_action = [0.0] * 6 + [left_gripper]
                if (right_action is None):
                    right_action = [0.0] * 6 + [right_gripper]
                
                episode["timestamp"].append(current_time)
                episode["left_joint_positions"].append(left_joints)
                episode["right_joint_positions"].append(right_joints)
                episode["left_delta_action"].append(left_action)
                episode["right_delta_action"].append(right_action)
                episode["left_gripper_position"].append(left_gripper)
                episode["right_gripper_position"].append(right_gripper)

                # Store compressed image (convert to JPEG in memory to save space)
                # _, jpeg_image = cv2.imencode('.jpg', camera_image, [cv2.IMWRITE_JPEG_QUALITY, 90])
                # episode["camera_image"].append(jpeg_image.tobytes())
                for i, img in enumerate(camera_images):
                    if img is not None:
                        _, jpeg_image = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
                        episode[f"camera_image_{i+1}"].append(jpeg_image.tobytes())
                    else:
                        # Store None or empty placeholder if camera failed
                        episode[f"camera_image_{i+1}"].append(None)
                
                # Display image with overlay
                # cv2.imshow('Camera View', camera_image)
                # cv2.waitKey(1)
                # if camera_images[0] is not None:
                #     display_img1 = camera_images[0].copy()
                #     cv2.imshow('Camera View', display_img1)
                # if camera_images[1] is not None:
                #     display_img2 = camera_images[1].copy()
                #     cv2.imshow('Camera View', display_img2)
                # if camera_images[2] is not None:
                #     display_img3 = camera_images[2].copy()
                #     cv2.imshow('Camera View', display_img3)

                if camera_images[0] is not None:
                    display_img1 = camera_images[0].copy()
                    display_img2 = camera_images[1].copy()
                    display_img3 = camera_images[2].copy()

                    # combined_image = np.hstack((display_img1, display_img2))
                    combined_image = np.hstack((display_img1, display_img2, display_img3))
                    # Add text overlay with episode info
                    # cv2.putText(combined_image, f"Episode: {episode_num}, Step: {len(episode['timestamp'])}", 
                    #             (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow('Camera View', combined_image)
                    cv2.waitKey(1)
                
    except KeyboardInterrupt:
        print("Finished episode.")
        done = True
    
    success = input("Success [s] or Failure [f]? ") == "s"
    
    if success:
        print("Saving episode.")
        ep_len = len(episode["timestamp"])
        ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        ep_filename = f"{ts}_{episode_num}_{ep_len}.npz"
        save_episode(episode, os.path.join(args.path, ep_filename))
        return True
    else:
        print("Discarding episode.")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-config", type=str, default="/home/prior/deoxys_control/deoxys/config/artoo.yml", 
                        help="Config file for left robot arm")
    parser.add_argument("--right-config", type=str, default="/home/prior/deoxys_control/deoxys/config/charmander.yml", 
                        help="Config file for right robot arm")
    parser.add_argument("--controller-type", type=str, default="OSC_POSE", 
                        help="Controller type (OSC_POSE, JOINT_POSITION, etc.)")
    parser.add_argument("--path", type=str, required=True, 
                        help="Path to save recorded episodes")
    parser.add_argument("--instr", type=str, default=None, 
                        help="Language instruction for the episode")
    
    args = parser.parse_args()
    
    # Create directory for saving episodes
    os.makedirs(args.path, exist_ok=True)
    
    # Initialize robot interfaces
    left_robot_interface = FrankaInterface(args.left_config, use_visualizer=False)
    right_robot_interface = FrankaInterface(args.right_config, use_visualizer=False)
    
    logger.info(f"Left robot interface: {left_robot_interface}")
    logger.info(f"Right robot interface: {right_robot_interface}")
    
    # Get controller configuration
    args.controller_cfg = get_default_controller_config(controller_type=args.controller_type)
    
    # Initialize VR controller
    oculus = BimanualVRController(pos_action_gain=22)
    assert oculus.get_info()["controller_on"], "ERROR: oculus controller off"
    logger.info("Oculus Connected")
    
    # Initialize camera
    # camera = RealSenseCamera()
    cameras = MultiRealSenseCamera(serial_numbers=None)
    logger.info(f"Initialized {cameras.cam_count} RealSense cameras")

    # Reset robot joints to starting position
    reset_joint(left_robot_interface)
    logger.info("Left robot interface reset")
    reset_joint(right_robot_interface)
    logger.info("Right robot interface reset")
    
    print("Starting data collection. Press Ctrl+C to exit.")
    
    # Main episode recording loop
    num_episodes = 0
    try:
        while True:
            reset_joint(left_robot_interface)
            logger.info("Left robot interface reset")
            reset_joint(right_robot_interface)
            logger.info("Right robot interface reset")

            result = record_episode(
                left_robot_interface, 
                right_robot_interface, 
                oculus, 
                cameras, 
                num_episodes, 
                args
            )
            
            if result:
                num_episodes += 1
            
            # Ask if user wants to record another episode
            if input("Record another episode? [y/n] ").lower() != 'y':
                break
    
    finally:
        # Clean up resources
        cameras.stop()

        cv2.destroyAllWindows()
        
        # Safely close robot interfaces
        left_robot_interface.close()
        right_robot_interface.close()
        
        print(f"Data collection complete. Recorded {num_episodes} episodes.")

if __name__ == "__main__":
    main()
