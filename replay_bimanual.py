import argparse
import os
import time
import numpy as np
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import YamlConfig
from deoxys.utils.log_utils import get_deoxys_example_logger

logger = get_deoxys_example_logger()

def precise_wait(t_end: float, slack_time: float = 0.001):
    t_start = time.time()
    t_wait = t_end - t_start
    if t_wait > 0:
        t_sleep = t_wait - slack_time
        if t_sleep > 0:
            time.sleep(t_sleep)
        while time.time() < t_end:
            pass

def load_episode(file_path):
    """Load a recorded episode from a .npz file."""
    logger.info(f"Loading episode from {file_path}")
    try:
        data = np.load(file_path, allow_pickle=True)
        return data
    except Exception as e:
        logger.error(f"Failed to load episode: {e}")
        return None


def replay_episode(left_robot_interface, right_robot_interface, episode_data, replay_speed=1.0):
    """Replay a recorded episode using absolute joint positions at exactly 20 Hz."""
    logger.info("Starting episode replay")
    
    # Extract joint positions from the episode data
    # Absolute joint positions
    # left_joint_positions = episode_data['left_joint_positions']
    # right_joint_positions = episode_data['right_joint_positions']

    #Delta joint positions
    left_joint_positions = episode_data['left_delta_action']
    right_joint_positions = episode_data['right_delta_action']
    
    # Define the control frequency (same as recording)
    control_hz = 10
    step_time = 1.0 / control_hz  # 0.05 seconds per step
    
    # Ensure we have joint positions
    if len(left_joint_positions) == 0 or len(right_joint_positions) == 0:
        logger.error("No joint positions found in episode data")
        return False
    
    logger.info(f"Replaying episode with {len(left_joint_positions)} steps at {control_hz} Hz")
    
    # Get controller configuration for joint position control
    # controller_type = "JOINT_POSITION"
    controller_type = "OSC_POSE"

    # controller_cfg = YamlConfig("/home/prior/deoxys_control/deoxys/config/joint-position-controller.yml").as_easydict()
    controller_cfg = YamlConfig("/home/prior/deoxys_control/deoxys/config/osc-pose-controller.yml").as_easydict()
    
    try:
        for i in range(len(left_joint_positions)):
            start_time = time.time()
            
            # Get current joint positions for both arms
            left_joints = left_joint_positions[i]
            right_joints = right_joint_positions[i]
            
            # Get left and right gripper positions
            left_gripper = episode_data['left_gripper_position'][i]
            right_gripper = episode_data['right_gripper_position'][i]
            
            # # Convert gripper value to action (-1 for open, 1 for close)
            # left_gripper_action = -1 if left_gripper > 0.5 else 1
            # right_gripper_action = -1 if right_gripper > 0.5 else 1
            
            # Create actions with joint positions and gripper commands
            left_action = list(left_joints)
            right_action = list(right_joints)
            
            # Send commands to both robots
            if (left_action != None): 
                left_robot_interface.control(
                    controller_type=controller_type,
                    action=left_action,
                    controller_cfg=controller_cfg,
                )
            # time.sleep(step_time)
            
            if (right_action != None):
                right_robot_interface.control(
                    controller_type=controller_type,
                    action=right_action,
                    controller_cfg=controller_cfg,
                )
            
            time.sleep(step_time)
            
            # Calculate remaining time to maintain exactly 20 Hz frequency
            # elapsed = time.time() - start_time
            # sleep_time = (step_time) - elapsed
            # time.sleep(sleep_time)
            
            # Print progress
            # if i % 20 == 0:
            #     logger.info(f"Replay progress: {i}/{len(left_joint_positions)} steps")
        
        logger.info("Episode replay completed successfully")
        return True
    
    except KeyboardInterrupt:
        logger.info("Replay interrupted by user")
        return False
    except Exception as e:
        logger.error(f"Error during replay: {e}")
        return False


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
    
    action = reset_joint_positions + [-1.0]  # Open gripper
    
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

def main():
    parser = argparse.ArgumentParser(description="Replay recorded robot demonstrations")
    parser.add_argument("--left-config", type=str, default="/home/prior/deoxys_control/deoxys/config/artoo.yml", 
                        help="Config file for left robot arm")
    parser.add_argument("--right-config", type=str, default="/home/prior/deoxys_control/deoxys/config/charmander.yml", 
                        help="Config file for right robot arm")
    parser.add_argument("--demo-file", type=str, required=True, 
                        help="Path to the .npz file containing the recorded demonstration")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Replay speed multiplier (1.0 = original speed, 0.5 = half speed, 2.0 = double speed)")
    parser.add_argument("--reset-before", action="store_true", 
                        help="Reset robots to default position before replay")
    
    args = parser.parse_args()
    
    # Check if demo file exists
    if not os.path.exists(args.demo_file):
        logger.error(f"Demo file not found: {args.demo_file}")
        return
    
    # Initialize robot interfaces
    left_robot_interface = FrankaInterface(args.left_config, use_visualizer=False)
    right_robot_interface = FrankaInterface(args.right_config, use_visualizer=False)
    
    logger.info(f"Left robot interface: {left_robot_interface}")
    logger.info(f"Right robot interface: {right_robot_interface}")
    
    try:
        # Load the episode data
        episode_data = load_episode(args.demo_file)
        if episode_data is None:
            return
        
        # Reset robots if requested
        if args.reset_before:
            logger.info("Resetting robots to default position")
            reset_joint(left_robot_interface)
            reset_joint(right_robot_interface)
        
        # Wait for user confirmation
        input("Press Enter to start replay...")
        
        # Replay the episode
        replay_episode(left_robot_interface, right_robot_interface, episode_data, args.speed)
        
    finally:
        # Safely close robot interfaces
        left_robot_interface.close()
        right_robot_interface.close()
        
        logger.info("Replay script completed")

if __name__ == "__main__":
    main()
