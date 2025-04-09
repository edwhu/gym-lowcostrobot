from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
import torch
import numpy as np

def analyze_dataset(dataset_path):
    fps = 30
    delta_timestamps = {
        "observation.rgb": [t / fps for t in range(2)],
        "observation.depth": [t / fps for t in range(2)],
        "observation.arm_qpos": [t / fps for t in range(2)],
        "observation.segmentation": [t / fps for t in range(2)],
        "observation.estimated_target_pos": [t / fps for t in range(2)], ### NOTE: this is new, not sure if it works
        "observation.target_eepos": [t / fps for t in range(2)],
        "observation.gripper_blocked": [t / fps for t in range(2)],
        "observation.ee_pos": [t / fps for t in range(2)],
        "observation.log_is_success": [t / fps for t in range(2)],
        "action": [t / fps for t in range(2)],
        "reward": [0],
        "done": [0],
        "terminated": [0],
    }
    repo_id="edwhu/koch_robot_dataset_scripted_est_target_pos"
    # Load the dataset from local path
    dataset = LeRobotDataset(repo_id=repo_id, root=dataset_path, delta_timestamps=delta_timestamps)
    
    # Print general dataset information
    print(f"Dataset information:")
    print(f"Number of episodes: {dataset.num_episodes}")
    print(f"Number of frames: {dataset.num_frames}")
    print(f"Features: {list(dataset.features.keys())}")
    print(f"Camera keys: {dataset.meta.camera_keys}")
    print("\n")
    
    # Get the first sample to analyze
    for i in range(len(dataset)):
        sample = dataset[i]
        # Print information about each feature in the sample
        print("Sample feature analysis:")
        if sample['reward'] > 0:
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    print(f"Feature: {key}")
                    print(f"  Shape: {value.shape}")
                    print(f"  Dtype: {value.dtype}")
                    # Skip non-numeric data for min/max calculation
                    if torch.is_floating_point(value) or torch.is_complex(value) or value.dtype in [torch.int32, torch.int64]:
                        try:
                            print(f"  Min: {value.min().item()}")
                            print(f"  Max: {value.max().item()}")
                        except (RuntimeError, ValueError):
                            print(f"  Min/Max: Not applicable")
                    print()
                else:
                    print(f"Feature: {key}")
                    print(f"  Value: {value}")
            import ipdb; ipdb.set_trace()

if __name__ == "__main__":
    analyze_dataset(dataset_path="./data/koch_robot_dataset_scripted_est_target_pos")