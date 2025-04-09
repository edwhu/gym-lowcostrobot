import matplotlib
matplotlib.use('Agg')  # Add this at the top of your file, before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import os
from pathlib import Path

def visualize_estimated_target_pos(dataset: LeRobotDataset) -> None:
    """Visualize the estimated target pos in the dataset as a 3D line plot."""
    x_coords = []
    y_coords = []
    z_coords = []
    episode_count = 0
    # To process all episodes individually
    for episode_idx in range(len(dataset)):
        episode_data = dataset[episode_idx]

        estimated_target_positions = episode_data['observation.estimated_target_pos']
        x_coords.append(estimated_target_positions[0])
        y_coords.append(estimated_target_positions[1])
        z_coords.append(estimated_target_positions[2])
        
        if episode_data ['done']:
            # create and display the 3d graph with x, y, z coordinates
            fig = plt.figure()
            fig.set_size_inches(10, 10)
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(x_coords, y_coords, z_coords)
            ax.scatter(x_coords[0], y_coords[0], z_coords[0], color='red')
            ax.scatter(x_coords[-1], y_coords[-1], z_coords[-1], color='green')
            # fixate the axes to be the same for all episodes
            ax.set_xlim(-0.2, 0)
            ax.set_ylim(-0.1, 0.1)
            ax.set_zlim(-0.0005, 0.07)

            # set labels and title
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title('Estimated Target Positions')
        
            # Instead of plt.show(), save the figure to a file in the estimated_target_pos_img folder. If it doesn't exist, create it.
            folder_name = 'estimated_target_pos_img_refactor_getobs'
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            plt.savefig(f'{folder_name}/episode_{episode_count}_trajectory.png')
            print(f'Saved episode {episode_count} to {folder_name}/episode_{episode_count}_trajectory.png')
            episode_count += 1
            # import ipdb; ipdb.set_trace()
            
            # clear the coordinates and close the figure
            x_coords = []
            y_coords = []
            z_coords = []
            plt.close('all')  # Use close('all') instead of just close()

if __name__ == "__main__":
    dataset_path = Path("./data/koch_robot_dataset_scripted_test_refactor_getobs")
    dataset = LeRobotDataset(repo_id="edwhu/koch_robot_dataset_scripted_v1.1", root=dataset_path)
    visualize_estimated_target_pos(dataset)

