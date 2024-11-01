import torch
import yaml
from pathlib import Path
import subprocess
import os

def setup_yolov5():
    """
    Clone and setup YOLOv5 repository if not already present
    """
    if not Path('yolov5').exists():
        subprocess.run(['git', 'clone', 'https://github.com/ultralytics/yolov5.git'])
        subprocess.run(['pip', 'install', '-r', 'yolov5/requirements.txt'])

def train_model(data_yaml_path, epochs=100, batch_size=16, img_size=640):
    """
    Train YOLOv5 model on solar panel dataset
    """
    # Change to YOLOv5 directory
    os.chdir('yolov5')
    
    # Start training
    cmd = [
        'python', 'train.py',
        '--img', str(img_size),
        '--batch', str(batch_size),
        '--epochs', str(epochs),
        '--data', str(data_yaml_path),
        '--weights', 'yolov5s.pt',  # Use small model
        '--project', '../runs/train',
        '--name', 'solar_defect_detection'
    ]
    
    subprocess.run(cmd)
    
    # Change back to original directory
    os.chdir('..')

def main():
    # Setup paths
    data_yaml_path = Path(r'C:\Users\Eduar\Documents\masters\cap5415\solar_project\scripts\solar_panel_data\processed\dataset_info.yaml')
    
    # Ensure CUDA is available
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Setup YOLOv5
    setup_yolov5()
    
    # Train model
    print("Starting training...")
    train_model(
        data_yaml_path=data_yaml_path,
        epochs=100,
        batch_size=16,
        img_size=640
    )
    
    print("Training completed! Check the runs/train directory for results.")

if __name__ == '__main__':
    main()