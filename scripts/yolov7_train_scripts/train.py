import torch
import yaml
from pathlib import Path
import subprocess
import os
from datetime import datetime
import sys
import logging
os.environ['WANDB_DISABLED'] = 'true'
class YOLOv7Trainer:
    def __init__(self, base_dir='yolo_experiments'):
        self.base_dir = Path(base_dir)
        self.model_dir = self.base_dir / 'v7'
        self.datasets_dir = self.base_dir / 'datasets'
        
        # Create directories
        for dir_path in [
            self.model_dir / 'runs',
            self.model_dir / 'configs',
            self.datasets_dir
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.setup_logging()
        self.setup_yolov7()

    def setup_logging(self):
        """Configure logging"""
        log_file = self.model_dir / 'training.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )

    def setup_yolov7(self):
        """Clone and setup YOLOv7 repository"""
        yolov7_dir = self.model_dir / 'yolov7'
        if not yolov7_dir.exists():
            logging.info("Cloning YOLOv7 repository...")
            subprocess.run(['git', 'clone', 'https://github.com/WongKinYiu/yolov7.git', str(yolov7_dir)])
            subprocess.run(['pip', 'install', '-r', str(yolov7_dir / 'requirements.txt')])
            
            # Download pretrained weights
            weights_dir = yolov7_dir / 'weights'
            weights_dir.mkdir(exist_ok=True)
            
            weights = {
                'n': 'yolov7-tiny.pt',  # tiny for nano
                's': 'yolov7.pt',       # base for small
                'm': 'yolov7-e6.pt',    # e6 for medium
                'l': 'yolov7-d6.pt'     # d6 for large
            }
            
            for size, weight_file in weights.items():
                weight_path = weights_dir / weight_file
                if not weight_path.exists():
                    logging.info(f"Downloading {weight_file}...")
                    subprocess.run([
                        'wget', 
                        f'https://github.com/WongKinYiu/yolov7/releases/download/v0.1/{weight_file}',
                        '-O', 
                        str(weight_path)
                    ])
            
            logging.info("YOLOv7 setup completed")

    def create_run_directory(self, dataset_name, model_size, batch_size, epochs):
        """Create uniquely named run directory with dataset subdirectory"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create dataset directory
        dataset_dir = self.model_dir / 'runs' / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Create specific run directory
        run_name = f"yolov7{model_size}_batch{batch_size}_epoch{epochs}_{timestamp}"
        run_dir = dataset_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def get_model_weight_file(self, model_size):
        """Get the appropriate weight file for the model size"""
        weight_mapping = {
            'n': 'yolov7-tiny.pt',
            's': 'yolov7.pt',
            'm': 'yolov7-e6.pt',
            'l': 'yolov7-d6.pt'
        }
        return weight_mapping.get(model_size, 'yolov7.pt')

    def train_model(self, data_yaml_path, dataset_name, model_size='s', batch_size=16, epochs=100, img_size=640):
        """Train YOLOv7 model"""
        try:
            run_dir = self.create_run_directory(dataset_name, model_size, batch_size, epochs)
            logging.info(f"Created run directory: {run_dir}")

            # Save configuration
            config = {
                'experiment_info': {
                    'model': f'yolov7{model_size}',
                    'dataset': dataset_name,
                    'batch_size': batch_size,
                    'epochs': epochs,
                    'img_size': img_size,
                    'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                },
                'data_path': str(data_yaml_path),
                'run_directory': str(run_dir)
            }
            
            with open(run_dir / 'experiment_config.yaml', 'w') as f:
                yaml.dump(config, f)

            # Change to YOLOv7 directory
            original_dir = os.getcwd()
            os.chdir(self.model_dir / 'yolov7')
            
            # Get appropriate weight file
            weight_file = self.get_model_weight_file(model_size)
            
            # Training command for YOLOv7
            cmd = [
                'python', 'train.py',
                '--workers', '8',
                '--device', '0',
                '--batch-size', str(batch_size),
                '--epochs', str(epochs),
                '--img', str(img_size),
                '--data', str(data_yaml_path),
                '--weights', f'weights/{weight_file}',
                '--cfg', f'cfg/training/yolov7{"-tiny" if model_size == "n" else ""}.yaml',
                '--name', str(run_dir.name),
                '--hyp', 'data/hyp.scratch.p5.yaml',
                '--exist-ok',


            ]
            
            logging.info(f"Starting training with command: {' '.join(cmd)}")
            subprocess.run(cmd)
            
            # Return to original directory
            os.chdir(original_dir)
            
            logging.info(f"Training completed. Results saved in: {run_dir}")
            return run_dir

        except Exception as e:
            logging.error(f"Error during training: {str(e)}")
            raise

def main():
    # Configuration
    DATASETS = {
        'original_solar': Path(r'C:\Users\Eduar\Documents\masters\cap5415\datasets\solar_panel_data\processed\dataset_info_v7.yaml'),
        'combined_solar': Path(r'C:\Users\Eduar\Documents\masters\cap5415\datasets\combined_solar_dataset\dataset_info_v7.yaml')
    }
    
    # Define model configurations
    MODEL_CONFIGS = [
        {'size': 'n', 'batch_size': 32},  # Nano model (tiny) with larger batch size
        {'size': 's', 'batch_size': 16},  # Small model (base)
    ]
    
    EPOCHS = 100
    IMG_SIZE = 640

    trainer = YOLOv7Trainer(base_dir=r'C:\Users\Eduar\Documents\masters\cap5415\solar_project\scripts\yolo_experiments')

    for dataset_name, data_yaml_path in DATASETS.items():
        for config in MODEL_CONFIGS:
            try:
                logging.info(f"\nStarting experiment:")
                logging.info(f"Dataset: {dataset_name}")
                logging.info(f"Model: YOLOv7{config['size']}")
                logging.info(f"Batch size: {config['batch_size']}")
                
                run_dir = trainer.train_model(
                    data_yaml_path=data_yaml_path,
                    dataset_name=dataset_name,
                    model_size=config['size'],
                    batch_size=config['batch_size'],
                    epochs=EPOCHS,
                    img_size=IMG_SIZE
                )
                
                logging.info(f"Experiment completed. Results saved in: {run_dir}")

            except Exception as e:
                logging.error(f"Experiment failed: {str(e)}")
                continue

if __name__ == "__main__":
    main()