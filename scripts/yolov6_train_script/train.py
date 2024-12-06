import torch
import yaml
from pathlib import Path
import subprocess
import os
from datetime import datetime
import sys
import logging

class YOLOv6Trainer:
    def __init__(self, base_dir='yolo_experiments'):
        self.base_dir = Path(base_dir)
        self.model_dir = self.base_dir / 'v6'
        self.datasets_dir = self.base_dir / 'datasets'
        
        # Create directories
        for dir_path in [
            self.model_dir / 'runs',
            self.model_dir / 'configs',
            self.datasets_dir
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.setup_logging()
        self.setup_yolov6()

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

    def setup_yolov6(self):
        """Clone and setup YOLOv6 repository"""
        yolov6_dir = self.model_dir / 'yolov6'
        if not yolov6_dir.exists():
            logging.info("Cloning YOLOv6 repository...")
            subprocess.run(['git', 'clone', 'https://github.com/meituan/YOLOv6.git', str(yolov6_dir)])
            subprocess.run(['pip', 'install', '-r', str(yolov6_dir / 'requirements.txt')])
            logging.info("YOLOv6 setup completed")

    def create_run_directory(self, dataset_name, model_size, batch_size, epochs):
        """Create uniquely named run directory with dataset subdirectory"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create dataset directory
        dataset_dir = self.model_dir / 'runs' / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Create specific run directory
        run_name = f"yolov6{model_size}_batch{batch_size}_epoch{epochs}_{timestamp}"
        run_dir = dataset_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir
    
    def create_yolov6_config(self, data_yaml_path, run_dir, model_size='n'):
        """Create YOLOv6 configuration file"""
        config = {
            'model': f'yolov6{model_size}',
            'data': str(data_yaml_path),
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'img_size': self.img_size,
            'device': '0' if torch.cuda.is_available() else 'cpu',
            'eval_interval': 1,
            'save_interval': 10,
            'workers': 8,
            'output_dir': str(run_dir),
            'name': 'train'
        }
        
        config_path = run_dir / 'yolov6_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        return config_path
    
    def train_model(self, data_yaml_path, dataset_name, model_size='n', batch_size=32, epochs=100, img_size=640):
        """Train YOLOv6 model"""
        try:
            self.batch_size = batch_size
            self.epochs = epochs
            self.img_size = img_size
            
            run_dir = self.create_run_directory(dataset_name, model_size, batch_size, epochs)
            logging.info(f"Created run directory: {run_dir}")

            # Create weights directory and download weights
            weights_dir = self.model_dir / 'yolov6' / 'weights'
            weights_dir.mkdir(parents=True, exist_ok=True)

            # Download weights
            weights_mapping = {
                'n': 'yolov6n.pt',
                's': 'yolov6s.pt',
                'm': 'yolov6m.pt',
                'l': 'yolov6l.pt'
            }
            
            weights_file = weights_dir / weights_mapping[model_size]
            if not weights_file.exists():
                logging.info(f"Downloading {weights_mapping[model_size]}...")
                weights_url = f"https://github.com/meituan/YOLOv6/releases/download/0.3.0/{weights_mapping[model_size]}"
                try:
                    import urllib.request
                    urllib.request.urlretrieve(weights_url, weights_file)
                    logging.info(f"Successfully downloaded weights to {weights_file}")
                except Exception as e:
                    logging.error(f"Error downloading weights: {str(e)}")
                    raise

            # Create config
            config_path = self.create_yolov6_config(data_yaml_path, run_dir, model_size)

            # Change to YOLOv6 directory
            original_dir = os.getcwd()
            os.chdir(self.model_dir / 'yolov6')
            
            # Updated training command for YOLOv6
            cmd = [
                'python', 'tools/train.py',
                '--data-path', str(data_yaml_path),
                '--conf-file', str(config_path),
                '--img-size', str(img_size),
                '--batch-size', str(batch_size),
                '--epochs', str(epochs),
                '--device', '0' if torch.cuda.is_available() else 'cpu',
                '--output-dir', str(run_dir)
            ]
            
            logging.info(f"Starting training with command: {' '.join(cmd)}")
            subprocess.run(cmd)
            
            # Return to original directory
            os.chdir(original_dir)
            
            return run_dir

        except Exception as e:
            logging.error(f"Error during training: {str(e)}")
            raise

def main():
    # Configuration
    DATASETS = {
        'original_solar': Path(r'C:\Users\Eduar\Documents\masters\cap5415\datasets\solar_panel_data\processed\dataset_info.yaml'),
        'combined_solar': Path(r'C:\Users\Eduar\Documents\masters\cap5415\datasets\combined_solar_dataset\dataset_info.yaml')
    }
    
    # Define model configurations
    MODEL_CONFIGS = [
        {'size': 'n', 'batch_size': 32},  # Nano model with larger batch size
        {'size': 's', 'batch_size': 16},  # Small model
    ]
    
    EPOCHS = 100
    IMG_SIZE = 640

    trainer = YOLOv6Trainer(base_dir=r'C:\Users\Eduar\Documents\masters\cap5415\solar_project\scripts\yolo_experiments')

    for dataset_name, data_yaml_path in DATASETS.items():
        for config in MODEL_CONFIGS:
            try:
                logging.info(f"\nStarting experiment:")
                logging.info(f"Dataset: {dataset_name}")
                logging.info(f"Model: YOLOv6{config['size']}")
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