
from ultralytics import RTDETR
import torch
import yaml
from pathlib import Path
import subprocess
import os
from datetime import datetime
import sys
import logging

class RTDETRTrainer:
    def __init__(self, base_dir='yolo_experiments'):
        self.base_dir = Path(base_dir)
        self.model_dir = self.base_dir / 'rt_detr'
        self.datasets_dir = self.base_dir / 'datasets'
        
        for dir_path in [
            self.model_dir / 'runs',
            self.model_dir / 'configs',
            self.datasets_dir
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.setup_logging()

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

    def create_run_directory(self, dataset_name, model_size, batch_size, epochs):
        """Create uniquely named run directory with dataset subdirectory"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        dataset_dir = self.model_dir / 'runs' / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        run_name = f"rtdetr_{model_size}_batch{batch_size}_epoch{epochs}_{timestamp}"
        run_dir = dataset_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def train_model(self, data_yaml_path, dataset_name, model_size='s', batch_size=16, epochs=100, img_size=640):
        """Train RT-DETR model"""
        try:
            run_dir = self.create_run_directory(dataset_name, model_size, batch_size, epochs)
            logging.info(f"Created run directory: {run_dir}")

            config = {
                'experiment_info': {
                    'model': f'rtdetr-{model_size}',
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

            model = RTDETR(f'rtdetr-{model_size}.pt')
            
            args = {
                'data': str(data_yaml_path),
                'epochs': epochs,
                'batch': batch_size,
                'imgsz': img_size,
                'save': True,
                'save_period': 10,
                'cache': 'disk',
                'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                'workers': 8,
                'project': str(run_dir),
                'name': dataset_name,
                'exist_ok': True,
                'pretrained': True,
                'optimizer': 'AdamW',
                'lr0': 0.0001,
                'lrf': 0.0001,
                'momentum': 0.937,
                'weight_decay': 0.0005,
                'warmup_epochs': 3.0,
                'warmup_momentum': 0.8,
                'warmup_bias_lr': 0.1,
                'box': 7.5,
                'cls': 0.5,
                'dfl': 1.5,
                'seed': 42,
            }
            
            logging.info(f"Starting training with RT-DETR-{model_size}")
            model.train(**args)
            
            logging.info(f"Training completed. Results saved in: {run_dir}")
            return run_dir

        except Exception as e:
            logging.error(f"Error during training: {str(e)}")
            raise

def main():
    DATASETS = {
        'original_solar': Path(r'C:\Users\Eduar\Documents\masters\cap5415\datasets\solar_panel_data\processed\dataset_info.yaml'),
        'combined_solar': Path(r'C:\Users\Eduar\Documents\masters\cap5415\datasets\combined_solar_dataset\dataset_info.yaml')
    }
    
    MODEL_CONFIGS = [
        {'size': 'n', 'batch_size': 32},    # Nano model with larger batch size
        #{'size': 's', 'batch_size': 16},    # Small model
        #{'size': 'l', 'batch_size': 8},    # Large model
        #{'size': 'x', 'batch_size': 4}     # Extra large model
    ]
    
    EPOCHS = 100
    IMG_SIZE = 640

    trainer = RTDETRTrainer(base_dir=r'C:\Users\Eduar\Documents\masters\cap5415\solar_project\scripts\yolo_experiments')

    for dataset_name, data_yaml_path in DATASETS.items():
        for config in MODEL_CONFIGS:
            try:
                logging.info(f"\nStarting experiment:")
                logging.info(f"Dataset: {dataset_name}")
                logging.info(f"Model: RT-DETR-{config['size']}")
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
