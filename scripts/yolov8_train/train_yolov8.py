from ultralytics import YOLO
import torch
import yaml
import os
from pathlib import Path
from datetime import datetime

class SolarPanelTrainer:
    def __init__(self):
        self.base_dir = Path(os.getcwd()).resolve()
        self.data_dir = self.base_dir / 'solar_panel_data' / 'processed'
        self.img_size = 640
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def create_save_dir(self, model_name, batch_size, epochs):
        """Create uniquely named save directory"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        dir_name = f"solar_defect_detection_{model_name}_batch{batch_size}_epoch{epochs}_{timestamp}"
        save_dir = self.base_dir / 'runs' / dir_name
        save_dir.mkdir(parents=True, exist_ok=True)
        return save_dir
    
    def create_data_yaml(self):
        """Create YAML file for dataset configuration"""
        try:
            train_path = self.data_dir / 'train' / 'images'
            val_path = self.data_dir / 'val' / 'images'
            test_path = self.data_dir / 'test' / 'images'

            # Verify paths exist
            for path, name in [(train_path, 'Train'), (val_path, 'Val'), (test_path, 'Test')]:
                if not path.exists():
                    raise FileNotFoundError(f"{name} path does not exist: {path}")
                print(f"{name} path exists: {path}")

            data_yaml = {
                'path': str(self.data_dir),
                'train': str(train_path.relative_to(self.data_dir)),
                'val': str(val_path.relative_to(self.data_dir)),
                'test': str(test_path.relative_to(self.data_dir)),
                'names': {
                    0: 'defect',
                    1: 'no_defect'
                },
                'nc': 2
            }
            
            yaml_path = self.data_dir / 'dataset.yaml'
            print(f"\nCreating dataset.yaml at: {yaml_path}")
            
            with open(yaml_path, 'w') as f:
                yaml.dump(data_yaml, f)
                
            return yaml_path

        except Exception as e:
            print(f"Error creating data.yaml: {str(e)}")
            raise

    def get_metrics_safely(self, metrics):
        """Safely extract metrics from validation results"""
        try:
            # YOLOv8 changes the metrics keys sometimes, so we need to handle both possibilities
            metrics_dict = {}
            
            # For mAP50
            if 'metrics/mAP50' in metrics.results_dict:
                metrics_dict['mAP50'] = metrics.results_dict['metrics/mAP50']
            elif 'mAP50' in metrics.results_dict:
                metrics_dict['mAP50'] = metrics.results_dict['mAP50']
            else:
                metrics_dict['mAP50'] = metrics.results_dict.get('mAP_50', 0.0)

            # For mAP50-95
            if 'metrics/mAP50-95' in metrics.results_dict:
                metrics_dict['mAP50-95'] = metrics.results_dict['metrics/mAP50-95']
            elif 'mAP50-95' in metrics.results_dict:
                metrics_dict['mAP50-95'] = metrics.results_dict['mAP50-95']
            else:
                metrics_dict['mAP50-95'] = metrics.results_dict.get('mAP_50-95', 0.0)

            # For Precision
            if 'metrics/precision' in metrics.results_dict:
                metrics_dict['precision'] = metrics.results_dict['metrics/precision']
            else:
                metrics_dict['precision'] = metrics.results_dict.get('precision', 0.0)

            # For Recall
            if 'metrics/recall' in metrics.results_dict:
                metrics_dict['recall'] = metrics.results_dict['metrics/recall']
            else:
                metrics_dict['recall'] = metrics.results_dict.get('recall', 0.0)

            return metrics_dict

        except Exception as e:
            print(f"Error extracting metrics: {str(e)}")
            return {
                'mAP50': 0.0,
                'mAP50-95': 0.0,
                'precision': 0.0,
                'recall': 0.0
            }

    def train_model(self, epochs=100, batch_size=16, model_size='n'):
        """Train YOLOv8 model"""
        try:
            # Model selection
            model_name = f'yolov8{model_size}'
            model = YOLO(f'{model_name}.pt')
            
            # Create save directory
            save_dir = self.create_save_dir(model_name, batch_size, epochs)
            weights_dir = save_dir / 'weights'
            weights_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"\nSaving results to: {save_dir}")
            
            # Create data.yaml
            data_yaml_path = self.create_data_yaml()
            
            # Training arguments
            args = {
                'data': str(data_yaml_path),
                'epochs': epochs,
                'batch': batch_size,
                'imgsz': self.img_size,
                'save': True,
                'save_period': 10,
                'cache': 'disk',  # Changed to disk for deterministic results
                'device': self.device,
                'workers': 8,
                'project': str(save_dir.parent),
                'name': save_dir.name,
                'exist_ok': True,
                
                # Augmentation parameters
                'hsv_h': 0.015,
                'hsv_s': 0.7,
                'hsv_v': 0.4,
                'degrees': 0.0,
                'translate': 0.1,
                'scale': 0.5,
                'fliplr': 0.5,
                'mosaic': 1.0,
                
                # Optimization parameters
                'optimizer': 'SGD',
                'lr0': 0.01,
                'lrf': 0.01,
                'momentum': 0.937,
                'weight_decay': 0.0005,
                'warmup_epochs': 3.0,
                'warmup_momentum': 0.8,
                'warmup_bias_lr': 0.1,
                'box': 7.5,
                'cls': 0.5,
                'dfl': 1.5,
                
                'verbose': True,
                'seed': 42,
            }
            
            # Save configuration
            with open(save_dir / 'training_config.yaml', 'w') as f:
                yaml.dump(args, f)
            
            # Train
            print(f"\nStarting training with {model_name}, batch_size={batch_size}, epochs={epochs}")
            results = model.train(**args)
            print("Training completed successfully!")
            
            return model, results, save_dir

        except Exception as e:
            print(f"Error during training: {str(e)}")
            raise

def main():
    # Training parameters
    EPOCHS = 100
    BATCH_SIZE = 32
    MODEL_SIZE = 'n'
    
    trainer = SolarPanelTrainer()
    
    try:
        # Train model
        print("Starting training process...")
        model, results, save_dir = trainer.train_model(
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            model_size=MODEL_SIZE
        )
        
        print("\nTraining completed!")
        print(f"Results saved in: {save_dir}")
        
        # Validate final model
        print("\nRunning final validation...")
        val_results = model.val()
        
        # Get metrics safely
        metrics = trainer.get_metrics_safely(val_results)
        
        # Print final metrics
        print("\nFinal Model Performance:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")
        
        # Save training summary
        summary = {
            'model_type': f'yolov8{MODEL_SIZE}',
            'batch_size': BATCH_SIZE,
            'epochs': EPOCHS,
            'final_metrics': metrics
        }
        
        with open(save_dir / 'training_summary.yaml', 'w') as f:
            yaml.dump(summary, f)
            
        print(f"\nTraining summary saved to: {save_dir / 'training_summary.yaml'}")

    except Exception as e:
        print(f"\nError occurred during execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
