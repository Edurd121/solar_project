
import torch
import yaml
from pathlib import Path
import subprocess
import os
from datetime import datetime
import sys
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import shutil
import time

class ResultsAnalyzer:
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        # Get the subdirectory with dataset name
        self.results_subdir = list(self.run_dir.glob('*'))[0]  # Get first subdirectory
        self.analysis_dir = self.run_dir / 'analysis'
        self.analysis_dir.mkdir(exist_ok=True)
        
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")

    def load_results(self):
        """Load training results from results.csv"""
        try:
            results_file = list(self.run_dir.glob('*/results.csv'))[0]
            df = pd.read_csv(results_file)
            
            # For single epoch results, create epoch column
            if 'epoch' not in df.columns:
                df.insert(0, 'epoch', range(len(df)))
            
            # Define exact column mappings based on YOLOv5's output format
            column_mapping = {
                'train/box_loss': 'box_loss',
                'train/obj_loss': 'obj_loss',
                'train/cls_loss': 'cls_loss',
                'metrics/precision': 'precision',
                'metrics/recall': 'recall',
                'metrics/mAP_0.5': 'mAP50',
                'metrics/mAP_0.5:0.95': 'mAP50-95',
                'val/box_loss': 'val_box_loss',
                'val/obj_loss': 'val_obj_loss',
                'val/cls_loss': 'val_cls_loss',
                'x/lr0': 'lr'
            }
            
            # Create new DataFrame with mapped columns
            mapped_df = pd.DataFrame()
            mapped_df['epoch'] = df['epoch'] if 'epoch' in df.columns else [0]
            
            # Map existing columns and fill missing ones with None
            for old_col, new_col in column_mapping.items():
                if old_col in df.columns:
                    mapped_df[new_col] = df[old_col]
                else:
                    mapped_df[new_col] = None
            
            # Debug output
            logging.info(f"Loaded results with columns: {mapped_df.columns.tolist()}")
            logging.info(f"Number of epochs: {len(mapped_df)}")
            
            return mapped_df
                
        except Exception as e:
            logging.error(f"Error loading results: {str(e)}")
            logging.error(f"Available files: {list(self.run_dir.glob('**/results.csv'))}")
            # If results.csv doesn't exist or is empty, create minimal DataFrame
            minimal_df = pd.DataFrame({
                'epoch': [0],
                'box_loss': [None],
                'obj_loss': [None],
                'cls_loss': [None],
                'precision': [None],
                'recall': [None],
                'mAP50': [None],
                'mAP50-95': [None]
            })
            return minimal_df

    def plot_training_curves(self, df):
        """Generate training curves"""
        try:
            # Skip plotting if only one epoch
            if len(df) <= 1:
                logging.info("Skipping plots for single epoch training")
                return
        
            # Loss Curves
            plt.figure(figsize=(12, 8))
            loss_columns = ['box_loss', 'obj_loss', 'cls_loss']
            for col in loss_columns:
                if col in df.columns:
                    plt.plot(df['epoch'], df[col], label=col.replace('_', ' ').title())
            plt.title('Training Losses Over Time')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            if any(col in df.columns for col in loss_columns):
                plt.legend()
            plt.grid(True)
            plt.savefig(self.analysis_dir / 'training_losses.png')
            plt.close()

            # Metrics Plot
            plt.figure(figsize=(12, 8))
            metric_columns = {'precision': 'Precision', 
                            'recall': 'Recall', 
                            'mAP50': 'mAP@0.5', 
                            'mAP50-95': 'mAP@0.5:0.95'}
            for col, label in metric_columns.items():
                if col in df.columns:
                    plt.plot(df['epoch'], df[col], label=label)
            plt.title('Training Metrics Over Time')
            plt.xlabel('Epoch')
            plt.ylabel('Value')
            if any(col in df.columns for col in metric_columns):
                plt.legend()
            plt.grid(True)
            plt.savefig(self.analysis_dir / 'training_metrics.png')
            plt.close()

        except Exception as e:
            logging.error(f"Error plotting training curves: {str(e)}")
            logging.error(f"Available columns: {df.columns.tolist()}")

    def plot_validation_metrics(self, df):
        """Plot validation metrics"""
        try:
            plt.figure(figsize=(12, 8))
            if 'val_box_loss' in df.columns:
                plt.plot(df['epoch'], df['val_box_loss'], label='Val Box Loss')
            if 'val_obj_loss' in df.columns:
                plt.plot(df['epoch'], df['val_obj_loss'], label='Val Obj Loss')
            if 'val_cls_loss' in df.columns:
                plt.plot(df['epoch'], df['val_cls_loss'], label='Val Cls Loss')
            plt.title('Validation Losses')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            plt.savefig(self.analysis_dir / 'validation_losses.png')
            plt.close()
        except Exception as e:
            logging.error(f"Error plotting validation metrics: {str(e)}")

    def create_training_summary(self, df):

        """Generate training summary statistics"""
        try:
            summary = {
                'final_metrics': {
                    'mAP50': float(df['mAP50'].iloc[-1]) if 'mAP50' in df.columns and not df['mAP50'].isna().all() else None,
                    'mAP50-95': float(df['mAP50-95'].iloc[-1]) if 'mAP50-95' in df.columns and not df['mAP50-95'].isna().all() else None,
                    'precision': float(df['precision'].iloc[-1]) if 'precision' in df.columns and not df['precision'].isna().all() else None,
                    'recall': float(df['recall'].iloc[-1]) if 'recall' in df.columns and not df['recall'].isna().all() else None
                },
                'loss_metrics': {
                    'box_loss': float(df['box_loss'].iloc[-1]) if 'box_loss' in df.columns and not df['box_loss'].isna().all() else None,
                    'obj_loss': float(df['obj_loss'].iloc[-1]) if 'obj_loss' in df.columns and not df['obj_loss'].isna().all() else None,
                    'cls_loss': float(df['cls_loss'].iloc[-1]) if 'cls_loss' in df.columns and not df['cls_loss'].isna().all() else None
                },
                'training_info': {
                    'epochs_completed': len(df),
                    'final_learning_rate': float(df['lr'].iloc[-1]) if 'lr' in df.columns and not df['lr'].isna().all() else None
                },
                'validation_metrics': {
                    'val_box_loss': float(df['val_box_loss'].iloc[-1]) if 'val_box_loss' in df.columns and not df['val_box_loss'].isna().all() else None,
                    'val_obj_loss': float(df['val_obj_loss'].iloc[-1]) if 'val_obj_loss' in df.columns and not df['val_obj_loss'].isna().all() else None,
                    'val_cls_loss': float(df['val_cls_loss'].iloc[-1]) if 'val_cls_loss' in df.columns and not df['val_cls_loss'].isna().all() else None
                }
            }

            # Add a more detailed logging
            logging.info(f"Created summary with metrics:")
            for category, metrics in summary.items():
                logging.info(f"{category}: {metrics}")
            
            # Save summary
            with open(self.analysis_dir / 'training_summary.yaml', 'w') as f:
                yaml.dump(summary, f)
            
            return summary
        
        except Exception as e:
            logging.error(f"Error creating training summary: {str(e)}")
            logging.error(f"DataFrame info: {df.info()}")
            logging.error(f"Available columns: {df.columns.tolist()}")
            logging.error(f"First row of data: {df.iloc[0].to_dict() if len(df) > 0 else 'No data'}")
            return None

    def analyze_results(self):
        """Run complete analysis"""
        df = self.load_results()
        self.plot_training_curves(df)
        self.plot_validation_metrics(df)
        return self.create_training_summary(df)

class YOLOv5Trainer:
    def __init__(self, base_dir='yolo_experiments'):
        self.base_dir = Path(base_dir)
        self.model_dir = self.base_dir / 'v5'
        self.datasets_dir = self.base_dir / 'datasets'
        
        # Create directories
        for dir_path in [
            self.model_dir / 'runs',
            self.model_dir / 'configs',
            self.datasets_dir
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.setup_logging()
        self.setup_yolov5()

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

    def setup_yolov5(self):
        """Clone and setup YOLOv5 repository"""
        yolov5_dir = self.model_dir / 'yolov5'
        if not yolov5_dir.exists():
            logging.info("Cloning YOLOv5 repository...")
            subprocess.run(['git', 'clone', 'https://github.com/ultralytics/yolov5.git', str(yolov5_dir)])
            subprocess.run(['pip', 'install', '-r', str(yolov5_dir / 'requirements.txt')])
            logging.info("YOLOv5 setup completed")

    def create_run_directory(self, dataset_name, model_size, batch_size, epochs):
        """Create uniquely named run directory with dataset subdirectory"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # First create dataset directory if it doesn't exist
        dataset_dir = self.model_dir / 'runs' / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Then create the specific run directory inside dataset directory
        run_name = f"yolov5{model_size}_batch{batch_size}_epoch{epochs}_{timestamp}"
        run_dir = dataset_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir
    
    def train_model(self, data_yaml_path, dataset_name, model_size='s', batch_size=16, epochs=100, img_size=640):
        """Train YOLOv5 model"""
        try:
            run_dir = self.create_run_directory(dataset_name, model_size, batch_size, epochs)
            logging.info(f"Created run directory: {run_dir}")

            # Save configuration
            config = {
                'experiment_info': {
                    'model': f'yolov5{model_size}',
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

            # Change to YOLOv5 directory
            original_dir = os.getcwd()
            os.chdir(self.model_dir / 'yolov5')
            
            # Training command
            cmd = [
                'python', 'train.py',
                '--img', str(img_size),
                '--batch', str(batch_size),
                '--epochs', str(epochs),
                '--data', str(data_yaml_path),
                '--weights', f'yolov5{model_size}.pt',
                '--project', str(run_dir),
                '--name', dataset_name
            ]
            
            logging.info(f"Starting training with command: {' '.join(cmd)}")
            subprocess.run(cmd)
            
            # Return to original directory
            os.chdir(original_dir)

            # Wait a moment for file system
            time.sleep(2)  

            # Analyze results
            try:
                analyzer = ResultsAnalyzer(run_dir)
                training_summary = analyzer.analyze_results()
                
                logging.info("\nTraining Summary:")
                logging.info(f"Best mAP@0.5: {training_summary['best_metrics']['mAP50']:.4f}")
                logging.info(f"Best mAP@0.5:0.95: {training_summary['best_metrics']['mAP50-95']:.4f}")
                
                return run_dir, training_summary
            except Exception as e:
                logging.error(f"Analysis failed but training completed successfully: {str(e)}")
                # Return just the run directory if analysis fails
                return run_dir, None

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
        {'size': 'm', 'batch_size': 16},  # Medium model
        {'size': 'l', 'batch_size': 8}    # Large model with smaller batch size
    ]
    
    EPOCHS = 100
    IMG_SIZE = 640

    trainer = YOLOv5Trainer(base_dir=r'C:\Users\Eduar\Documents\masters\cap5415\solar_project\scripts\yolo_experiments')

    for dataset_name, data_yaml_path in DATASETS.items():
        for config in MODEL_CONFIGS:
            try:
                logging.info(f"\nStarting experiment:")
                logging.info(f"Dataset: {dataset_name}")
                logging.info(f"Model: YOLOv5{config['size']}")
                logging.info(f"Batch size: {config['batch_size']}")
                
                run_dir, summary = trainer.train_model(
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
