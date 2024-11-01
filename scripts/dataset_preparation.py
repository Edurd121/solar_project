import os
import requests
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split
import yaml

class SolarDatasetPreparer:
    def __init__(self, base_dir='solar_panel_data'):
        """
        Initialize the dataset preparer
        Args:
            base_dir: Base directory to store the dataset
        """
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / 'raw'
        self.processed_dir = self.base_dir / 'processed'
        self.train_dir = self.processed_dir / 'train'
        self.val_dir = self.processed_dir / 'val'
        self.test_dir = self.processed_dir / 'test'
        
        # Create directories
        for dir_path in [self.raw_dir, self.train_dir, self.val_dir, self.test_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            (dir_path / 'images').mkdir(exist_ok=True)
            (dir_path / 'labels').mkdir(exist_ok=True)

    def download_elpv_dataset(self):
        """
        Download the ELPV dataset
        """
        print("Downloading dataset...")
        
        # Base URL for the dataset
        base_url = "https://raw.githubusercontent.com/zae-bayern/elpv-dataset/master/src/elpv_dataset/data/images/"
        
        # Download labels
        labels_url = "https://raw.githubusercontent.com/zae-bayern/elpv-dataset/refs/heads/master/src/elpv_dataset/data/labels.csv"
        print("Downloading labels...")
        response = requests.get(labels_url)
        labels_path = self.raw_dir / 'labels.csv'
        with open(labels_path, 'w') as f:
            f.write(response.text)
        
        # Read labels file
        # The file has no header, so we'll add our own
        labels_df = pd.read_csv(labels_path, 
                              sep='\s+',  # Split on whitespace
                              names=['image_path', 'defect_score', 'cell_type'])
        
        # Create images directory if it doesn't exist
        images_dir = self.raw_dir / 'images'
        images_dir.mkdir(exist_ok=True)
        
        print("Downloading images...")
        total_images = len(labels_df)
        for idx, row in labels_df.iterrows():
            image_name = Path(row['image_path']).name
            image_url = f"{base_url}/{image_name}"
            output_path = images_dir / image_name
            
            if not output_path.exists():  # Skip if image already downloaded
                try:
                    print(f"Downloading image {idx+1}/{total_images}: {image_name}")
                    response = requests.get(image_url)
                    response.raise_for_status()  # Raise an error for bad status codes
                    
                    with open(output_path, 'wb') as f:
                        f.write(response.content)
                except requests.exceptions.RequestException as e:
                    print(f"Error downloading {image_name}: {e}")
                    continue
        
        print("Dataset downloaded successfully!")
        return labels_df

    def convert_to_yolo_format(self, defect_score):
        """
        Convert defect score to YOLO format label
        Args:
            defect_score: Original defect score (0 or 1)
        Returns:
            YOLO format label string
        """
        # For ELPV dataset, defect_score is binary (0 or 1)
        class_id = int(defect_score)  # 0 for no defect, 1 for defect
        
        # Center coordinates and dimensions (assuming defect covers most of the cell)
        x_center, y_center = 0.5, 0.5  # center of image
        width, height = 0.8, 0.8       # 80% of image size
        
        return f"{class_id} {x_center} {y_center} {width} {height}"

    def prepare_dataset(self, test_size=0.2, val_size=0.2, random_state=42):
        """
        Prepare the dataset by splitting into train/val/test sets and converting to YOLO format
        """
        print("Preparing dataset...")
        
        # Read labels
        labels_df = pd.read_csv(self.raw_dir / 'labels.csv', 
                              sep='\s+',
                              names=['image_path', 'defect_score', 'cell_type'])
        
        # Get image names from the image_path column
        image_names = [Path(path).name for path in labels_df['image_path']]
        
        # Split dataset
        train_val_names, test_names = train_test_split(
            image_names,
            test_size=test_size,
            random_state=random_state,
            stratify=labels_df['defect_score']
        )
        
        train_names, val_names = train_test_split(
            train_val_names,
            test_size=val_size,
            random_state=random_state,
            stratify=labels_df[labels_df['image_path'].apply(lambda x: Path(x).name).isin(train_val_names)]['defect_score']
        )
        
        # Process each split
        splits = {
            'train': (self.train_dir, train_names),
            'val': (self.val_dir, val_names),
            'test': (self.test_dir, test_names)
        }
        
        for split_name, (split_dir, split_names) in splits.items():
            print(f"Processing {split_name} split...")
            
            for image_name in split_names:
                # Copy image
                src_img = self.raw_dir / 'images' / image_name
                dst_img = split_dir / 'images' / image_name
                if src_img.exists():
                    shutil.copy2(src_img, dst_img)
                else:
                    print(f"Warning: Image {image_name} not found")
                    continue
                
                # Create YOLO format label
                defect_score = labels_df[labels_df['image_path'].str.contains(image_name)]['defect_score'].values[0]
                label_content = self.convert_to_yolo_format(defect_score)
                
                # Save label
                label_file = split_dir / 'labels' / f"{Path(image_name).stem}.txt"
                with open(label_file, 'w') as f:
                    f.write(label_content)
        
        dataset_info = {
            'train_size': len(train_names),
            'val_size': len(val_names),
            'test_size': len(test_names),
            'classes': ['no_defect', 'defect']
        }
        
        # Save dataset info
        with open(self.processed_dir / 'dataset_info.yaml', 'w') as f:
            yaml.dump(dataset_info, f)
        
        print("Dataset preparation completed!")
        return dataset_info

def main():
    # Initialize dataset preparer
    preparer = SolarDatasetPreparer()
    
    # Download dataset
    labels_df = preparer.download_elpv_dataset()
    
    # Prepare dataset
    dataset_info = preparer.prepare_dataset()
    
    print("\nDataset Statistics:")
    print(f"Train images: {dataset_info['train_size']}")
    print(f"Validation images: {dataset_info['val_size']}")
    print(f"Test images: {dataset_info['test_size']}")
    print(f"Classes: {dataset_info['classes']}")

if __name__ == "__main__":
    main()