import torch
import cv2
import numpy as np
from pathlib import Path
import yaml
import subprocess

class DefectDetector:
    def __init__(self, weights_path='runs/train/solar_defect_detection/weights/best.pt'):
        """
        Initialize the detector with trained weights
        """
        self.weights_path = weights_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.setup_model()
        
    def setup_model(self):
        """
        Setup YOLOv5 model
        """
        if not Path('yolov5').exists():
            subprocess.run(['git', 'clone', 'https://github.com/ultralytics/yolov5.git'])
        
        self.model = torch.hub.load('yolov5', 'custom', path=self.weights_path, source='local')
        self.model.to(self.device)
        self.model.eval()
    
    def detect(self, image_path, conf_thresh=0.25):
        """
        Detect defects in an image
        """
        # Run inference
        results = self.model(image_path)
        
        # Get predictions
        pred = results.pred[0]
        pred = pred[pred[:, 4] > conf_thresh]  # Filter by confidence
        
        return results, pred
    
    def draw_results(self, image_path, results, output_path=None):
        """
        Draw detection results on image
        """
        # Get class names
        names = self.model.names
        
        # Draw results
        img = results.render()[0]
        
        if output_path:
            cv2.imwrite(output_path, img)
        
        return img
    
    def process_directory(self, input_dir, output_dir):
        """
        Process all images in a directory
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        for img_path in input_dir.glob('*.png'):
            # Detect defects
            results, pred = self.detect(str(img_path))
            
            # Draw results
            output_path = output_dir / f'detected_{img_path.name}'
            self.draw_results(img_path, results, str(output_path))
            
            print(f"Processed {img_path.name}")

def main():
    # Initialize detector
    detector = DefectDetector()
    
    # Process test directory
    test_dir = r'C:\Users\Eduar\Documents\masters\cap5415\solar_project\scripts\solar_panel_data\processed\test\images'
    output_dir = 'results'
    
    print("Processing test images...")
    detector.process_directory(test_dir, output_dir)
    print(f"Results saved to {output_dir}")

if __name__ == '__main__':
    main()