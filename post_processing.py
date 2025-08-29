#!/usr/bin/env python3
"""
Real Post-Processing Script - Uses actual trained YOLO model
Applies anatomical post-processing to real model predictions
"""

import sys
from pathlib import Path
import torch
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from tooth_numbering_yolo.post_processor import AnatomicalPostProcessor, ToothPrediction
from tooth_numbering_yolo.fdi_system import FDISystem
from tooth_numbering_yolo.utils import setup_logging


def main():
 """Real post-processing using actual trained model."""
 print(" Real Tooth Numbering YOLO - Post-Processing")
 print("=" * 60)
 print("Using ACTUAL trained model for real post-processing")
 print()
 
 # Setup logging
 setup_logging()
 
 # Model path
 model_path = "outputs/models/yolov8s_tooth_numbering_best.pt"
 
 if not Path(model_path).exists():
 print(f"ERROR: Model not found: {model_path}")
 print("Please run training first: python train_yolo_model.py")
 return False
 
 print(f" Loading model: {model_path}")
 
 # Load the trained model
 try:
 model = YOLO(model_path)
 print(" Model loaded successfully")
 except Exception as e:
 print(f"ERROR: Failed to load model: {e}")
 return False
 
 # Initialize post-processor
 post_processor = AnatomicalPostProcessor()
 
 # Get validation images
 val_images_dir = Path("outputs/dataset/val/images")
 val_images = list(val_images_dir.glob("*.jpg"))[:5] # Process 5 real images
 
 if not val_images:
 print("ERROR: No validation images found")
 return False
 
 print(f"📸 Processing {len(val_images)} real validation images...")
 
 # Create output directory
 output_dir = Path("outputs/predictions/post_processed")
 output_dir.mkdir(parents=True, exist_ok=True)
 
 processed_count = 0
 total_corrections = 0
 
 for i, image_path in enumerate(val_images):
 print(f"\n Processing: {image_path.name}")
 
 try:
 # Run inference with the real model
 results = model(str(image_path), save=False, conf=0.3, verbose=False)
 
 if len(results[0].boxes) == 0:
 print(f" WARNING: No teeth detected in {image_path.name}")
 continue
 
 # Extract predictions
 boxes = results[0].boxes
 raw_predictions = []
 
 for j in range(len(boxes)):
 # Get bounding box in normalized format
 bbox = boxes.xywhn[j].cpu().numpy() # normalized x_center, y_center, width, height
 confidence = float(boxes.conf[j].cpu())
 class_id = int(boxes.cls[j].cpu())
 
 raw_predictions.append({
 'bbox': tuple(bbox),
 'confidence': confidence,
 'class_id': class_id
 })
 
 print(f" Detected {len(raw_predictions)} teeth")
 
 # Apply anatomical post-processing
 processed_predictions = post_processor.process_predictions(
 predictions=raw_predictions,
 image_width=640,
 image_height=640
 )
 
 print(f" Post-processed to {len(processed_predictions)} teeth")
 
 # Calculate corrections applied
 corrections = len(raw_predictions) - len(processed_predictions)
 if corrections != 0:
 total_corrections += abs(corrections)
 print(f" Applied {abs(corrections)} anatomical corrections")
 
 # Show quadrant distribution
 quadrants = {1: 0, 2: 0, 3: 0, 4: 0}
 fdi_numbers = []
 
 for pred in processed_predictions:
 quadrants[pred.quadrant] += 1
 fdi_numbers.append(pred.fdi_number)
 
 print(f" Quadrant 1 (Upper Right): {quadrants[1]} teeth")
 print(f" Quadrant 2 (Upper Left): {quadrants[2]} teeth")
 print(f" Quadrant 3 (Lower Left): {quadrants[3]} teeth")
 print(f" Quadrant 4 (Lower Right): {quadrants[4]} teeth")
 print(f" FDI numbers: {sorted(fdi_numbers)}")
 
 # Create visualization comparing before/after
 img = cv2.imread(str(image_path))
 img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 
 fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
 
 # Before post-processing (raw predictions)
 ax1.imshow(img_rgb)
 ax1.set_title(f'Before Post-Processing\n{len(raw_predictions)} detections', fontsize=14)
 
 for pred in raw_predictions:
 bbox = pred['bbox']
 class_id = pred['class_id']
 confidence = pred['confidence']
 fdi_num = FDISystem.class_to_fdi(class_id)
 
 # Convert normalized coordinates to pixel coordinates
 h, w = img_rgb.shape[:2]
 x_center, y_center, width, height = bbox
 x1 = int((x_center - width/2) * w)
 y1 = int((y_center - height/2) * h)
 x2 = int((x_center + width/2) * w)
 y2 = int((y_center + height/2) * h)
 
 # Draw bounding box
 rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
 fill=False, color='red', linewidth=2)
 ax1.add_patch(rect)
 
 # Add label
 ax1.text(x1, y1-5, f'FDI {fdi_num}\n{confidence:.2f}', 
 color='red', fontsize=10, fontweight='bold',
 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
 
 ax1.axis('off')
 
 # After post-processing (anatomically corrected)
 ax2.imshow(img_rgb)
 ax2.set_title(f'After Post-Processing\n{len(processed_predictions)} anatomically correct', fontsize=14)
 
 for pred in processed_predictions:
 bbox = pred.bbox
 confidence = pred.confidence
 fdi_num = pred.fdi_number
 
 # Convert normalized coordinates to pixel coordinates
 h, w = img_rgb.shape[:2]
 x_center, y_center, width, height = bbox
 x1 = int((x_center - width/2) * w)
 y1 = int((y_center - height/2) * h)
 x2 = int((x_center + width/2) * w)
 y2 = int((y_center + height/2) * h)
 
 # Color by quadrant
 colors = {1: 'blue', 2: 'green', 3: 'orange', 4: 'purple'}
 color = colors.get(pred.quadrant, 'red')
 
 # Draw bounding box
 rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
 fill=False, color=color, linewidth=2)
 ax2.add_patch(rect)
 
 # Add label
 ax2.text(x1, y1-5, f'FDI {fdi_num}\nQ{pred.quadrant}', 
 color=color, fontsize=10, fontweight='bold',
 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
 
 ax2.axis('off')
 
 # Add legend
 legend_elements = [
 plt.Line2D([0], [0], color='blue', lw=2, label='Q1 (Upper Right)'),
 plt.Line2D([0], [0], color='green', lw=2, label='Q2 (Upper Left)'),
 plt.Line2D([0], [0], color='orange', lw=2, label='Q3 (Lower Left)'),
 plt.Line2D([0], [0], color='purple', lw=2, label='Q4 (Lower Right)')
 ]
 ax2.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
 
 plt.tight_layout()
 
 # Save comparison
 output_path = output_dir / f"post_processing_comparison_{i+1}.png"
 plt.savefig(output_path, dpi=300, bbox_inches='tight')
 plt.close()
 
 print(f" Saved comparison: {output_path.name}")
 processed_count += 1
 
 except Exception as e:
 print(f" ERROR: Failed to process {image_path.name}: {e}")
 
 # Performance analysis
 print(f"\n Real Post-Processing Performance Analysis")
 print("=" * 40)
 
 if processed_count > 0:
 print(f" Successfully processed {processed_count}/{len(val_images)} real images")
 print(f" Success rate: {(processed_count/len(val_images)*100):.1f}%")
 print(f" Total anatomical corrections applied: {total_corrections}")
 print(f" Average corrections per image: {total_corrections/processed_count:.1f}")
 
 # Demonstrate anatomical logic with real data
 print(f"\n Real Anatomical Logic Demonstration")
 print("=" * 40)
 
 # Process one image in detail to show the logic
 if val_images:
 demo_image = val_images[0]
 print(f" Detailed analysis of: {demo_image.name}")
 
 # Run inference
 results = model(str(demo_image), save=False, conf=0.3, verbose=False)
 
 if len(results[0].boxes) > 0:
 # Extract predictions
 boxes = results[0].boxes
 demo_predictions = []
 
 for j in range(len(boxes)):
 bbox = boxes.xywhn[j].cpu().numpy()
 confidence = float(boxes.conf[j].cpu())
 class_id = int(boxes.cls[j].cpu())
 
 demo_predictions.append({
 'bbox': tuple(bbox),
 'confidence': confidence,
 'class_id': class_id
 })
 
 print(f" Raw detections: {len(demo_predictions)}")
 
 # Apply post-processing with detailed logging
 processed = post_processor.process_predictions(
 predictions=demo_predictions,
 image_width=640,
 image_height=640
 )
 
 print(f" Final predictions: {len(processed)}")
 
 # Show arch separation
 upper_teeth = [p for p in processed if p.quadrant in [1, 2]]
 lower_teeth = [p for p in processed if p.quadrant in [3, 4]]
 
 print(f" Upper arch: {len(upper_teeth)} teeth")
 print(f" Lower arch: {len(lower_teeth)} teeth")
 
 # Show sequential FDI assignment by quadrant
 for quadrant in [1, 2, 3, 4]:
 quad_teeth = [p for p in processed if p.quadrant == quadrant]
 quad_fdi = sorted([p.fdi_number for p in quad_teeth])
 print(f" Quadrant {quadrant} FDI sequence: {quad_fdi}")
 
 # Summary
 print("\n" + "=" * 60)
 print(" REAL POST-PROCESSING COMPLETE!")
 print("=" * 60)
 print(" Used actual trained YOLO model")
 print(" Applied real anatomical post-processing")
 print(" Generated before/after comparisons")
 print(" Demonstrated arch separation and quadrant division")
 print(" Applied sequential FDI assignment")
 print(" Handled missing teeth and anatomical corrections")
 print()
 print("Post-processing results saved in:")
 print(" - Comparisons: outputs/predictions/post_processed/")
 
 return True


if __name__ == "__main__":
 success = main()
 sys.exit(0 if success else 1)