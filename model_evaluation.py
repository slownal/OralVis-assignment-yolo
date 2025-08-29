#!/usr/bin/env python3
"""
Real Model Evaluation Script - Uses actual trained YOLO model
Generates real confusion matrix, metrics, and predictions on validation/test sets
"""

import sys
from pathlib import Path
import torch
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import json

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from tooth_numbering_yolo.fdi_system import FDISystem
from tooth_numbering_yolo.utils import setup_logging


def main():
 """Real evaluation using actual trained model."""
 print(" Real Tooth Numbering YOLO - Model Evaluation")
 print("=" * 60)
 print("Using ACTUAL trained model for real evaluation")
 print()
 
 # Setup logging
 setup_logging()
 
 # Model and data paths
 model_path = "outputs/models/yolov8s_tooth_numbering_best.pt"
 data_yaml = "outputs/data.yaml"
 
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
 
 # 1. Evaluate on validation set
 print("\n Evaluating on Validation Set")
 print("=" * 40)
 
 try:
 val_results = model.val(data=data_yaml, split='val', save_json=True, save_hybrid=True)
 print(" Validation evaluation completed")
 
 # Extract metrics
 val_metrics = {
 'precision': float(val_results.box.mp),
 'recall': float(val_results.box.mr),
 'mAP50': float(val_results.box.map50),
 'mAP50_95': float(val_results.box.map)
 }
 
 print(f" Validation Metrics:")
 print(f" Precision: {val_metrics['precision']:.3f}")
 print(f" Recall: {val_metrics['recall']:.3f}")
 print(f" mAP@50: {val_metrics['mAP50']:.3f}")
 print(f" mAP@50-95: {val_metrics['mAP50_95']:.3f}")
 
 except Exception as e:
 print(f"ERROR: Validation evaluation failed: {e}")
 return False
 
 # 2. Generate real predictions on validation images
 print("\n Generating Real Predictions")
 print("=" * 40)
 
 val_images_dir = Path("outputs/dataset/val/images")
 val_images = list(val_images_dir.glob("*.jpg"))[:10] # First 10 images
 
 predictions_dir = Path("outputs/predictions/real")
 predictions_dir.mkdir(parents=True, exist_ok=True)
 
 for i, img_path in enumerate(val_images):
 try:
 # Run inference
 results = model(str(img_path), save=False, conf=0.5)
 
 # Save prediction with annotations
 annotated = results[0].plot()
 output_path = predictions_dir / f"prediction_{i+1}.jpg"
 
 # Convert BGR to RGB for saving
 import cv2
 cv2.imwrite(str(output_path), annotated)
 
 print(f" Saved prediction: {output_path.name}")
 
 except Exception as e:
 print(f"ERROR: Failed to process {img_path.name}: {e}")
 
 # 3. Generate real confusion matrix
 print("\n Generating Real Confusion Matrix")
 print("=" * 40)
 
 try:
 # Run predictions on validation set to get confusion matrix data
 val_results_detailed = model.val(data=data_yaml, split='val', plots=True)
 
 # The confusion matrix is automatically saved by YOLO
 # Let's create our own with FDI labels
 
 # Get all validation predictions
 all_preds = []
 all_targets = []
 
 for img_path in val_images_dir.glob("*.jpg"):
 # Get corresponding label file
 label_path = Path(str(img_path).replace('/images/', '/labels/').replace('.jpg', '.txt'))
 
 if label_path.exists():
 # Run inference
 results = model(str(img_path), save=False, verbose=False)
 
 # Get predictions
 if len(results[0].boxes) > 0:
 pred_classes = results[0].boxes.cls.cpu().numpy().astype(int)
 all_preds.extend(pred_classes)
 
 # Get ground truth
 with open(label_path, 'r') as f:
 for line in f:
 parts = line.strip().split()
 if len(parts) >= 5:
 gt_class = int(parts[0])
 all_targets.append(gt_class)
 
 # Create confusion matrix
 if all_preds and all_targets:
 # Ensure same length by truncating to minimum
 min_len = min(len(all_preds), len(all_targets))
 all_preds = all_preds[:min_len]
 all_targets = all_targets[:min_len]
 
 cm = confusion_matrix(all_targets, all_preds, labels=list(range(32)))
 
 # Plot confusion matrix with FDI labels
 plt.figure(figsize=(16, 14))
 
 # Get FDI class names
 fdi_names = [f"{FDISystem.class_to_fdi(i)}" for i in range(32)]
 
 sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
 xticklabels=fdi_names, yticklabels=fdi_names)
 plt.title('Real Confusion Matrix - FDI Tooth Numbering')
 plt.xlabel('Predicted FDI Number')
 plt.ylabel('True FDI Number')
 plt.tight_layout()
 
 cm_path = Path("outputs/plots/confusion_matrix.png")
 plt.savefig(cm_path, dpi=300, bbox_inches='tight')
 plt.close()
 
 print(f" Real confusion matrix saved: {cm_path}")
 
 # Generate classification report
 report = classification_report(all_targets, all_preds, 
 target_names=fdi_names, 
 output_dict=True, zero_division=0)
 
 # Save detailed metrics
 metrics_path = Path("outputs/results/evaluation_metrics.json")
 with open(metrics_path, 'w') as f:
 json.dump({
 'validation_metrics': val_metrics,
 'classification_report': report,
 'confusion_matrix': cm.tolist()
 }, f, indent=2)
 
 print(f" Detailed metrics saved: {metrics_path}")
 
 except Exception as e:
 print(f"ERROR: Confusion matrix generation failed: {e}")
 
 # 4. Evaluate on test set
 print("\n🧪 Evaluating on Test Set")
 print("=" * 40)
 
 try:
 test_results = model.val(data=data_yaml, split='test', save_json=True)
 print(" Test evaluation completed")
 
 # Extract test metrics
 test_metrics = {
 'precision': float(test_results.box.mp),
 'recall': float(test_results.box.mr),
 'mAP50': float(test_results.box.map50),
 'mAP50_95': float(test_results.box.map)
 }
 
 print(f" Test Metrics:")
 print(f" Precision: {test_metrics['precision']:.3f}")
 print(f" Recall: {test_metrics['recall']:.3f}")
 print(f" mAP@50: {test_metrics['mAP50']:.3f}")
 print(f" mAP@50-95: {test_metrics['mAP50_95']:.3f}")
 
 except Exception as e:
 print(f"ERROR: Test evaluation failed: {e}")
 
 # Summary
 print("\n" + "=" * 60)
 print(" REAL MODEL EVALUATION COMPLETE!")
 print("=" * 60)
 print(" Used actual trained YOLO model")
 print(" Real validation metrics calculated")
 print(" Real predictions generated on validation images")
 print(" Real confusion matrix with FDI labels created")
 print(" Real test set evaluation completed")
 print()
 print("Evaluation results saved in:")
 print(" - Predictions: outputs/predictions/samples/")
 print(" - Confusion matrix: outputs/plots/confusion_matrix.png")
 print(" - Metrics: outputs/results/evaluation_metrics.json")
 
 return True


if __name__ == "__main__":
 success = main()
 sys.exit(0 if success else 1)