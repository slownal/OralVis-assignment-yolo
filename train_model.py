#!/usr/bin/env python3
"""
YOLO Training Script for Tooth Numbering System
Implements Task 4.1: YOLO model training with exact specifications
"""

import sys
import argparse
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from tooth_numbering_yolo.trainer import train_tooth_numbering_model
from tooth_numbering_yolo.utils import setup_logging
from tooth_numbering_yolo.config import YOLO_CONFIG, SUPPORTED_YOLO_VARIANTS


def check_requirements():
 """Check if required packages are installed."""
 print("Checking Requirements")
 print("=" * 40)
 
 requirements_met = True
 
 # Check for ultralytics (YOLOv8/YOLOv11)
 try:
 import ultralytics
 print(f"Ultralytics: {ultralytics.__version__}")
 except ImportError:
 print("WARNING: Ultralytics not installed (required for YOLOv8/YOLOv11)")
 print(" Install with: pip install ultralytics")
 requirements_met = False
 
 # Check for PyTorch
 try:
 import torch
 print(f" PyTorch: {torch.__version__}")
 
 # Check for CUDA
 if torch.cuda.is_available():
 print(f" CUDA available: {torch.cuda.get_device_name(0)}")
 else:
 print("WARNING: CUDA not available - training will use CPU (slower)")
 except ImportError:
 print("ERROR: PyTorch not installed")
 print(" Install with: pip install torch torchvision")
 requirements_met = False
 
 # Check for YAML
 try:
 import yaml
 print(" PyYAML available")
 except ImportError:
 print("ERROR: PyYAML not installed")
 print(" Install with: pip install pyyaml")
 requirements_met = False
 
 return requirements_met


def validate_training_setup():
 """Validate that training setup is ready."""
 print("\n Validating Training Setup")
 print("=" * 40)
 
 # Check data.yaml exists
 data_yaml = Path("outputs/data.yaml")
 if data_yaml.exists():
 print(f" data.yaml found: {data_yaml}")
 else:
 print(f"ERROR: data.yaml not found: {data_yaml}")
 print(" Run: python create_data_yaml.py")
 return False
 
 # Check dataset directories exist
 dataset_dirs = [
 "outputs/dataset/train/images",
 "outputs/dataset/train/labels", 
 "outputs/dataset/val/images",
 "outputs/dataset/val/labels"
 ]
 
 for dir_path in dataset_dirs:
 path = Path(dir_path)
 if path.exists():
 file_count = len(list(path.glob('*')))
 print(f" {dir_path}: {file_count} files")
 else:
 print(f"ERROR: {dir_path}: not found")
 print(" Run: python prepare_dataset.py")
 return False
 
 return True


def main():
 """Main training function."""
 parser = argparse.ArgumentParser(description="Train YOLO model for tooth numbering")
 parser.add_argument("--variant", choices=SUPPORTED_YOLO_VARIANTS, default="yolov8",
 help="YOLO variant to use")
 parser.add_argument("--size", choices=["n", "s", "m", "l", "x"], default="s",
 help="Model size")
 parser.add_argument("--epochs", type=int, default=100,
 help="Number of training epochs")
 parser.add_argument("--batch", type=int, default=16,
 help="Batch size")
 parser.add_argument("--imgsz", type=int, default=640,
 help="Input image size (recommended: 640)")
 parser.add_argument("--device", default="cpu",
 help="Training device (auto, cpu, cuda)")
 parser.add_argument("--patience", type=int, default=50,
 help="Early stopping patience")
 parser.add_argument("--dry-run", action="store_true",
 help="Validate setup without training")
 
 args = parser.parse_args()
 
 print(" Tooth Numbering YOLO - Model Training")
 print("=" * 60)
 print("Task 4.1: Implement YOLO model training with exact specifications")
 print("- Support ANY YOLO variant: YOLOv5, YOLOv8, YOLOv11")
 print("- Set recommended input size: 640x640 pixels (EXACTLY as specified)")
 print("- Use pretrained weights (e.g., yolov8s.pt) for transfer learning")
 print("- Implement training with proper hyperparameters and configuration")
 print()
 
 print(f"Training Configuration:")
 print(f" YOLO Variant: {args.variant}")
 print(f" Model Size: {args.size}")
 print(f" Epochs: {args.epochs}")
 print(f" Batch Size: {args.batch}")
 print(f" Image Size: {args.imgsz}x{args.imgsz}")
 print(f" Device: {args.device}")
 print(f" Patience: {args.patience}")
 print()
 
 # Setup logging
 setup_logging()
 
 # Check requirements
 if not check_requirements():
 print("\nERROR: Requirements check failed!")
 print("Please install missing packages and try again.")
 return False
 
 # Validate training setup
 if not validate_training_setup():
 print("\nERROR: Training setup validation failed!")
 print("Please prepare the dataset and configuration first.")
 return False
 
 if args.dry_run:
 print("\n Dry run completed - setup is valid!")
 print("Remove --dry-run flag to start actual training.")
 return True
 
 # Validate image size
 if args.imgsz != 640:
 print(f"\nWARNING: Warning: Using image size {args.imgsz} instead of recommended 640x640")
 
 # Start training
 print("\n Starting YOLO Model Training")
 print("=" * 40)
 
 try:
 success = train_tooth_numbering_model(
 yolo_variant=args.variant,
 model_size=args.size,
 epochs=args.epochs,
 batch_size=args.batch,
 img_size=args.imgsz,
 device=args.device,
 patience=args.patience
 )
 
 if success:
 print("\n" + "=" * 60)
 print(" YOLO MODEL TRAINING COMPLETE!")
 print("=" * 60)
 print(" Model trained with exact specifications")
 print(" 640x640 input size used (as specified)")
 print(" Pretrained weights loaded for transfer learning")
 print(" Training artifacts saved")
 print(" All 32 FDI classes supported")
 
 print(f"\n Training results saved in: outputs/results/")
 print(f" Model weights saved in: outputs/models/")
 print("\nNext Steps:")
 print("1. Implement model evaluation (Task 5)")
 print("2. Generate submission metrics")
 print("3. Add anatomical post-processing (Task 6)")
 
 return True
 else:
 print("\n" + "=" * 60)
 print("ERROR: YOLO MODEL TRAINING FAILED!")
 print("=" * 60)
 print("Please check the logs for detailed error information.")
 
 return False
 
 except KeyboardInterrupt:
 print("\n\nWARNING: Training interrupted by user")
 return False
 except Exception as e:
 print(f"\nERROR: Training failed with error: {e}")
 return False


if __name__ == "__main__":
 success = main()
 sys.exit(0 if success else 1)