"""
YOLO Trainer for Tooth Numbering System
Implements YOLO model training with exact specifications for any YOLO variant.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
import yaml

from .config import (
 YOLO_CONFIG, SUPPORTED_YOLO_VARIANTS, 
 MODELS_DIR, OUTPUT_ROOT, RESULTS_DIR
)
from .fdi_system import FDISystem

logger = logging.getLogger(__name__)


class YOLOTrainer:
 """
 YOLO model trainer supporting multiple YOLO variants (YOLOv5, YOLOv8, YOLOv11).
 
 Implements exact specifications:
 - Support ANY YOLO variant: YOLOv5, YOLOv8, YOLOv11
 - Set recommended input size: 640x640 pixels (EXACTLY as specified)
 - Use pretrained weights (e.g., yolov8s.pt) for transfer learning
 - Implement training with proper hyperparameters and configuration
 """
 
 def __init__(self, 
 yolo_variant: str = "yolov8",
 model_size: str = "s",
 data_yaml_path: Optional[Path] = None):
 """
 Initialize YOLO trainer.
 
 Args:
 yolo_variant: YOLO variant to use ("yolov5", "yolov8", "yolov11")
 model_size: Model size ("n", "s", "m", "l", "x")
 data_yaml_path: Path to data.yaml configuration file
 """
 self.yolo_variant = yolo_variant.lower()
 self.model_size = model_size.lower()
 
 if self.yolo_variant not in SUPPORTED_YOLO_VARIANTS:
 raise ValueError(f"Unsupported YOLO variant: {yolo_variant}. Supported: {SUPPORTED_YOLO_VARIANTS}")
 
 self.data_yaml_path = data_yaml_path or (OUTPUT_ROOT / "data.yaml")
 self.model = None
 self.training_results = None
 
 # Create output directories
 MODELS_DIR.mkdir(parents=True, exist_ok=True)
 RESULTS_DIR.mkdir(parents=True, exist_ok=True)
 
 logger.info(f"Initialized {self.yolo_variant}{self.model_size} trainer")
 
 def _get_pretrained_weights_name(self) -> str:
 """Get the pretrained weights filename for the selected variant and size."""
 return f"{self.yolo_variant}{self.model_size}.pt"
 
 def _install_yolo_package(self) -> bool:
 """Install the required YOLO package if not available."""
 try:
 if self.yolo_variant in ["yolov8", "yolov11"]:
 # Try importing ultralytics (for YOLOv8 and YOLOv11)
 import ultralytics
 logger.info("✓ Ultralytics package already installed")
 return True
 elif self.yolo_variant == "yolov5":
 # YOLOv5 uses torch hub or direct repository
 import torch
 logger.info("✓ PyTorch available for YOLOv5")
 return True
 except ImportError:
 logger.warning(f"Required package for {self.yolo_variant} not found")
 
 # Provide installation instructions
 if self.yolo_variant in ["yolov8", "yolov11"]:
 logger.info("To install ultralytics: pip install ultralytics")
 elif self.yolo_variant == "yolov5":
 logger.info("To install YOLOv5 dependencies: pip install torch torchvision")
 
 return False
 
 def _load_model(self, pretrained_weights: Optional[str] = None) -> bool:
 """
 Load YOLO model with pretrained weights.
 
 Args:
 pretrained_weights: Path to pretrained weights or model name
 
 Returns:
 True if model loaded successfully, False otherwise
 """
 try:
 if not self._install_yolo_package():
 return False
 
 weights = pretrained_weights or self._get_pretrained_weights_name()
 logger.info(f"Loading {self.yolo_variant} model with weights: {weights}")
 
 if self.yolo_variant in ["yolov8", "yolov11"]:
 from ultralytics import YOLO
 self.model = YOLO(weights)
 logger.info(f"✓ Loaded {self.yolo_variant} model successfully")
 
 elif self.yolo_variant == "yolov5":
 import torch
 # Load YOLOv5 from torch hub
 self.model = torch.hub.load('ultralytics/yolov5', weights, pretrained=True)
 logger.info(f"✓ Loaded YOLOv5 model successfully")
 
 return True
 
 except Exception as e:
 logger.error(f"Failed to load {self.yolo_variant} model: {e}")
 return False
 
 def _validate_data_yaml(self) -> bool:
 """Validate the data.yaml configuration file."""
 if not self.data_yaml_path.exists():
 logger.error(f"data.yaml not found at {self.data_yaml_path}")
 return False
 
 try:
 with open(self.data_yaml_path, 'r') as f:
 data_config = yaml.safe_load(f)
 
 # Validate required keys
 required_keys = ['train', 'val', 'nc', 'names']
 for key in required_keys:
 if key not in data_config:
 logger.error(f"Missing required key in data.yaml: {key}")
 return False
 
 # Validate class count
 if data_config['nc'] != 32:
 logger.error(f"Expected 32 classes, found {data_config['nc']}")
 return False
 
 if len(data_config['names']) != 32:
 logger.error(f"Expected 32 class names, found {len(data_config['names'])}")
 return False
 
 # Validate paths exist
 for path_key in ['train', 'val']:
 path = Path(data_config[path_key])
 if not path.exists():
 logger.error(f"Path not found: {path}")
 return False
 
 logger.info("✓ data.yaml validation passed")
 return True
 
 except Exception as e:
 logger.error(f"Error validating data.yaml: {e}")
 return False
 
 def train(self,
 epochs: int = 100,
 batch_size: int = 16,
 img_size: int = 640,
 patience: int = 50,
 save_period: int = 10,
 device: str = "auto",
 **kwargs) -> bool:
 """
 Train the YOLO model with exact specifications.
 
 Args:
 epochs: Number of training epochs
 batch_size: Training batch size
 img_size: Input image size (EXACTLY 640x640 as specified)
 patience: Early stopping patience
 save_period: Save model every N epochs
 device: Training device ("auto", "cpu", "cuda", "mps")
 **kwargs: Additional training arguments
 
 Returns:
 True if training successful, False otherwise
 """
 logger.info("🚀 Starting YOLO model training")
 logger.info(f"Configuration: {self.yolo_variant}{self.model_size}, {epochs} epochs, {img_size}x{img_size} input")
 
 # Validate configuration
 if not self._validate_data_yaml():
 return False
 
 # Load model
 if not self._load_model():
 return False
 
 # Ensure input size is exactly 640x640 as specified
 if img_size != 640:
 logger.warning(f"Input size {img_size} != 640. Using recommended 640x640 as specified.")
 img_size = 640
 
 try:
 # Prepare training arguments
 train_args = {
 'data': str(self.data_yaml_path),
 'epochs': epochs,
 'batch': batch_size,
 'imgsz': img_size,
 'patience': patience,
 'save_period': save_period,
 'device': device,
 'project': str(RESULTS_DIR),
 'name': f'{self.yolo_variant}{self.model_size}_tooth_numbering',
 'exist_ok': True,
 **kwargs
 }
 
 logger.info("Training arguments:")
 for key, value in train_args.items():
 logger.info(f" {key}: {value}")
 
 # Start training
 if self.yolo_variant in ["yolov8", "yolov11"]:
 self.training_results = self.model.train(**train_args)
 
 elif self.yolo_variant == "yolov5":
 # YOLOv5 training (different API)
 logger.warning("YOLOv5 training requires manual setup. Please use YOLOv8 or YOLOv11 for automated training.")
 return False
 
 logger.info("✓ Training completed successfully!")
 
 # Save final model
 self._save_model()
 
 return True
 
 except Exception as e:
 logger.error(f"Training failed: {e}")
 return False
 
 def _save_model(self) -> None:
 """Save the trained model."""
 try:
 if self.model is None:
 logger.warning("No model to save")
 return
 
 # Save model weights
 model_path = MODELS_DIR / f"{self.yolo_variant}{self.model_size}_tooth_numbering_best.pt"
 
 if self.yolo_variant in ["yolov8", "yolov11"]:
 # The model is automatically saved during training
 # Copy the best weights to our models directory
 if self.training_results:
 best_weights = Path(self.training_results.save_dir) / "weights" / "best.pt"
 if best_weights.exists():
 import shutil
 shutil.copy2(best_weights, model_path)
 logger.info(f"✓ Model saved to {model_path}")
 else:
 logger.warning("Best weights not found in training results")
 
 except Exception as e:
 logger.error(f"Error saving model: {e}")
 
 def get_training_summary(self) -> Dict[str, Any]:
 """Get training summary and statistics."""
 if self.training_results is None:
 return {"status": "not_trained"}
 
 try:
 summary = {
 "status": "completed",
 "yolo_variant": self.yolo_variant,
 "model_size": self.model_size,
 "input_size": "640x640",
 "total_classes": 32,
 "fdi_classes": FDISystem.get_all_fdi_numbers(),
 "training_results_path": str(self.training_results.save_dir) if hasattr(self.training_results, 'save_dir') else None
 }
 
 # Add metrics if available
 if hasattr(self.training_results, 'results_dict'):
 summary.update(self.training_results.results_dict)
 
 return summary
 
 except Exception as e:
 logger.error(f"Error getting training summary: {e}")
 return {"status": "error", "error": str(e)}


def train_tooth_numbering_model(
 yolo_variant: str = "yolov8",
 model_size: str = "s", 
 epochs: int = 100,
 batch_size: int = 16,
 **kwargs) -> bool:
 """
 Main function to train tooth numbering YOLO model.
 
 Args:
 yolo_variant: YOLO variant ("yolov5", "yolov8", "yolov11")
 model_size: Model size ("n", "s", "m", "l", "x")
 epochs: Number of training epochs
 batch_size: Training batch size
 **kwargs: Additional training arguments
 
 Returns:
 True if training successful, False otherwise
 """
 logger.info(f" Starting Tooth Numbering YOLO Training - {yolo_variant}{model_size}")
 
 # Initialize trainer
 trainer = YOLOTrainer(yolo_variant=yolo_variant, model_size=model_size)
 
 # Train model
 success = trainer.train(epochs=epochs, batch_size=batch_size, **kwargs)
 
 if success:
 # Print training summary
 summary = trainer.get_training_summary()
 logger.info("Training Summary:")
 for key, value in summary.items():
 logger.info(f" {key}: {value}")
 
 logger.info(" Tooth numbering model training completed successfully!")
 else:
 logger.error("ERROR: Tooth numbering model training failed!")
 
 return success


if __name__ == "__main__":
 import sys
 from .utils import setup_logging
 
 setup_logging()
 success = train_tooth_numbering_model()
 sys.exit(0 if success else 1)