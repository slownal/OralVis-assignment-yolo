"""
Configuration settings for the Tooth Numbering YOLO system.
"""

import os
from pathlib import Path
from typing import Dict, Any

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "ToothNumber_TaskDataset"
IMAGES_DIR = DATA_ROOT / "images"
LABELS_DIR = DATA_ROOT / "labels"

# Output directories
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUT_ROOT / "models"
RESULTS_DIR = OUTPUT_ROOT / "results"
PLOTS_DIR = OUTPUT_ROOT / "plots"
PREDICTIONS_DIR = OUTPUT_ROOT / "predictions"

# Dataset split ratios (as specified in requirements)
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# YOLO training configuration
YOLO_CONFIG = {
    "input_size": 640,  # Recommended 640x640 as specified
    "batch_size": 16,
    "epochs": 100,
    "patience": 50,
    "pretrained_weights": "yolov8s.pt",  # Default pretrained weights
    "device": "auto",  # Auto-detect GPU/CPU
}

# Model variants supported
SUPPORTED_YOLO_VARIANTS = ["yolov5", "yolov8", "yolov11"]

# FDI system constants
NUM_CLASSES = 32
FDI_QUADRANTS = {
    1: "Upper Right",
    2: "Upper Left", 
    3: "Lower Left",
    4: "Lower Right"
}

# Evaluation metrics to calculate
EVALUATION_METRICS = [
    "precision",
    "recall", 
    "mAP@50",
    "mAP@50-95",
    "confusion_matrix"
]

# Post-processing parameters (optional but recommended)
POST_PROCESSING_CONFIG = {
    "enable_anatomical_logic": True,
    "y_threshold_ratio": 0.5,  # For separating upper/lower arches
    "confidence_threshold": 0.5,
    "nms_threshold": 0.4,
    "missing_tooth_spacing_threshold": 1.5  # Relative to average tooth width
}

def create_output_directories():
    """Create all necessary output directories."""
    directories = [
        OUTPUT_ROOT,
        MODELS_DIR,
        RESULTS_DIR,
        PLOTS_DIR,
        PREDICTIONS_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        
def get_dataset_paths() -> Dict[str, Path]:
    """Get paths to dataset directories."""
    return {
        "images": IMAGES_DIR,
        "labels": LABELS_DIR,
        "train_images": OUTPUT_ROOT / "dataset" / "train" / "images",
        "train_labels": OUTPUT_ROOT / "dataset" / "train" / "labels", 
        "val_images": OUTPUT_ROOT / "dataset" / "val" / "images",
        "val_labels": OUTPUT_ROOT / "dataset" / "val" / "labels",
        "test_images": OUTPUT_ROOT / "dataset" / "test" / "images",
        "test_labels": OUTPUT_ROOT / "dataset" / "test" / "labels"
    }

def validate_dataset_structure() -> bool:
    """Validate that the dataset structure exists."""
    required_paths = [DATA_ROOT, IMAGES_DIR, LABELS_DIR]
    
    for path in required_paths:
        if not path.exists():
            print(f"Missing required path: {path}")
            return False
    
    return True