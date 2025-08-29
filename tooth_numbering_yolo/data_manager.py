"""
Data Manager for Tooth Numbering YOLO System
Handles dataset organization, splitting, and validation according to exact specifications.
"""

import os
import random
import shutil
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Set
from collections import defaultdict

from .config import (
 DATA_ROOT, IMAGES_DIR, LABELS_DIR, 
 TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
 get_dataset_paths
)
from .utils import (
 get_image_label_pairs, 
 copy_file_pairs, 
 validate_yolo_label_format,
 validate_class_distribution,
 check_missing_classes
)
from .fdi_system import FDISystem

logger = logging.getLogger(__name__)


class DataManager:
 """
 Manages dataset preparation and splitting for YOLO training.
 
 Implements exact specifications:
 - Split ~500 dental panoramic images: Train (80%), Validation (10%), Test (10%)
 - Ensure images and YOLO-format labels (.txt files) are paired correctly
 - Validate each image has corresponding .txt file with bounding boxes using FDI numbering
 - Verify YOLO Label Format: class_id center_x center_y width height (normalized 0-1)
 """
 
 def __init__(self):
 self.dataset_paths = get_dataset_paths()
 self.image_label_pairs = []
 self.class_distribution = {}
 
 def validate_dataset_structure(self) -> bool:
 """
 Validate that the dataset structure exists and is correct.
 
 Returns:
 True if dataset structure is valid, False otherwise
 """
 logger.info("Validating dataset structure...")
 
 # Check if required directories exist
 required_dirs = [DATA_ROOT, IMAGES_DIR, LABELS_DIR]
 for directory in required_dirs:
 if not directory.exists():
 logger.error(f"Required directory not found: {directory}")
 return False
 
 # Get image-label pairs
 self.image_label_pairs = get_image_label_pairs(IMAGES_DIR, LABELS_DIR)
 
 if len(self.image_label_pairs) == 0:
 logger.error("No valid image-label pairs found!")
 return False
 
 logger.info(f"Found {len(self.image_label_pairs)} valid image-label pairs")
 
 # Validate YOLO label format for all files and filter out invalid ones
 valid_pairs = []
 invalid_labels = []
 
 for image_path, label_path in self.image_label_pairs:
 if validate_yolo_label_format(label_path):
 valid_pairs.append((image_path, label_path))
 else:
 invalid_labels.append(label_path)
 
 if invalid_labels:
 logger.warning(f"Found {len(invalid_labels)} invalid label files (likely polygon annotations):")
 for label_file in invalid_labels[:5]: # Show first 5
 logger.warning(f" - {label_file.name}")
 if len(invalid_labels) > 5:
 logger.warning(f" ... and {len(invalid_labels) - 5} more")
 logger.warning("These files will be excluded from the dataset (polygon annotations not supported)")
 
 # Update pairs to only include valid ones
 self.image_label_pairs = valid_pairs
 logger.info(f"Using {len(valid_pairs)} valid image-label pairs (excluded {len(invalid_labels)} with polygon annotations)")
 
 logger.info("✓ All label files have valid YOLO format")
 
 # Validate class distribution
 self.class_distribution = validate_class_distribution(LABELS_DIR)
 missing_classes = check_missing_classes(self.class_distribution)
 
 if missing_classes:
 logger.warning(f"Missing {len(missing_classes)} classes in dataset")
 # This is a warning, not an error - we can still proceed
 else:
 logger.info("✓ All 32 FDI classes are present in the dataset")
 
 return True
 
 def split_dataset(self, 
 train_ratio: float = TRAIN_RATIO,
 val_ratio: float = VAL_RATIO, 
 test_ratio: float = TEST_RATIO,
 random_seed: int = 42) -> Dict[str, List[Tuple[Path, Path]]]:
 """
 Split dataset into train/validation/test sets with exact ratios.
 
 Args:
 train_ratio: Training set ratio (default: 0.8)
 val_ratio: Validation set ratio (default: 0.1)
 test_ratio: Test set ratio (default: 0.1)
 random_seed: Random seed for reproducible splits
 
 Returns:
 Dictionary with 'train', 'val', 'test' keys containing file pairs
 """
 logger.info(f"Splitting dataset with ratios - Train: {train_ratio}, Val: {val_ratio}, Test: {test_ratio}")
 
 # Validate ratios
 total_ratio = train_ratio + val_ratio + test_ratio
 if abs(total_ratio - 1.0) > 1e-6:
 raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
 
 # Set random seed for reproducible splits
 random.seed(random_seed)
 
 # Shuffle the pairs
 pairs = self.image_label_pairs.copy()
 random.shuffle(pairs)
 
 total_samples = len(pairs)
 train_count = int(total_samples * train_ratio)
 val_count = int(total_samples * val_ratio)
 test_count = total_samples - train_count - val_count # Ensure all samples are used
 
 logger.info(f"Dataset split: {total_samples} total samples")
 logger.info(f" Train: {train_count} samples ({train_count/total_samples:.1%})")
 logger.info(f" Val: {val_count} samples ({val_count/total_samples:.1%})")
 logger.info(f" Test: {test_count} samples ({test_count/total_samples:.1%})")
 
 # Split the data
 train_pairs = pairs[:train_count]
 val_pairs = pairs[train_count:train_count + val_count]
 test_pairs = pairs[train_count + val_count:]
 
 splits = {
 'train': train_pairs,
 'val': val_pairs,
 'test': test_pairs
 }
 
 # Validate class distribution across splits
 self._validate_split_distribution(splits)
 
 return splits
 
 def _validate_split_distribution(self, splits: Dict[str, List[Tuple[Path, Path]]]) -> None:
 """
 Validate that class distribution is maintained across train/val/test splits.
 
 Args:
 splits: Dictionary containing train/val/test splits
 """
 logger.info("Validating class distribution across splits...")
 
 split_distributions = {}
 
 for split_name, pairs in splits.items():
 class_counts = defaultdict(int)
 
 for _, label_path in pairs:
 try:
 with open(label_path, 'r') as f:
 for line in f:
 line = line.strip()
 if line:
 class_id = int(line.split()[0])
 if 0 <= class_id <= 31:
 class_counts[class_id] += 1
 except Exception as e:
 logger.error(f"Error reading {label_path}: {e}")
 
 split_distributions[split_name] = dict(class_counts)
 
 # Log distribution for this split
 total_instances = sum(class_counts.values())
 logger.info(f"{split_name.capitalize()} split: {len(pairs)} images, {total_instances} tooth instances")
 
 # Check for classes missing in any split (warning only)
 all_classes = set(range(32))
 for split_name, distribution in split_distributions.items():
 present_classes = set(distribution.keys())
 missing_classes = all_classes - present_classes
 
 if missing_classes:
 logger.warning(f"{split_name.capitalize()} split missing classes: {sorted(missing_classes)}")
 else:
 logger.info(f"✓ {split_name.capitalize()} split has all 32 classes")
 
 def copy_splits_to_directories(self, splits: Dict[str, List[Tuple[Path, Path]]]) -> None:
 """
 Copy split data to organized train/val/test directories.
 
 Args:
 splits: Dictionary containing train/val/test splits
 """
 logger.info("Copying splits to organized directories...")
 
 for split_name, pairs in splits.items():
 # Get destination directories
 dest_images_dir = self.dataset_paths[f'{split_name}_images']
 dest_labels_dir = self.dataset_paths[f'{split_name}_labels']
 
 logger.info(f"Copying {len(pairs)} pairs to {split_name} split...")
 
 # Copy files
 copy_file_pairs(pairs, dest_images_dir, dest_labels_dir)
 
 logger.info(f"✓ {split_name.capitalize()} split copied successfully")
 
 def prepare_dataset(self) -> bool:
 """
 Complete dataset preparation pipeline.
 
 Returns:
 True if preparation successful, False otherwise
 """
 logger.info("Starting dataset preparation pipeline...")
 
 try:
 # Step 1: Validate dataset structure
 if not self.validate_dataset_structure():
 logger.error("Dataset structure validation failed!")
 return False
 
 # Step 2: Split dataset
 splits = self.split_dataset()
 
 # Step 3: Copy splits to organized directories
 self.copy_splits_to_directories(splits)
 
 # Step 4: Final validation
 self._final_validation()
 
 logger.info("✓ Dataset preparation completed successfully!")
 return True
 
 except Exception as e:
 logger.error(f"Dataset preparation failed: {e}")
 return False
 
 def _final_validation(self) -> None:
 """Perform final validation of the prepared dataset."""
 logger.info("Performing final validation...")
 
 for split_name in ['train', 'val', 'test']:
 images_dir = self.dataset_paths[f'{split_name}_images']
 labels_dir = self.dataset_paths[f'{split_name}_labels']
 
 if not images_dir.exists() or not labels_dir.exists():
 raise ValueError(f"Missing {split_name} directories")
 
 # Count files
 image_count = len(list(images_dir.glob('*')))
 label_count = len(list(labels_dir.glob('*.txt')))
 
 if image_count != label_count:
 raise ValueError(f"{split_name} split: image count ({image_count}) != label count ({label_count})")
 
 logger.info(f"✓ {split_name.capitalize()} split: {image_count} image-label pairs")
 
 def get_dataset_statistics(self) -> Dict[str, any]:
 """
 Get comprehensive dataset statistics.
 
 Returns:
 Dictionary containing dataset statistics
 """
 stats = {
 'total_images': len(self.image_label_pairs),
 'class_distribution': self.class_distribution,
 'missing_classes': check_missing_classes(self.class_distribution),
 'total_instances': sum(self.class_distribution.values()),
 'classes_present': len([c for c in self.class_distribution.values() if c > 0])
 }
 
 # Add FDI mapping information
 stats['fdi_mapping'] = {}
 for class_id, count in self.class_distribution.items():
 if count > 0:
 fdi_number = FDISystem.class_to_fdi(class_id)
 tooth_type = FDISystem.get_tooth_type_name(fdi_number)
 stats['fdi_mapping'][class_id] = {
 'fdi_number': fdi_number,
 'tooth_type': tooth_type,
 'count': count
 }
 
 return stats
 
 def print_dataset_summary(self) -> None:
 """Print a comprehensive dataset summary."""
 stats = self.get_dataset_statistics()
 
 print("\n" + "="*60)
 print(" DATASET SUMMARY")
 print("="*60)
 print(f"Total Images: {stats['total_images']}")
 print(f"Total Tooth Instances: {stats['total_instances']}")
 print(f"Classes Present: {stats['classes_present']}/32")
 
 if stats['missing_classes']:
 print(f"Missing Classes: {len(stats['missing_classes'])}")
 for class_id in stats['missing_classes'][:5]: # Show first 5
 fdi_number = FDISystem.class_to_fdi(class_id)
 tooth_type = FDISystem.get_tooth_type_name(fdi_number)
 print(f" - Class {class_id}: {tooth_type} ({fdi_number})")
 else:
 print("✓ All 32 FDI classes are present!")
 
 print("\nClass Distribution (Top 10):")
 sorted_classes = sorted(stats['fdi_mapping'].items(), 
 key=lambda x: x[1]['count'], reverse=True)
 
 for class_id, info in sorted_classes[:10]:
 print(f" Class {class_id:2d}: {info['tooth_type']} ({info['fdi_number']}) - {info['count']} instances")
 
 if len(sorted_classes) > 10:
 print(f" ... and {len(sorted_classes) - 10} more classes")


def prepare_tooth_dataset() -> bool:
 """
 Main function to prepare the tooth numbering dataset.
 
 Returns:
 True if preparation successful, False otherwise
 """
 logger.info(" Starting Tooth Numbering Dataset Preparation")
 
 # Initialize data manager
 data_manager = DataManager()
 
 # Print initial dataset summary
 if data_manager.validate_dataset_structure():
 data_manager.print_dataset_summary()
 
 # Prepare dataset
 success = data_manager.prepare_dataset()
 
 if success:
 logger.info(" Dataset preparation completed successfully!")
 else:
 logger.error("ERROR: Dataset preparation failed!")
 
 return success


if __name__ == "__main__":
 import sys
 from .utils import setup_logging
 
 setup_logging()
 success = prepare_tooth_dataset()
 sys.exit(0 if success else 1)