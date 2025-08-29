#!/usr/bin/env python3
"""
Dataset Preparation Script for Tooth Numbering YOLO System
Implements Task 2.1: Dataset splitting with exact ratios (80/10/10)
"""

import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from tooth_numbering_yolo.data_manager import prepare_tooth_dataset
from tooth_numbering_yolo.utils import setup_logging
from tooth_numbering_yolo.config import create_output_directories


def main():
 """Main dataset preparation function."""
 print(" Tooth Numbering YOLO - Dataset Preparation")
 print("=" * 60)
 print("Task 2.1: Implement dataset splitting with exact ratios")
 print("- Split ~500 dental panoramic images: Train (80%), Validation (10%), Test (10%)")
 print("- Ensure images and YOLO-format labels (.txt files) are paired correctly")
 print("- Validate YOLO Label Format: class_id center_x center_y width height (normalized 0-1)")
 print("- Verify all 32 FDI classes and maintain class order")
 print()
 
 # Setup logging
 setup_logging()
 
 # Create output directories
 create_output_directories()
 print("✓ Output directories created")
 
 # Prepare dataset
 success = prepare_tooth_dataset()
 
 if success:
 print("\n" + "=" * 60)
 print(" DATASET PREPARATION COMPLETE!")
 print("=" * 60)
 print(" Dataset split into Train (80%), Validation (10%), Test (10%)")
 print(" All image-label pairs validated and copied")
 print(" YOLO label format verified")
 print(" Class distribution maintained across splits")
 print(" All 32 FDI classes validated")
 print("\nNext Steps:")
 print("1. Create data.yaml configuration (Task 3)")
 print("2. Implement YOLO model training (Task 4)")
 print("3. Implement evaluation metrics (Task 5)")
 
 return True
 else:
 print("\n" + "=" * 60)
 print("ERROR: DATASET PREPARATION FAILED!")
 print("=" * 60)
 print("Please check the logs for detailed error information.")
 
 return False


if __name__ == "__main__":
 success = main()
 sys.exit(0 if success else 1)