"""
Utility functions for the Tooth Numbering YOLO system.
"""

import os
import logging
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Any
import yaml

from .fdi_system import FDISystem
from .config import get_dataset_paths, OUTPUT_ROOT

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO") -> None:
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    # Ensure output directory exists
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(OUTPUT_ROOT / "tooth_numbering.log", encoding='utf-8')
        ]
    )


def create_data_yaml(output_path: Path = None) -> Path:
    """
    Create the data.yaml file required for YOLO training.
    
    Args:
        output_path: Path where to save the data.yaml file
        
    Returns:
        Path to the created data.yaml file
    """
    if output_path is None:
        output_path = OUTPUT_ROOT / "data.yaml"
    
    # Get dataset paths
    paths = get_dataset_paths()
    
    # Get class names in exact order from FDI system
    class_names = FDISystem.get_class_names()
    
    # Create data.yaml content
    data_config = {
        'train': str(paths['train_images']),
        'val': str(paths['val_images']),
        'test': str(paths['test_images']),
        'nc': 32,  # Number of classes
        'names': class_names
    }
    
    # Write YAML file
    with open(output_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Created data.yaml at {output_path}")
    return output_path


def validate_yolo_label_format(label_file: Path) -> bool:
    """
    Validate YOLO label file format.
    
    Args:
        label_file: Path to the label file
        
    Returns:
        True if format is valid, False otherwise
    """
    try:
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
                
            parts = line.split()
            if len(parts) != 5:
                logger.error(f"Invalid format in {label_file}:{line_num} - Expected 5 values, got {len(parts)}")
                return False
            
            try:
                class_id = int(parts[0])
                center_x = float(parts[1])
                center_y = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # Validate class_id range
                if not (0 <= class_id <= 31):
                    logger.error(f"Invalid class_id {class_id} in {label_file}:{line_num} - Must be 0-31")
                    return False
                
                # Validate normalized coordinates (0-1)
                for coord_name, coord_value in [("center_x", center_x), ("center_y", center_y), 
                                              ("width", width), ("height", height)]:
                    if not (0.0 <= coord_value <= 1.0):
                        logger.error(f"Invalid {coord_name} {coord_value} in {label_file}:{line_num} - Must be 0-1")
                        return False
                        
            except ValueError as e:
                logger.error(f"Invalid number format in {label_file}:{line_num}: {e}")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error reading {label_file}: {e}")
        return False


def get_image_label_pairs(images_dir: Path, labels_dir: Path) -> List[Tuple[Path, Path]]:
    """
    Get paired image and label files.
    
    Args:
        images_dir: Directory containing images
        labels_dir: Directory containing labels
        
    Returns:
        List of (image_path, label_path) tuples
    """
    pairs = []
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    for image_file in images_dir.iterdir():
        if image_file.suffix.lower() in image_extensions:
            # Find corresponding label file
            label_file = labels_dir / f"{image_file.stem}.txt"
            
            if label_file.exists():
                pairs.append((image_file, label_file))
            else:
                logger.warning(f"No label file found for image: {image_file}")
    
    logger.info(f"Found {len(pairs)} image-label pairs")
    return pairs


def copy_file_pairs(pairs: List[Tuple[Path, Path]], dest_images_dir: Path, dest_labels_dir: Path) -> None:
    """
    Copy image-label pairs to destination directories.
    
    Args:
        pairs: List of (image_path, label_path) tuples
        dest_images_dir: Destination directory for images
        dest_labels_dir: Destination directory for labels
    """
    # Create destination directories
    dest_images_dir.mkdir(parents=True, exist_ok=True)
    dest_labels_dir.mkdir(parents=True, exist_ok=True)
    
    for image_path, label_path in pairs:
        # Copy image
        dest_image = dest_images_dir / image_path.name
        shutil.copy2(image_path, dest_image)
        
        # Copy label
        dest_label = dest_labels_dir / label_path.name
        shutil.copy2(label_path, dest_label)
    
    logger.info(f"Copied {len(pairs)} file pairs to {dest_images_dir.parent}")


def validate_class_distribution(labels_dir: Path) -> Dict[int, int]:
    """
    Validate class distribution in the dataset.
    
    Args:
        labels_dir: Directory containing label files
        
    Returns:
        Dictionary mapping class_id to count
    """
    class_counts = {i: 0 for i in range(32)}
    
    for label_file in labels_dir.glob("*.txt"):
        try:
            with open(label_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        class_id = int(line.split()[0])
                        if 0 <= class_id <= 31:
                            class_counts[class_id] += 1
        except Exception as e:
            logger.error(f"Error processing {label_file}: {e}")
    
    # Log class distribution
    logger.info("Class distribution:")
    for class_id, count in class_counts.items():
        fdi_number = FDISystem.class_to_fdi(class_id)
        tooth_type = FDISystem.get_tooth_type_name(fdi_number)
        logger.info(f"  Class {class_id} ({tooth_type} {fdi_number}): {count} instances")
    
    return class_counts


def check_missing_classes(class_counts: Dict[int, int]) -> List[int]:
    """
    Check for missing classes in the dataset.
    
    Args:
        class_counts: Dictionary mapping class_id to count
        
    Returns:
        List of missing class IDs
    """
    missing_classes = [class_id for class_id, count in class_counts.items() if count == 0]
    
    if missing_classes:
        logger.warning(f"Missing classes: {missing_classes}")
        for class_id in missing_classes:
            fdi_number = FDISystem.class_to_fdi(class_id)
            tooth_type = FDISystem.get_tooth_type_name(fdi_number)
            logger.warning(f"  Class {class_id}: {tooth_type} ({fdi_number})")
    else:
        logger.info("✓ All 32 classes are present in the dataset")
    
    return missing_classes


def print_project_structure():
    """Print the project directory structure."""
    print("\nTooth Numbering YOLO Project Structure:")
    print("=" * 50)
    print("tooth_numbering_yolo/")
    print("├── __init__.py")
    print("├── fdi_system.py      # FDI numbering system implementation")
    print("├── config.py          # Configuration settings")
    print("├── utils.py           # Utility functions")
    print("├── data_manager.py    # Dataset management")
    print("├── trainer.py         # YOLO model training")
    print("├── evaluator.py       # Model evaluation")
    print("└── post_processor.py  # Anatomical post-processing")
    print("\nOutputs/")
    print("├── dataset/           # Organized train/val/test splits")
    print("├── models/            # Trained model weights")
    print("├── results/           # Evaluation results")
    print("├── plots/             # Training curves and visualizations")
    print("└── predictions/       # Sample prediction images")


if __name__ == "__main__":
    setup_logging()
    print_project_structure()