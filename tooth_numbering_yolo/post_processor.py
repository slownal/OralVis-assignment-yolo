"""
Post-Processor for Tooth Numbering YOLO System
Implements anatomical correctness logic to improve tooth detection accuracy.
"""

import logging
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict

from .fdi_system import FDISystem
from .config import POST_PROCESSING_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class ToothPrediction:
 """
 Represents a single tooth prediction with all relevant information.
 """
 bbox: Tuple[float, float, float, float] # x_center, y_center, width, height (normalized)
 confidence: float
 class_id: int
 fdi_number: int
 quadrant: int
 position: int
 tooth_type: str
 
 @classmethod
 def from_yolo_detection(cls, detection: Dict[str, Any]) -> 'ToothPrediction':
 """Create ToothPrediction from YOLO detection result."""
 class_id = int(detection['class_id'])
 fdi_number = FDISystem.class_to_fdi(class_id)
 
 return cls(
 bbox=detection['bbox'],
 confidence=float(detection['confidence']),
 class_id=class_id,
 fdi_number=fdi_number,
 quadrant=FDISystem.get_quadrant(fdi_number),
 position=FDISystem.get_position(fdi_number),
 tooth_type=FDISystem.get_tooth_type_name(fdi_number)
 )


class AnatomicalPostProcessor:
 """
 Post-processor that applies anatomical logic to improve tooth detection accuracy.
 
 Implements exact specifications from aim.txt section 8:
 - Separate upper vs lower arch using Y-axis clustering
 - Divide left vs right quadrants using X-midline detection
 - Sort teeth horizontally within quadrants and assign FDI sequentially
 - Handle missing teeth by skipping numbers where spacing is wide
 """
 
 def __init__(self, config: Optional[Dict[str, Any]] = None):
 """
 Initialize anatomical post-processor.
 
 Args:
 config: Configuration dictionary for post-processing parameters
 """
 self.config = config or POST_PROCESSING_CONFIG
 self.y_threshold_ratio = self.config.get('y_threshold_ratio', 0.5)
 self.confidence_threshold = self.config.get('confidence_threshold', 0.5)
 self.nms_threshold = self.config.get('nms_threshold', 0.4)
 self.missing_tooth_spacing_threshold = self.config.get('missing_tooth_spacing_threshold', 1.5)
 
 logger.info("Initialized anatomical post-processor")
 
 def process_predictions(self, predictions: List[Dict[str, Any]], 
 image_width: int = 640, 
 image_height: int = 640) -> List[ToothPrediction]:
 """
 Apply complete anatomical post-processing pipeline.
 
 Args:
 predictions: List of raw YOLO predictions
 image_width: Image width for coordinate conversion
 image_height: Image height for coordinate conversion
 
 Returns:
 List of anatomically corrected tooth predictions
 """
 logger.info(f"Processing {len(predictions)} raw predictions")
 
 # Step 1: Convert to ToothPrediction objects
 tooth_predictions = [ToothPrediction.from_yolo_detection(pred) for pred in predictions]
 
 # Step 2: Filter by confidence
 tooth_predictions = self._filter_by_confidence(tooth_predictions)
 
 # Step 3: Apply Non-Maximum Suppression
 tooth_predictions = self._apply_nms(tooth_predictions)
 
 # Step 4: Separate upper and lower arches
 upper_teeth, lower_teeth = self._separate_arches(tooth_predictions)
 
 # Step 5: Divide into quadrants and apply anatomical logic
 corrected_predictions = []
 
 # Process upper arch (quadrants 1 and 2)
 if upper_teeth:
 q1_teeth, q2_teeth = self._divide_quadrants(upper_teeth, is_upper=True)
 corrected_predictions.extend(self._apply_anatomical_sequencing(q1_teeth, target_quadrant=1))
 corrected_predictions.extend(self._apply_anatomical_sequencing(q2_teeth, target_quadrant=2))
 
 # Process lower arch (quadrants 3 and 4)
 if lower_teeth:
 q3_teeth, q4_teeth = self._divide_quadrants(lower_teeth, is_upper=False)
 corrected_predictions.extend(self._apply_anatomical_sequencing(q3_teeth, target_quadrant=3))
 corrected_predictions.extend(self._apply_anatomical_sequencing(q4_teeth, target_quadrant=4))
 
 # Step 6: Validate anatomical consistency
 corrected_predictions = self._validate_anatomical_consistency(corrected_predictions)
 
 logger.info(f"Post-processing complete: {len(corrected_predictions)} anatomically corrected predictions")
 return corrected_predictions
 
 def _filter_by_confidence(self, predictions: List[ToothPrediction]) -> List[ToothPrediction]:
 """Filter predictions by confidence threshold."""
 filtered = [pred for pred in predictions if pred.confidence >= self.confidence_threshold]
 logger.info(f"Confidence filtering: {len(predictions)} → {len(filtered)} predictions")
 return filtered
 
 def _apply_nms(self, predictions: List[ToothPrediction]) -> List[ToothPrediction]:
 """Apply Non-Maximum Suppression to remove duplicate detections."""
 if len(predictions) <= 1:
 return predictions
 
 # Group by class for NMS
 class_groups = defaultdict(list)
 for pred in predictions:
 class_groups[pred.class_id].append(pred)
 
 nms_predictions = []
 for class_id, class_preds in class_groups.items():
 if len(class_preds) == 1:
 nms_predictions.extend(class_preds)
 else:
 # Apply NMS within each class
 nms_predictions.extend(self._nms_single_class(class_preds))
 
 logger.info(f"NMS filtering: {len(predictions)} → {len(nms_predictions)} predictions")
 return nms_predictions
 
 def _nms_single_class(self, predictions: List[ToothPrediction]) -> List[ToothPrediction]:
 """Apply NMS for a single class."""
 if len(predictions) <= 1:
 return predictions
 
 # Sort by confidence (descending)
 predictions = sorted(predictions, key=lambda x: x.confidence, reverse=True)
 
 keep = []
 while predictions:
 # Keep the highest confidence prediction
 current = predictions.pop(0)
 keep.append(current)
 
 # Remove overlapping predictions
 predictions = [pred for pred in predictions 
 if self._calculate_iou(current.bbox, pred.bbox) < self.nms_threshold]
 
 return keep
 
 def _calculate_iou(self, bbox1: Tuple[float, float, float, float], 
 bbox2: Tuple[float, float, float, float]) -> float:
 """Calculate Intersection over Union (IoU) between two bounding boxes."""
 x1_c, y1_c, w1, h1 = bbox1
 x2_c, y2_c, w2, h2 = bbox2
 
 # Convert center coordinates to corner coordinates
 x1_min, y1_min = x1_c - w1/2, y1_c - h1/2
 x1_max, y1_max = x1_c + w1/2, y1_c + h1/2
 x2_min, y2_min = x2_c - w2/2, y2_c - h2/2
 x2_max, y2_max = x2_c + w2/2, y2_c + h2/2
 
 # Calculate intersection
 inter_x_min = max(x1_min, x2_min)
 inter_y_min = max(y1_min, y2_min)
 inter_x_max = min(x1_max, x2_max)
 inter_y_max = min(y1_max, y2_max)
 
 if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
 return 0.0
 
 inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
 
 # Calculate union
 area1 = w1 * h1
 area2 = w2 * h2
 union_area = area1 + area2 - inter_area
 
 return inter_area / union_area if union_area > 0 else 0.0
 
 def _separate_arches(self, predictions: List[ToothPrediction]) -> Tuple[List[ToothPrediction], List[ToothPrediction]]:
 """
 Separate upper and lower arches using Y-axis clustering.
 
 Returns:
 Tuple of (upper_teeth, lower_teeth)
 """
 if not predictions:
 return [], []
 
 # Extract Y coordinates (center of bounding boxes)
 y_coords = [pred.bbox[1] for pred in predictions]
 
 # Use simple threshold-based separation
 # In dental panoramic images, upper teeth typically appear in upper half
 y_threshold = self.y_threshold_ratio
 
 upper_teeth = [pred for pred in predictions if pred.bbox[1] < y_threshold]
 lower_teeth = [pred for pred in predictions if pred.bbox[1] >= y_threshold]
 
 logger.info(f"Arch separation: {len(upper_teeth)} upper, {len(lower_teeth)} lower teeth")
 return upper_teeth, lower_teeth
 
 def _divide_quadrants(self, arch_predictions: List[ToothPrediction], 
 is_upper: bool) -> Tuple[List[ToothPrediction], List[ToothPrediction]]:
 """
 Divide arch into left and right quadrants using X-midline detection.
 
 Args:
 arch_predictions: Predictions for one arch (upper or lower)
 is_upper: True for upper arch, False for lower arch
 
 Returns:
 Tuple of (right_quadrant, left_quadrant) predictions
 """
 if not arch_predictions:
 return [], []
 
 # Calculate X-midline (center of image is typically 0.5 in normalized coordinates)
 x_midline = 0.5
 
 # Divide based on X coordinate
 right_quadrant = [pred for pred in arch_predictions if pred.bbox[0] < x_midline]
 left_quadrant = [pred for pred in arch_predictions if pred.bbox[0] >= x_midline]
 
 quadrant_names = ("upper-right & upper-left" if is_upper else "lower-left & lower-right")
 logger.info(f"Quadrant division ({quadrant_names}): {len(right_quadrant)} right, {len(left_quadrant)} left")
 
 return right_quadrant, left_quadrant
 
 def _apply_anatomical_sequencing(self, quadrant_predictions: List[ToothPrediction], 
 target_quadrant: int) -> List[ToothPrediction]:
 """
 Apply anatomical sequencing within a quadrant.
 
 Args:
 quadrant_predictions: Predictions for one quadrant
 target_quadrant: Target FDI quadrant (1, 2, 3, or 4)
 
 Returns:
 List of predictions with corrected FDI numbers
 """
 if not quadrant_predictions:
 return []
 
 # Sort teeth horizontally within quadrant
 # For quadrants 1 and 4 (right side): sort left to right (ascending X)
 # For quadrants 2 and 3 (left side): sort right to left (descending X)
 if target_quadrant in [1, 4]:
 sorted_teeth = sorted(quadrant_predictions, key=lambda x: x.bbox[0])
 else:
 sorted_teeth = sorted(quadrant_predictions, key=lambda x: x.bbox[0], reverse=True)
 
 # Apply sequential FDI numbering with gap detection
 corrected_predictions = []
 expected_positions = list(range(1, 9)) # Positions 1-8 in each quadrant
 
 for i, tooth in enumerate(sorted_teeth):
 # Detect gaps based on spacing
 if i > 0:
 prev_tooth = sorted_teeth[i-1]
 spacing = abs(tooth.bbox[0] - prev_tooth.bbox[0])
 avg_tooth_width = (tooth.bbox[2] + prev_tooth.bbox[2]) / 2
 
 # If spacing is significantly larger than average tooth width, there might be missing teeth
 if spacing > self.missing_tooth_spacing_threshold * avg_tooth_width:
 # Skip position(s) for missing teeth
 positions_to_skip = int(spacing / avg_tooth_width) - 1
 for _ in range(positions_to_skip):
 if expected_positions:
 expected_positions.pop(0)
 
 # Assign next available position
 if expected_positions:
 new_position = expected_positions.pop(0)
 new_fdi_number = target_quadrant * 10 + new_position
 
 # Create corrected prediction
 corrected_tooth = ToothPrediction(
 bbox=tooth.bbox,
 confidence=tooth.confidence,
 class_id=FDISystem.fdi_to_class(new_fdi_number),
 fdi_number=new_fdi_number,
 quadrant=target_quadrant,
 position=new_position,
 tooth_type=FDISystem.get_tooth_type_name(new_fdi_number)
 )
 corrected_predictions.append(corrected_tooth)
 
 logger.info(f"Anatomical sequencing (Q{target_quadrant}): {len(quadrant_predictions)} → {len(corrected_predictions)} teeth")
 return corrected_predictions
 
 def _validate_anatomical_consistency(self, predictions: List[ToothPrediction]) -> List[ToothPrediction]:
 """
 Validate and ensure anatomical consistency of final predictions.
 
 Args:
 predictions: List of anatomically processed predictions
 
 Returns:
 List of validated predictions
 """
 # Check for duplicate FDI numbers
 fdi_counts = defaultdict(int)
 for pred in predictions:
 fdi_counts[pred.fdi_number] += 1
 
 # Remove duplicates (keep highest confidence)
 unique_predictions = {}
 for pred in predictions:
 fdi = pred.fdi_number
 if fdi not in unique_predictions or pred.confidence > unique_predictions[fdi].confidence:
 unique_predictions[fdi] = pred
 
 validated_predictions = list(unique_predictions.values())
 
 # Log any anatomical inconsistencies
 duplicates_removed = len(predictions) - len(validated_predictions)
 if duplicates_removed > 0:
 logger.warning(f"Removed {duplicates_removed} duplicate FDI predictions")
 
 # Validate quadrant consistency
 for pred in validated_predictions:
 expected_quadrant = FDISystem.get_quadrant(pred.fdi_number)
 if pred.quadrant != expected_quadrant:
 logger.warning(f"Quadrant inconsistency: FDI {pred.fdi_number} in quadrant {pred.quadrant}, expected {expected_quadrant}")
 
 logger.info(f"Anatomical validation: {len(predictions)} → {len(validated_predictions)} consistent predictions")
 return validated_predictions
 
 def get_processing_statistics(self, original_predictions: List[Dict[str, Any]], 
 final_predictions: List[ToothPrediction]) -> Dict[str, Any]:
 """
 Get statistics about the post-processing pipeline.
 
 Args:
 original_predictions: Original YOLO predictions
 final_predictions: Final anatomically corrected predictions
 
 Returns:
 Dictionary containing processing statistics
 """
 stats = {
 "original_count": len(original_predictions),
 "final_count": len(final_predictions),
 "confidence_filtered": 0,
 "nms_filtered": 0,
 "anatomically_corrected": 0,
 "quadrant_distribution": defaultdict(int),
 "tooth_type_distribution": defaultdict(int),
 "fdi_numbers": []
 }
 
 # Calculate quadrant and tooth type distributions
 for pred in final_predictions:
 stats["quadrant_distribution"][pred.quadrant] += 1
 stats["tooth_type_distribution"][pred.tooth_type] += 1
 stats["fdi_numbers"].append(pred.fdi_number)
 
 # Sort FDI numbers for easier analysis
 stats["fdi_numbers"] = sorted(stats["fdi_numbers"])
 
 return stats


def apply_anatomical_post_processing(predictions: List[Dict[str, Any]], 
 config: Optional[Dict[str, Any]] = None) -> List[ToothPrediction]:
 """
 Main function to apply anatomical post-processing to YOLO predictions.
 
 Args:
 predictions: List of raw YOLO predictions
 config: Optional configuration for post-processing
 
 Returns:
 List of anatomically corrected tooth predictions
 """
 logger.info(" Starting anatomical post-processing")
 
 # Initialize post-processor
 processor = AnatomicalPostProcessor(config)
 
 # Apply post-processing
 corrected_predictions = processor.process_predictions(predictions)
 
 # Get statistics
 stats = processor.get_processing_statistics(predictions, corrected_predictions)
 
 logger.info("Post-processing statistics:")
 logger.info(f" Original predictions: {stats['original_count']}")
 logger.info(f" Final predictions: {stats['final_count']}")
 logger.info(f" Quadrant distribution: {dict(stats['quadrant_distribution'])}")
 logger.info(f" FDI numbers detected: {stats['fdi_numbers']}")
 
 logger.info(" Anatomical post-processing completed successfully!")
 return corrected_predictions


if __name__ == "__main__":
 import sys
 from .utils import setup_logging
 
 setup_logging()
 
 # Demo with simulated predictions
 demo_predictions = [
 {"bbox": (0.3, 0.3, 0.05, 0.08), "confidence": 0.85, "class_id": 7}, # Central Incisor (11)
 {"bbox": (0.4, 0.3, 0.05, 0.08), "confidence": 0.82, "class_id": 19}, # Lateral Incisor (12)
 {"bbox": (0.6, 0.3, 0.05, 0.08), "confidence": 0.78, "class_id": 16}, # Lateral Incisor (22)
 {"bbox": (0.7, 0.3, 0.05, 0.08), "confidence": 0.81, "class_id": 4}, # Central Incisor (21)
 ]
 
 corrected = apply_anatomical_post_processing(demo_predictions)
 
 print("\nDemo Results:")
 for pred in corrected:
 print(f"FDI {pred.fdi_number}: {pred.tooth_type} (Q{pred.quadrant}, Pos{pred.position}) - Conf: {pred.confidence:.3f}")
 
 sys.exit(0)