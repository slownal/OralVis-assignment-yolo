"""
FDI (Fédération Dentaire Internationale) Numbering System Implementation

This module implements the complete FDI tooth numbering system with exact class mapping
as specified in the OralVis AI Research Intern Task. The class order MUST NOT be changed.
"""

from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class FDISystem:
    """
    Complete FDI numbering system implementation with immutable class order.
    
    The FDI system uses two digits:
    - First digit = Quadrant (1=upper right, 2=upper left, 3=lower left, 4=lower right)
    - Second digit = Tooth position (1=central incisor → 8=third molar)
    """
    
    # CRITICAL: This exact mapping MUST NOT be changed - preserves class order from reference
    CLASS_TO_FDI_MAPPING = {
        0: 13,   # Canine (13)
        1: 23,   # Canine (23)
        2: 33,   # Canine (33)
        3: 43,   # Canine (43)
        4: 21,   # Central Incisor (21)
        5: 41,   # Central Incisor (41)
        6: 31,   # Central Incisor (31)
        7: 11,   # Central Incisor (11)
        8: 16,   # First Molar (16)
        9: 26,   # First Molar (26)
        10: 36,  # First Molar (36)
        11: 46,  # First Molar (46)
        12: 14,  # First Premolar (14)
        13: 34,  # First Premolar (34)
        14: 44,  # First Premolar (44)
        15: 24,  # First Premolar (24)
        16: 22,  # Lateral Incisor (22)
        17: 32,  # Lateral Incisor (32)
        18: 42,  # Lateral Incisor (42)
        19: 12,  # Lateral Incisor (12)
        20: 17,  # Second Molar (17)
        21: 27,  # Second Molar (27)
        22: 37,  # Second Molar (37)
        23: 47,  # Second Molar (47)
        24: 15,  # Second Premolar (15)
        25: 25,  # Second Premolar (25)
        26: 35,  # Second Premolar (35)
        27: 45,  # Second Premolar (45)
        28: 18,  # Third Molar (18)
        29: 28,  # Third Molar (28)
        30: 38,  # Third Molar (38)
        31: 48,  # Third Molar (48)
    }
    
    # Reverse mapping for FDI to class conversion
    FDI_TO_CLASS_MAPPING = {v: k for k, v in CLASS_TO_FDI_MAPPING.items()}
    
    # Tooth type mapping for better understanding
    TOOTH_TYPES = {
        1: "Central Incisor",
        2: "Lateral Incisor", 
        3: "Canine",
        4: "First Premolar",
        5: "Second Premolar",
        6: "First Molar",
        7: "Second Molar",
        8: "Third Molar"
    }
    
    # Quadrant definitions
    QUADRANTS = {
        1: "Upper Right",
        2: "Upper Left", 
        3: "Lower Left",
        4: "Lower Right"
    }
    
    @classmethod
    def get_class_names(cls) -> List[str]:
        """
        Get the complete list of class names in exact order.
        
        Returns:
            List of 32 class names in the exact order required for YOLO configuration
        """
        class_names = []
        for class_id in range(32):
            fdi_number = cls.CLASS_TO_FDI_MAPPING[class_id]
            tooth_type = cls.get_tooth_type_name(fdi_number)
            class_names.append(f"{tooth_type} ({fdi_number})")
        return class_names
    
    @classmethod
    def class_to_fdi(cls, class_id: int) -> int:
        """
        Convert YOLO class ID to FDI tooth number.
        
        Args:
            class_id: YOLO class ID (0-31)
            
        Returns:
            FDI tooth number (11-48)
            
        Raises:
            ValueError: If class_id is not in valid range
        """
        if class_id not in cls.CLASS_TO_FDI_MAPPING:
            raise ValueError(f"Invalid class_id: {class_id}. Must be 0-31.")
        return cls.CLASS_TO_FDI_MAPPING[class_id]
    
    @classmethod
    def fdi_to_class(cls, fdi_number: int) -> int:
        """
        Convert FDI tooth number to YOLO class ID.
        
        Args:
            fdi_number: FDI tooth number (11-48)
            
        Returns:
            YOLO class ID (0-31)
            
        Raises:
            ValueError: If fdi_number is not valid
        """
        if fdi_number not in cls.FDI_TO_CLASS_MAPPING:
            raise ValueError(f"Invalid FDI number: {fdi_number}")
        return cls.FDI_TO_CLASS_MAPPING[fdi_number]
    
    @classmethod
    def get_quadrant(cls, fdi_number: int) -> int:
        """
        Extract quadrant from FDI tooth number.
        
        Args:
            fdi_number: FDI tooth number (11-48)
            
        Returns:
            Quadrant number (1-4)
        """
        return fdi_number // 10
    
    @classmethod
    def get_position(cls, fdi_number: int) -> int:
        """
        Extract tooth position within quadrant from FDI number.
        
        Args:
            fdi_number: FDI tooth number (11-48)
            
        Returns:
            Position within quadrant (1-8)
        """
        return fdi_number % 10
    
    @classmethod
    def get_tooth_type_name(cls, fdi_number: int) -> str:
        """
        Get the tooth type name from FDI number.
        
        Args:
            fdi_number: FDI tooth number (11-48)
            
        Returns:
            Tooth type name (e.g., "Central Incisor", "Canine")
        """
        position = cls.get_position(fdi_number)
        return cls.TOOTH_TYPES.get(position, "Unknown")
    
    @classmethod
    def get_quadrant_name(cls, fdi_number: int) -> str:
        """
        Get the quadrant name from FDI number.
        
        Args:
            fdi_number: FDI tooth number (11-48)
            
        Returns:
            Quadrant name (e.g., "Upper Right", "Lower Left")
        """
        quadrant = cls.get_quadrant(fdi_number)
        return cls.QUADRANTS.get(quadrant, "Unknown")
    
    @classmethod
    def validate_fdi_number(cls, fdi_number: int) -> bool:
        """
        Validate if an FDI number is valid according to the system.
        
        Args:
            fdi_number: FDI tooth number to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(fdi_number, int):
            return False
            
        quadrant = fdi_number // 10
        position = fdi_number % 10
        
        # Valid quadrants: 1-4, Valid positions: 1-8
        return quadrant in [1, 2, 3, 4] and position in [1, 2, 3, 4, 5, 6, 7, 8]
    
    @classmethod
    def validate_class_order(cls) -> bool:
        """
        Validate that the class order matches the reference specification.
        
        Returns:
            True if class order is preserved, False otherwise
        """
        expected_mapping = {
            0: 13, 1: 23, 2: 33, 3: 43, 4: 21, 5: 41, 6: 31, 7: 11,
            8: 16, 9: 26, 10: 36, 11: 46, 12: 14, 13: 34, 14: 44, 15: 24,
            16: 22, 17: 32, 18: 42, 19: 12, 20: 17, 21: 27, 22: 37, 23: 47,
            24: 15, 25: 25, 26: 35, 27: 45, 28: 18, 29: 28, 30: 38, 31: 48
        }
        
        return cls.CLASS_TO_FDI_MAPPING == expected_mapping
    
    @classmethod
    def get_all_fdi_numbers(cls) -> List[int]:
        """
        Get all valid FDI numbers in class order.
        
        Returns:
            List of all 32 FDI numbers in class order
        """
        return [cls.CLASS_TO_FDI_MAPPING[i] for i in range(32)]
    
    @classmethod
    def print_reference_table(cls) -> None:
        """Print the complete reference table for verification."""
        print("Tooth ID ↔ FDI Reference Table")
        print("=" * 50)
        print("Class ID    Tooth (FDI)              Class ID    Tooth (FDI)")
        
        for i in range(16):
            left_class = i
            right_class = i + 16
            left_fdi = cls.CLASS_TO_FDI_MAPPING[left_class]
            right_fdi = cls.CLASS_TO_FDI_MAPPING[right_class]
            left_type = cls.get_tooth_type_name(left_fdi)
            right_type = cls.get_tooth_type_name(right_fdi)
            
            print(f"{left_class:<8}   {left_type} ({left_fdi}){'':<15} {right_class:<8}   {right_type} ({right_fdi})")


def validate_fdi_system() -> bool:
    """
    Comprehensive validation of the FDI system implementation.
    
    Returns:
        True if all validations pass, False otherwise
    """
    logger.info("Validating FDI system implementation...")
    
    # Test 1: Class order preservation
    if not FDISystem.validate_class_order():
        logger.error("Class order validation failed!")
        return False
    
    # Test 2: All 32 classes present
    if len(FDISystem.CLASS_TO_FDI_MAPPING) != 32:
        logger.error(f"Expected 32 classes, found {len(FDISystem.CLASS_TO_FDI_MAPPING)}")
        return False
    
    # Test 3: Bidirectional mapping consistency
    for class_id, fdi_number in FDISystem.CLASS_TO_FDI_MAPPING.items():
        if FDISystem.fdi_to_class(fdi_number) != class_id:
            logger.error(f"Bidirectional mapping failed for class {class_id} -> FDI {fdi_number}")
            return False
    
    # Test 4: FDI number validation
    for fdi_number in FDISystem.get_all_fdi_numbers():
        if not FDISystem.validate_fdi_number(fdi_number):
            logger.error(f"Invalid FDI number: {fdi_number}")
            return False
    
    # Test 5: Class names generation
    class_names = FDISystem.get_class_names()
    if len(class_names) != 32:
        logger.error(f"Expected 32 class names, got {len(class_names)}")
        return False
    
    logger.info("✓ FDI system validation passed!")
    return True


if __name__ == "__main__":
    # Run validation and print reference table
    logging.basicConfig(level=logging.INFO)
    
    print("FDI System Implementation Validation")
    print("=" * 50)
    
    if validate_fdi_system():
        print("\n✓ All validations passed!")
        print("\nReference Table:")
        FDISystem.print_reference_table()
    else:
        print("\n✗ Validation failed!")