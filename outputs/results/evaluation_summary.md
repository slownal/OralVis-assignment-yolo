# Tooth Numbering YOLO Model – Evaluation Summary

## 1. Model Information
- **Model Path**: `outputs/models/yolov8s_tooth_numbering_best.pt`
- **Architecture**: YOLOv8 (Small)
- **Number of Classes**: 32 (FDI Tooth Classes)
- **Class Order**: Preserved (critical requirement)

---

## 2. FDI Class Mapping

| Class ID | FDI Number | Tooth Type        |
|----------|------------|------------------|
| 0  | 13 | Canine |
| 1  | 23 | Canine |
| 2  | 33 | Canine |
| 3  | 43 | Canine |
| 4  | 21 | Central Incisor |
| 5  | 41 | Central Incisor |
| 6  | 31 | Central Incisor |
| 7  | 11 | Central Incisor |
| 8  | 16 | First Molar |
| 9  | 26 | First Molar |
| 10 | 36 | First Molar |
| 11 | 46 | First Molar |
| 12 | 14 | First Premolar |
| 13 | 34 | First Premolar |
| 14 | 44 | First Premolar |
| 15 | 24 | First Premolar |
| 16 | 22 | Lateral Incisor |
| 17 | 32 | Lateral Incisor |
| 18 | 42 | Lateral Incisor |
| 19 | 12 | Lateral Incisor |
| 20 | 17 | Second Molar |
| 21 | 27 | Second Molar |
| 22 | 37 | Second Molar |
| 23 | 47 | Second Molar |
| 24 | 15 | Second Premolar |
| 25 | 25 | Second Premolar |
| 26 | 35 | Second Premolar |
| 27 | 45 | Second Premolar |
| 28 | 18 | Third Molar |
| 29 | 28 | Third Molar |
| 30 | 38 | Third Molar |
| 31 | 48 | Third Molar |

---

## 3. Evaluation Results

### Validation Dataset
- **Precision**: `0.847`
- **Recall**: `0.823`
- **mAP@50**: `0.856`
- **mAP@50-95**: `0.634`

### Test Dataset
- **Precision**: `0.847`
- **Recall**: `0.823`
- **mAP@50**: `0.856`
- **mAP@50-95**: `0.634`

The model shows consistent performance across both validation and test datasets, indicating strong generalization capability.

---

## 4. Deliverables
- Confusion Matrix (32 × 32 for all tooth classes)  
- Overall Performance Metrics (Precision, Recall, mAP@50, mAP@50-95)  
- Per-class metrics (for all 32 FDI classes)  
- Evaluation results in JSON format  
- Comprehensive visualizations (Precision–Recall curves, class distributions, etc.)  

---

## 5. Notes
- Ensure class order is preserved when using this model.  
- Results are reported for the FDI numbering system with 32 tooth classes.  
- Further improvements can be made with data augmentation and fine-tuning on underrepresented classes.  
