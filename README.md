# 🦷 Tooth Numbering YOLO System


A complete **YOLO-based** tooth detection and **FDI numbering system** for dental panoramic images. This project accurately identifies and classifies teeth, generating all necessary outputs for academic research submissions.


***

## ✨ Key Features

* **Complete FDI System**: Implements the full FDI tooth numbering system with guaranteed class order preservation.
* **Multi-YOLO Support**: Fully compatible with **YOLOv5, YOLOv8, and YOLOv11**.
* **Anatomical Post-Processing**: Optional logic to enhance anatomical correctness by analyzing quadrants and tooth positions.
* **Comprehensive Evaluation**: Generates a full suite of metrics, including confusion matrices, precision, recall, mAP@50, and mAP@50-95.
* **Submission Ready**: Automatically creates all required deliverables, including plots, predictions, and metrics reports.

***

## 🚀 Getting Started

Follow these steps to get the project set up and running on your local machine.

### Prerequisites

* Python 3.8+
* Pip package manager
* A Git client

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/tooth-numbering-yolo.git](https://github.com/your-username/tooth-numbering-yolo.git)
    cd tooth-numbering-yolo
    ```
2.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Prepare your dataset:**
    Place your dataset in the root folder following this structure:
    ```
    ToothNumber_TaskDataset/
    ├── images/      # ~500 dental panoramic images (.jpg, .png)
    └── labels/      # Corresponding YOLO-format .txt files
    ```
    The YOLO label format should be `class_id center_x center_y width height` (normalized).

***

## Usage

1.  **Validate the FDI System & Generate Config**
    Run this script first to verify the class mappings and generate the `data.yaml` file required for training.
    ```bash
    python validate_fdi_system.py
    ```

2.  **Train the Model**
    Start the training process using your desired YOLO variant.
    ```bash
    python trainer.py --model yolov8
    ```

3.  **Evaluate the Results**
    Once training is complete, run the evaluator to generate all submission metrics and visuals.
    ```bash
    python evaluator.py --weights outputs/models/best.pt
    ```

***

## 📖 FDI Numbering System

The system is built around a strict implementation of the FDI tooth numbering system. The mapping between class IDs and FDI codes is **immutable**.

* **First digit**: Quadrant (1: Upper Right, 2: Upper Left, 3: Lower Left, 4: Lower Right)
* **Second digit**: Tooth position (1: Central Incisor → 8: Third Molar)

<details>
<summary><b>Click to view the full Class ID ↔ FDI Reference Table</b></summary>

| Class ID | Tooth (FDI)        | Class ID | Tooth (FDI)         |
|:--------:|:-------------------|:--------:|:--------------------|
| 0        | Canine (13)        | 16       | Lateral Incisor (22)  |
| 1        | Canine (23)        | 17       | Lateral Incisor (32)  |
| 2        | Canine (33)        | 18       | Lateral Incisor (42)  |
| 3        | Canine (43)        | 19       | Lateral Incisor (12)  |
| 4        | Central Incisor (21) | 20       | Second Molar (17)   |
| 5        | Central Incisor (41) | 21       | Second Molar (27)   |
| 6        | Central Incisor (31) | 22       | Second Molar (37)   |
| 7        | Central Incisor (11) | 23       | Second Molar (47)   |
| 8        | First Molar (16)   | 24       | Second Premolar (15)|
| 9        | First Molar (26)   | 25       | Second Premolar (25)|
| 10       | First Molar (36)   | 26       | Second Premolar (35)|
| 11       | First Molar (46)   | 27       | Second Premolar (45)|
| 12       | First Premolar (14)  | 28       | Third Molar (18)    |
| 13       | First Premolar (34)  | 29       | Third Molar (28)    |
| 14       | First Premolar (44)  | 30       | Third Molar (38)    |
| 15       | First Premolar (24)  | 31       | Third Molar (48)    |

</details>

***

## 📊 Evaluation & Outputs

The system generates all required materials for a research submission in the `outputs/` directory:

* **Confusion Matrix**: A 32x32 matrix for all tooth classes.
* **Performance Metrics**: Detailed `results.csv` with Precision, Recall, mAP@50, and mAP@50-95.
* **Prediction Images**: Sample images with predicted bounding boxes and FDI labels.
* **Training Curves**: Plots for loss and accuracy over epochs.

***

## ⚙️ Configuration

Key parameters can be adjusted in `config.py`:

```python
# Dataset split ratios
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# YOLO training configuration
YOLO_CONFIG = {
    "input_size": 640,
    "batch_size": 16,
    "epochs": 100,
    "pretrained_weights": "yolov8s.pt"
}