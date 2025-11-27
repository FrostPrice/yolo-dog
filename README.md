# YOLO Dog Detection Project

A complete implementation of YOLOv8 for detecting dogs in images and videos, with two specialized notebooks for different use cases.

## Overview

This project uses YOLOv8 (You Only Look Once version 8) to detect dogs in images and video streams. The implementation is based on Ultralytics' YOLOv8 framework and provides two distinct approaches:

1. **COCO Dataset Detection** (`yolo_dog_detection_coco.ipynb`) - Use pre-trained models for immediate detection
2. **Custom Training** (`yolo_dog_detection_custom.ipynb`) - Train on your own labeled dog dataset

## Features

- Dog detection in static images
- Real-time dog detection in videos
- Batch processing for multiple images
- Pre-trained COCO model (ready to use)
- Custom dataset preparation and training support
- Model evaluation with metrics (mAP, precision, recall)
- Model export for deployment (ONNX, TensorFlow, etc.)
- Visualization with bounding boxes and confidence scores

## Project Structure

```text
yolo-dog/
├── data/
│   ├── images/           # Input images
│   │   ├── coco_dogs/   # COCO dataset images (for pre-trained model)
│   │   ├── train/       # Training images (for custom training)
│   │   ├── val/         # Validation images
│   │   └── test/        # Test images
│   ├── labels/          # YOLO format labels (for custom training)
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── dataset.yaml     # Dataset configuration (for custom training)
├── models/              # Saved model weights
├── outputs/             # Detection results and visualizations
├── yolo_dog_detection_coco.ipynb    # COCO pre-trained detection notebook
├── yolo_dog_detection_custom.ipynb  # Custom dataset training notebook
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Notebooks

### 1. COCO Pre-trained Detection (`yolo_dog_detection_coco.ipynb`)

**Use this notebook when:**

- You want to detect dogs immediately without training
- You need quick results on general dog images
- You don't have a labeled dataset

**Features:**

- Downloads COCO dog images automatically
- Uses pre-trained YOLOv8 model (no training needed)
- Detects dogs in images and videos
- Batch processing capabilities

**Workflow:**

1. Load pre-trained model
2. Download COCO dog samples (optional)
3. Run detection on your images
4. Process videos
5. View results

### 2. Custom Dataset Training (`yolo_dog_detection_custom.ipynb`)

**Use this notebook when:**

- You have your own labeled dog dataset
- You need specialized detection (specific dog breeds, poses, etc.)
- You want to fine-tune the model for your use case

**Features:**

- Prepares custom dataset structure
- Fine-tunes YOLOv8 on your data
- Provides training metrics and visualization
- Model evaluation on test set

**Workflow:**

1. Prepare labeled dataset (YOLO format)
2. Load base model
3. Configure training parameters
4. Train custom model
5. Evaluate and use trained model

**Dataset Requirements:**

- Images in `data/images/train/`, `data/images/val/`, `data/images/test/`
- Labels in `data/labels/train/`, `data/labels/val/`, `data/labels/test/`
- YOLO format: `class_id center_x center_y width height` (normalized 0-1)

## Getting Started

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended but not required)
- Virtual environment (venv)

### Installation

1. **Create .venv environment:**

   ```bash
   python -m venv .venv
   ```

2. **Activate the virtual environment:**

   ```bash
   source .venv/bin/activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

### Quick Start

#### Using Pre-trained COCO Model

1. **Launch the COCO notebook:**

   ```bash
   jupyter notebook yolo_dog_detection_coco.ipynb
   ```

2. **Run the cells sequentially:**
   - The notebook will download COCO dog images automatically
   - Detect dogs using the pre-trained model
   - View results in `outputs/` directory

#### Training Custom Model

1. **Prepare your dataset** in YOLO format (see Dataset Format section below)

2. **Launch the custom training notebook:**

   ```bash
   jupyter notebook yolo_dog_detection_custom.ipynb
   ```

3. **Run the cells to train:**
   - Set up dataset configuration
   - Train the model on your data
   - Evaluate and test the trained model

## Usage Examples

### Using Pre-trained Model (COCO Notebook)

```python
from ultralytics import YOLO

# Load pre-trained model
model = YOLO('yolov8n.pt')

# Run inference on dog images (class 16 = dog in COCO)
results = model('path/to/image.jpg', classes=[16])

# Display results
results[0].show()
```

### Training Custom Model (Custom Notebook)

```python
from ultralytics import YOLO

# Load base model
model = YOLO('yolov8n.pt')

# Train on custom dataset
results = model.train(
    data='data/dataset.yaml',
    epochs=50,
    imgsz=640,
    batch=16
)

# Use trained model
model = YOLO('models/yolo_dog_custom/weights/best.pt')
results = model('path/to/image.jpg')
```

### Process Videos

```python
# Process video file with pre-trained model
results = model('path/to/video.mp4', classes=[16])

# Save annotated video
for result in results:
    result.save()
```

## Configuration

### COCO Notebook Configuration

```python
config = {
    'model_name': 'yolov8n.pt',      # Pre-trained model
    'img_size': 640,                 # Input image size
    'confidence_threshold': 0.25,    # Detection confidence threshold
    'iou_threshold': 0.45,           # IOU threshold for NMS
    'dog_class_id': 16,              # COCO dog class ID
}
```

### Custom Training Notebook Configuration

```python
config = {
    'model_name': 'yolov8n.pt',      # Base model to fine-tune
    'img_size': 640,                 # Input image size
    'batch_size': 16,                # Batch size for training
    'epochs': 50,                    # Training epochs
    'confidence_threshold': 0.25,    # Detection confidence threshold
    'iou_threshold': 0.45,           # IOU threshold for NMS
    'dog_class_id': 0,               # Custom dataset class ID
}
```

## Troubleshooting

### No GPU detected

- The model will run on CPU (slower but functional)
- Install CUDA-compatible PyTorch: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`

### Out of memory error

- Reduce batch size in config
- Use a smaller model (yolov8n instead of yolov8l)
- Reduce image size

### Poor detection results (COCO notebook)

- Adjust confidence threshold
- Try a larger model (yolov8m or yolov8l)
- For specialized cases, use custom training notebook

### Poor training results (Custom notebook)

- Collect more diverse training data
- Increase training epochs
- Use data augmentation (built-in to YOLO)
- Check label quality and accuracy
- Try different model sizes

## Which Notebook Should I Use?

| Scenario                               | Recommended Notebook              |
| -------------------------------------- | --------------------------------- |
| Quick dog detection on general images  | `yolo_dog_detection_coco.ipynb`   |
| No labeled dataset available           | `yolo_dog_detection_coco.ipynb`   |
| Need to detect specific dog breeds     | `yolo_dog_detection_custom.ipynb` |
| Have custom labeled dog dataset        | `yolo_dog_detection_custom.ipynb` |
| Production deployment with fine-tuning | `yolo_dog_detection_custom.ipynb` |
| Rapid prototyping and testing          | `yolo_dog_detection_coco.ipynb`   |

## Resources

- [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)
- [YOLO GitHub Repository](https://github.com/ultralytics/ultralytics)
- [COCO Dataset](https://cocodataset.org/)
- [Roboflow for Dataset Annotation](https://roboflow.com/)

## 📝 Dataset Format

For custom training (Custom notebook), labels should follow YOLO format:

```text
class_id x_center y_center width height
```

Where:

- `class_id`: 0 for dog (if single class)
- `x_center`, `y_center`: center coordinates (normalized 0-1)
- `width`, `height`: box dimensions (normalized 0-1)

Example label file (`dog.txt`):

```text
0 0.5 0.5 0.3 0.4
```

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

This project uses the Ultralytics YOLO framework, which is licensed under AGPL-3.0.

## Authors

- **Author**: Jonathas de Oliveira Meine, Mateus Barbosa, Mateus José da Silva, Matheus de Oliveira Rocha, Rodrigo Faistauer
- **Course**: Inteligência Artificial II
- **Institution**: UNIVALI
- **Year**: 2025

## Acknowledgments

- [Ultralytics](https://ultralytics.com/) for the YOLO implementation
- COCO dataset for pre-trained weights
- OpenCV and PyTorch communities

---

## Happy Dog Detecting!
