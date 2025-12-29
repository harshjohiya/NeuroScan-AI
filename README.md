# NeuroScan-AI 🧠

**Brain Tumor Detection and Classification using Deep Learning**

A deep learning project for automated detection and classification of brain tumors from MRI images using Convolutional Neural Networks (CNN).

## 📋 Overview

This project implements a CNN-based system to classify brain MRI images into four categories:
- **Glioma Tumor**
- **Meningioma Tumor**
- **Pituitary Tumor**
- **No Tumor**

The system achieves high accuracy through a carefully designed CNN architecture with data augmentation and proper regularization techniques.

## 🚀 Features

- **Modular Architecture**: Clean separation of concerns with dedicated modules for data loading, preprocessing, model definition, training, evaluation, and prediction
- **Configurable**: YAML-based configuration for easy experimentation
- **Comprehensive Evaluation**: Detailed metrics, confusion matrices, and classification reports
- **Easy Prediction**: Simple interface for making predictions on new images
- **Experiment Tracking**: Organized experiment documentation

## 📁 Project Structure

```
NeuroScan-AI/
│
├── data/                           # Data directory
│   ├── raw/                        # Raw data storage
│   └── processed/                  # Processed data storage
│
├── src/                            # Source code
│   ├── data_loader.py             # Data loading utilities
│   ├── preprocessing.py            # Data preprocessing and augmentation
│   ├── model.py                    # CNN model architecture
│   ├── train.py                    # Training script
│   ├── evaluate.py                 # Evaluation script
│   └── predict.py                  # Prediction script
│
├── experiments/                    # Experiment logs and notes
│   ├── exp_01_baseline.md
│   └── exp_02_transfer_learning.md
│
├── notebooks/                      # Jupyter notebooks
│   └── exploration.ipynb
│
├── models/                         # Saved models
│   └── best_model.h5
│
├── configs/                        # Configuration files
│   └── config.yaml
│
├── README.md                       # This file
├── requirements.txt                # Python dependencies
└── roadmap.md                      # Project roadmap
```

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   cd NeuroScan-AI
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Dataset

The project uses brain MRI images organized in the following structure:

```
Dataset/
├── Training/
│   ├── glioma_tumor/
│   ├── meningioma_tumor/
│   ├── no_tumor/
│   └── pituitary_tumor/
└── Testing/
    ├── glioma_tumor/
    ├── meningioma_tumor/
    ├── no_tumor/
    └── pituitary_tumor/
```

Place your dataset in the parent directory or update paths in `configs/config.yaml`.

## 🎯 Usage

### Training

Train the model with default configuration:

```bash
cd src
python train.py
```

### Evaluation

Evaluate the trained model on test data:

```bash
cd src
python evaluate.py
```

This will generate:
- Classification report
- Confusion matrices
- Performance metrics

### Prediction

Make predictions on new images:

```bash
cd src
python predict.py --image path/to/image.jpg
```

With custom model:
```bash
python predict.py --model ../models/custom_model.h5 --image path/to/image.jpg
```

### Configuration

Modify `configs/config.yaml` to customize:
- Model architecture parameters
- Training hyperparameters
- Data augmentation settings
- File paths

## 🧪 Model Architecture

The CNN architecture consists of:
- **4 Convolutional Blocks**: Each with Conv2D, MaxPooling, and Dropout layers
- **Dense Layers**: Fully connected layers with dropout for regularization
- **Output Layer**: Softmax activation for 4-class classification

**Key Features:**
- Input size: 150x150x3 (RGB images)
- Optimizer: Adam (learning rate: 0.001)
- Loss function: Categorical crossentropy
- Regularization: Dropout layers to prevent overfitting

## 📈 Results

Results are saved in the `Results/` directory including:
- Training history plots
- Confusion matrices
- Classification reports
- Model checkpoints

## 🔬 Experiments

Document your experiments in the `experiments/` directory:
- `exp_01_baseline.md`: Baseline model results
- `exp_02_transfer_learning.md`: Transfer learning experiments

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is open source and available under the MIT License.

## 👥 Authors

Your Name - Brain Tumor Detection Project

## 🙏 Acknowledgments

- Dataset providers
- TensorFlow/Keras community
- Research papers in medical image analysis

## 📞 Contact

For questions or feedback, please open an issue on the project repository.

---

**Note**: This is a research/educational project. For medical diagnosis, always consult healthcare professionals.
