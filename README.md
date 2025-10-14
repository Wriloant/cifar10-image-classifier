# CIFAR-10 Image Classification with PyTorch

A clean and efficient implementation of a convolutional neural network from scratch to classify images from the CIFAR-10 dataset. This project demonstrates how to achieve **80%+ accuracy** without using pre-trained models or complex architectures like ResNet.

## 📊 Project Overview

CIFAR-10 is a classic computer vision dataset containing 60,000 32x32 color images across 10 different classes:
- ✈️ Airplane
- 🚗 Automobile  
- 🐦 Bird
- 🐱 Cat
- 🦌 Deer
- 🐕 Dog
- 🐸 Frog
- 🐴 Horse
- 🚢 Ship
- 🚚 Truck

This project implements a custom CNN architecture that learns to classify these images with impressive accuracy using modern deep learning techniques.

## 🚀 Features

- **Custom CNN Architecture** - Built from scratch without pre-trained models
- **Modern Training Techniques** - OneCycle learning rate, AdamW optimizer, Batch Normalization
- **Data Augmentation** - Random crops and flips for better generalization
- **Clean & Readable Code** - Well-structured and easy to understand
- **Comprehensive Evaluation** - Includes class-wise accuracy analysis

## 🏗️ Model Architecture

The network follows a simple yet effective design:

```
Input (32x32x3)
    ↓
[Conv → BatchNorm → ReLU] × 2 (32 filters) → MaxPool
    ↓
[Conv → BatchNorm → ReLU] × 2 (64 filters) → MaxPool  
    ↓
[Conv → BatchNorm → ReLU] × 2 (128 filters) → MaxPool
    ↓
Global Average Pooling
    ↓
Linear Layer (10 classes)
```

**Key Design Choices:**
- **Batch Normalization** after every convolution for stable training
- **Global Average Pooling** instead of large fully-connected layers
- **Progressive filter increase** (32 → 64 → 128) as spatial size decreases
- **Small 3x3 kernels** throughout the network

## 📈 Performance

With this architecture and training strategy, you can expect:
- **80-85% test accuracy** after 30 epochs
- **Fast training** - completes in reasonable time on a single GPU
- **Good generalization** - minimal overfitting thanks to proper regularization

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.7+
- PyTorch 1.8+
- torchvision
- CUDA-capable GPU (recommended)

### Install Dependencies

```bash
pip install torch torchvision
```

## 🎯 Quick Start

### Training the Model


The script will automatically:
- Download the CIFAR-10 dataset
- Apply data augmentation
- Train the model for 30 epochs
- Print training progress and test accuracy

### Evaluating the Model

Use the evaluation functions to analyze model performance:

```python
from eval import evaluate_model, class_wise_accuracy

# Evaluate overall accuracy
accuracy = evaluate_model(model, test_loader, device)
print(f"Test Accuracy: {accuracy:.2f}%")

# Get detailed class-wise performance
class_wise_accuracy(model, test_loader, device, class_names)
```

## ⚙️ Training Strategy

The success comes from combining several modern techniques:

### 1. **Data Augmentation**
- Random horizontal flips
- Random crops (padding + random 32x32 crop)
- Normalization using CIFAR-10 statistics

### 2. **Optimization**
- **Optimizer**: AdamW with weight decay (1e-4)
- **Learning Rate**: OneCycle policy (max_lr=1e-3)
- **Batch Size**: 128
- **Epochs**: 100

### 3. **Regularization**
- Batch Normalization in every convolutional block
- Weight decay via AdamW
- Data augmentation as implicit regularization

## 🎓 Key Concepts Demonstrated

This project serves as an excellent learning resource for:

- **CNN Architecture Design** - How to structure a modern convolutional network
- **Training Best Practices** - Proper use of learning rate scheduling and optimization
- **Data Preprocessing** - Effective augmentation and normalization strategies
- **Model Evaluation** - Beyond just accuracy to class-wise performance analysis

## 🔧 Customization

Feel free to experiment with:
- Different architecture variations (more layers, different filter sizes)
- Alternative optimization strategies
- Additional data augmentation techniques
- Different learning rate schedules

## 🤝 Contributing

Contributions are welcome! If you have ideas for improvements or find any issues, please feel free to open an issue or submit a pull request.

## 📝 License

This project is open source and available under the MIT License.

---

**Happy coding!** 🚀 If this project helped you learn something new, give it a ⭐!
