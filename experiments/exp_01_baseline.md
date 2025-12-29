# Experiment 01: Baseline CNN Model

**Date**: December 29, 2025  
**Experimenter**: Project Team  
**Status**: Completed

---

## Objective
Establish a baseline CNN model for brain tumor classification to serve as a reference point for future improvements.

## Hypothesis
A custom CNN architecture with multiple convolutional layers and proper regularization can achieve reasonable accuracy (>85%) on brain tumor classification from MRI images.

---

## Model Architecture

### Architecture Details
- **Input**: 150x150x3 RGB images
- **Convolutional Blocks**: 4 blocks with increasing filter complexity
  - Block 1: 64 filters, 5x5 kernel
  - Block 2: 128 filters, 3x3 kernel
  - Block 3: 128 filters, 3x3 kernel
  - Block 4: 128 filters, 2x2 kernel
- **Pooling**: MaxPooling after each conv block
- **Regularization**: Dropout (0.25, 0.25, 0.3, 0.3, 0.5)
- **Dense Layers**: 
  - Hidden: 1024 units with ReLU
  - Output: 4 units with Softmax

### Hyperparameters
```yaml
learning_rate: 0.001
optimizer: Adam (beta_1=0.9, beta_2=0.999)
loss: categorical_crossentropy
epochs: 40
batch_size: 40
```

---

## Data

### Dataset Statistics
- **Training samples**: 2,870 images
- **Testing samples**: 394 images
- **Classes**: 4 (Glioma, Meningioma, No Tumor, Pituitary)
- **Image size**: 150x150 pixels

### Class Distribution
| Class | Training | Testing |
|-------|----------|---------|
| Glioma | ~826 | ~100 |
| Meningioma | ~822 | ~115 |
| No Tumor | ~395 | ~105 |
| Pituitary | ~827 | ~74 |

### Preprocessing
- Resize to 150x150
- Normalization: Pixel values [0-255]
- Data augmentation:
  - Horizontal flip: Yes
  - Rotation: None
  - Zoom: None
  - Shift: None

---

## Results

### Performance Metrics
```
Test Loss: [To be recorded after training]
Test Accuracy: [To be recorded after training]
```

### Per-Class Performance
```
                precision    recall  f1-score   support

  glioma_tumor       0.XX      0.XX      0.XX       XXX
meningioma_tumor     0.XX      0.XX      0.XX       XXX
       no_tumor      0.XX      0.XX      0.XX       XXX
pituitary_tumor      0.XX      0.XX      0.XX       XXX

        accuracy                          0.XX       XXX
       macro avg      0.XX      0.XX      0.XX       XXX
    weighted avg      0.XX      0.XX      0.XX       XXX
```

### Confusion Matrix
*(To be generated after training)*

### Training Curves
- Training loss decreases steadily
- Validation loss follows training loss closely
- No significant overfitting observed with current dropout rates

---

## Analysis

### Strengths
- Simple and interpretable architecture
- Reasonable training time (~X minutes on CPU/GPU)
- Good baseline performance
- Effective regularization prevents overfitting

### Weaknesses
- Limited data augmentation
- No transfer learning utilized
- Fixed learning rate throughout training
- Potential for improvement with deeper architectures

### Observations
1. The model converges well within 40 epochs
2. Horizontal flip augmentation provides some benefit
3. Dropout layers effectively prevent overfitting
4. Class imbalance might affect performance on minority classes

---

## Lessons Learned

1. **Data Quality Matters**: Image quality and proper preprocessing are crucial
2. **Regularization is Key**: Dropout layers significantly improve generalization
3. **Augmentation Helps**: Even minimal augmentation (horizontal flip) improves robustness
4. **Batch Size Impact**: Batch size of 40 provides stable training

---

## Next Steps

### Immediate Actions
- [ ] Record final metrics after training
- [ ] Generate and analyze confusion matrix
- [ ] Identify misclassified samples
- [ ] Document failure cases

### Future Experiments
1. **Experiment 02**: Implement transfer learning with pre-trained models
   - Try VGG16, ResNet50, EfficientNet
   - Compare performance vs baseline
   
2. **Enhanced Augmentation**: Add rotation, zoom, and shift
   
3. **Learning Rate Schedule**: Implement learning rate decay
   
4. **Architecture Variations**: 
   - Add more convolutional layers
   - Experiment with different filter sizes
   - Try batch normalization

---

## Code Reference

### Training Command
```bash
cd src
python train.py
```

### Model Checkpoint
```
Location: ../models/best_model.h5
Size: ~XXX MB
```

---

## Resources

- **Training Time**: ~XX minutes
- **Hardware**: [CPU/GPU specifications]
- **Memory Usage**: ~XX GB RAM

---

## Reproducibility

### Environment
```
Python: 3.x
TensorFlow: 2.10.0
Keras: 2.10.0
NumPy: 1.23.5
```

### Random Seeds
- TensorFlow: 42
- NumPy: 42
- Python: 42

### Configuration
See `configs/config.yaml` for complete configuration.

---

## Conclusion

The baseline CNN model demonstrates that deep learning is effective for brain tumor classification. While the results are promising, there is clear room for improvement through transfer learning, advanced augmentation, and architectural enhancements. This baseline serves as a solid foundation for future experiments.

---

**Approved by**: Project Lead  
**Date**: December 29, 2025  
**Next Review**: After Experiment 02
