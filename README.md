# Image Classification Using Deep Learning Models for Leukemia Type Detection

## Project Overview

This project focuses on the automated detection and classification of leukemia types from blood cell images using Deep Learning. By leveraging Transfer Learning with state-of-the-art architectures, the system identifies 6 distinct classes of blood cells (including healthy and various leukemia types) to assist in medical diagnosis.

The project implements a robust pipeline including data augmentation, two-stage training (feature extraction + fine-tuning), and comprehensive evaluation metrics.

## Dataset

The model is trained on a dataset of blood cell images classified into **6 categories**:

1. **ALL** (Acute Lymphoblastic Leukemia)
2. **AML** (Acute Myeloid Leukemia)
3. **CLL** (Chronic Lymphocytic Leukemia)
4. **CML** (Chronic Myeloid Leukemia)
5. **MM** (Multiple Myeloma)
6. **Healthy** (Normal cell)

* **Total Images:** ~2,400 (Split: 60% Train, 20% Validation, 20% Test)
* **Image Size:** 224 x 224 pixels

## Methodology

### 1. Data Preprocessing & Augmentation

To handle the limited dataset size and prevent overfitting, we applied:

* **tf.data Pipeline:** Efficient loading with `cache()` and `prefetch()`.
* **Augmentation (Training only):**
  * Random Flip (Horizontal)
  * Random Rotation (10%)
  * Random Zoom (10%)
  * Random Contrast (10%)
* **Preprocessing:** specific to each model (e.g., VGG16 preprocessing).

### 2. Model Architectures

We experimented with three powerful Transfer Learning models pre-trained on ImageNet:

* **VGG16** (Simpler, deep architecture)
* **DenseNet121** (Densely connected networks)
* **EfficientNetB3** (Optimized for efficiency and accuracy)

### 3. Training Strategy (Two-Stage)

We used a **Two-Stage Training** approach to maximize performance:

* **Stage 1 (Head Training):** Frozen base model. Trained only the custom classifier head (Global Average Pooling -> Dropout 0.35 -> Softmax) for 15 epochs using `Adam(lr=1e-3)`.
* **Stage 2 (Fine-Tuning):** Unfroze the top 30% of the base model layers. Retrained with a lower learning rate `Adam(lr=1e-5)` for 20 epochs to adapt features to leukemia cells.
* **Callbacks:** `ModelCheckpoint` (save best), `EarlyStopping` (patience=6), `ReduceLROnPlateau` (factor=0.3).

## Results & Analysis

We evaluated all models on the unseen Test set (20% of data).

| Model Ranking      | Test Accuracy    | Performance Notes                                                                    |
| :----------------- | :--------------- | :----------------------------------------------------------------------------------- |
| **1. VGG16** | **90.62%** | **Best Model.** Achieved 100% precision/recall on 'ALL' and 'Healthy' classes. |
| 2. DenseNet121     | 86.87%           | Strong performer but struggled slightly with 'CML' class.                            |
| 3. EfficientNetB3  | 74.17%           | Lowest accuracy, likely due to overfitting on the small dataset.                     |

### Why VGG16 Performed Best?

Despite being an older architecture, **VGG16** outperformed the more modern EfficientNetB3 and DenseNet121.

1. **Dataset Size:** Our dataset is relatively small (~1,440 training images). VGG16's simpler, block-based architecture is less prone to overfitting on small data compared to the highly complex EfficientNet.
2. **Feature Generalization:** VGG16's learned features from ImageNet transferred exceptionally well to the microscopic blood cell domain, requiring less fine-tuning data to converge than the deeper networks.
3. **Class Performance:** VGG16 achieved **perfect scores (1.0)** for `ALL` and `Healthy` classes, whereas EfficientNet struggled significantly with `AML` and `CML` (F1-scores ~0.50), pulling down its average.

## Requirements

* Python
* TensorFlow / Keras
* NumPy, Matplotlib, Seaborn, Scikit-learn
* Google Colab

## Author

**Md. Tausif Jafar**
