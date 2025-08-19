# Selection-Based Histopathology Cancer Detection

## Introduction
Breast cancer remains one of the most common and deadly cancers worldwide. Histopathology images are critical for early detection and diagnosis. However, manual inspection is subjective and time-consuming.

In this work, we present a framework that combines deep learning with feature selection to support automated breast cancer detection from histopathology images. The approach involves:

- A deep CNN pretrained on CIFAR-100 and fine-tuned on histopathology data.
- A feature selection pipeline combining R-Relief and Pearson Correlation Coefficient (PCC), followed by PCA.
- A comprehensive evaluation with multiple classifiers (SVMs with different kernels, Decision Trees, Ensembles).

---

## Background

- CNNs: extract informative features from medical images.
- Raw features: often high-dimensional and redundant.
- Feature selection improves accuracy and efficiency:

    - R-Relief: Identifies discriminative features.
    - PCC: Removes redundant/correlated features.
    - PCA: Fuses selected features into a compact vector.

---

## Proposed Methodology
### Step 1: Deep CNN Training
- Pretrained CNN model on CIFAR-100 (60,000 images, 100 classes).
- Fine-tuned on histopathology dataset.
- Training details:
    
    - Epochs: 60
    - Batch size: 128
    - Initial learning rate: 0.01
    - Optimizer: "sgdm"

### Step 2: Feature Extraction
- Extract deep feature vectors (DF) from the last fully connected layers.
- Each histopathology image to get feature vectors.

### Step 3: Feature Selection
- **Parallel selection strategy**:

    - Apply R-Relief to rank features by discriminative power.
    - Apply PCC to remove redundant / irrelevant features.

- **Fusion step**: Apply PCA on outputs of R-Relief with PCC to get final compact feature vector (FV).

### Step 4: Classification
- Multiple classifiers trained on selected features:

    - Linear SVM (LSVM)
    - Quadratic SVM (QSVM)
    - Cubic SVM (CSVM)
    - Gaussian SVM (Fine, Medium, Coarse)
    - Fine Tree (FT)
    - Ensemble Boosted Trees (EBT)
    - Ensemble Subspace Discriminant (ESD)

- Evaluation metric: Accuracy, Precision, Recall, F1-score with 5-fold cross-validation.

---

## Dataset
- Dataset used: Histopathology IDC dataset (breast cancer tissue images).
- Classes: IDC positive vs IDC negative.
- Preprocessing:

    - Resize images to fixed dimensions.
    - Normalization to [0,1].
    - Train/test split with stratification.

## Quick Start

### 1. Install Dependencies
```
pip install -r requirements.txt
```

### 2. Pretrain CNN

### 3. Extract Features

### 4. Feature Selection

### 5. Feature Selection

## Experiments
- Baseline: CNN features without FS.
- Comparison: CNN features with FS (R-Relief + PCC + PCA).
- Evaluation: 5-fold cross-validation.

## Results

## Limitations & Future Work
Classification relied only on deep features, without combining them with classical hand-crafted features.
No data augmentation methods were applied.
No preprocessing steps were introduced to improve input quality.
- Future work:

    - Apply preprocessing techniques and feature fusion across multiple domains to enhance accuracy.
    - Develop an intelligent CAD system for IDC classification.
    - Explore advanced CNN models such as DenseNet and CapsuleNet, integrated with diverse feature fusion and selection strategies.

## Citation
If you use this repository, please cite:

```bibtex
@article{selectioncancer2022,
  title={A Framework of Deep Learning and Selection-Based Breast Cancer Detection from Histopathology Images},
  author={Muhammad Junaid Umer, Muhammad Sharif, Majed Alhaisoni, Usman Tariq, Ye Jin Kim and Byoungchol Chang},
  journal={Computer Systems Science & Engineering},
  year={2022},
  publisher = {Computer Systems Science & Engineering}
}
```