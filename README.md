# Selection-Based Cancer Detection from Histopathology Images

## Introduction
Breast cancer remains one of the most common and deadly cancers worldwide. Histopathology images are critical for early detection and diagnosis. However, manual inspection is subjective and time-consuming.

This project proposes a deep learning + feature selection framework to automatically detect breast cancer from histopathology images. Main contributions:

- A deep CNN pretrained on CIFAR-100 and fine-tuned on histopathology data.
- A hybrid feature selection framework using R-Relief and Pearson Correlation Coefficient (PCC), fused with PCA.
- A comprehensive evaluation with multiple classifiers (SVMs with different kernels, Decision Trees, Ensembles).

---

## Background
CNNs are powerful for feature extraction from medical images.

Raw deep features may be high-dimensional and redundant.

Feature Selection (FS) reduces dimensionality → better accuracy, efficiency, and generalization.

R-Relief identifies discriminative features.

PCC removes redundant/correlated features.

PCA fusion combines selected features into a final compact feature vector.

---

## Proposed Methodology

### Step 1: Deep CNN Training
- Pretrained CNN model on CIFAR-100 (60,000 images, 100 classes).
- Fine-tuned on histopathology dataset.
- Training details:
    - Epochs: 60
    - Batch size: 128
    - Initial learning rate: 0.01
    - Optimizer: SGD with momentum

### Step 2: Feature Extraction
- Extract deep feature vectors (DF) from the last fully connected layers.
- Each histopathology image → vector of dimension d.

### Step 3: Feature Selection
- **Parallel selection strategy**:
    - Apply R-Relief to rank features by discriminative power.
    - Apply PCC to remove redundant/irrelevant features.
- **Fusion step**: Apply PCA on outputs of R-Relief + PCC → final compact feature vector (FV).

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

## Citation

If you use this repository, please cite:

@article{selectioncancer2022,
  title={A Framework of Deep Learning and Selection-Based Breast Cancer Detection from Histopathology Images},
  author={Muhammad Junaid Umer, Muhammad Sharif, Majed Alhaisoni, Usman Tariq, Ye Jin Kim and Byoungchol Chang},
  journal={Computer Systems Science & Engineering},
  year={2022}
}
