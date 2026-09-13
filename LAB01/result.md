# Experimental Results

## Overview

This section presents the experimental results of the transfer learning and deep feature classification experiments performed on the selected five-class skin lesion dataset.

The experiments cover three main aspects:

1. **Transfer Learning Models** — comparison of eight pretrained CNN architectures.
2. **Classifiers on Deep Features** — comparison of traditional machine learning classifiers using extracted deep features.
3. **Computational Efficiency** — comparison of parameter count, model size, FLOPs, inference time, and classification accuracy.

The official **Test** split was kept separate for final evaluation. The official **Train** split was used for a stratified 90/10 train-validation split. The transfer learning procedure used a classification-head warm-up followed by fine-tuning, with validation-based early stopping and restoration of the best validation-loss weights.

The experiment was performed on five manually selected classes:

- actinic keratosis
- basal cell carcinoma
- melanoma
- nevus
- pigmented benign keratosis

---

## Table 1 — Comparison of Transfer Learning Models

| Model | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | AUC (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| AlexNet | 57.50 | 60.97 | 57.50 | 54.29 | 85.86 |
| VGG16 | 63.75 | 66.82 | 63.75 | 59.71 | 89.80 |
| VGG19 | 65.00 | 68.10 | 65.00 | 60.82 | **92.85** |
| ResNet18 | 62.50 | 69.79 | 62.50 | 59.05 | 87.68 |
| **ResNet50** | **70.00** | **75.97** | **70.00** | **67.64** | 88.89 |
| ResNet101 | 63.75 | 74.11 | 63.75 | 62.11 | 90.84 |
| DenseNet121 | 58.75 | 67.39 | 58.75 | 53.95 | 87.85 |
| EfficientNet-B0 | 60.00 | 73.55 | 60.00 | 57.01 | 90.61 |

### Summary

Among the evaluated transfer learning models, **ResNet50 achieved the highest test accuracy of 70.00%**, along with the highest Precision (75.97%), Recall (70.00%), and F1-Score (67.64%).

**VGG19 achieved the highest AUC of 92.85%**, indicating the strongest overall class-ranking performance according to AUC among the evaluated transfer learning models.

Based on the final test results, **ResNet50 provided the strongest predictive performance** among the evaluated CNN architectures.

---

## Table 2 — Comparison of Different Classifiers

Deep features extracted using the **VGG16 feature extractor** were evaluated using several traditional machine learning classifiers. The feature extractor was selected using the validation-based model-selection procedure in the notebook.

| Feature Extractor | Classifier | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | AUC (%) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Deep Features (VGG16) | Logistic Regression | 55.00 | 57.84 | 55.00 | 48.07 | 88.28 |
| Deep Features (VGG16) | Decision Tree | 60.00 | 58.12 | 60.00 | 55.05 | 74.99 |
| Deep Features (VGG16) | Random Forest | 56.25 | 59.77 | 56.25 | 49.48 | **93.29** |
| Deep Features (VGG16) | K-Nearest Neighbors (KNN) | 58.75 | 63.82 | 58.75 | 53.60 | 84.78 |
| Deep Features (VGG16) | Linear SVM | 57.50 | 60.95 | 57.50 | 52.49 | 88.44 |
| **Deep Features (VGG16)** | **RBF-SVM** | **65.00** | **73.18** | **65.00** | **60.21** | 92.54 |
| Deep Features (VGG16) | XGBoost | 57.50 | 59.20 | 57.50 | 51.16 | 90.29 |

### Summary

Among the traditional classifiers, **RBF-SVM achieved the highest classification accuracy of 65.00%**, together with the highest Precision (73.18%), Recall (65.00%), and F1-Score (60.21%).

**Random Forest achieved the highest AUC of 93.29%** among the evaluated classifiers, despite having a lower classification accuracy of 56.25%.

These results show that classifier choice affects the performance of the same deep feature representation. In this experiment, **RBF-SVM provided the strongest classification performance by accuracy, precision, recall, and F1-score**.

---

## Table 3 — Computational Efficiency Comparison

| Model | Parameters (M) | Model Size (MB) | FLOPs (G) | Inference Time (ms) | Accuracy (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| AlexNet | 57.02 | 217.54 | 1.42 | **2.08** | 57.50 |
| VGG16 | 134.28 | 512.25 | 30.93 | 9.82 | 63.75 |
| VGG19 | 139.59 | 532.51 | 39.26 | 11.49 | 65.00 |
| ResNet18 | 11.18 | 42.72 | 3.65 | 3.69 | 62.50 |
| **ResNet50** | **23.52** | **90.02** | **8.26** | 8.65 | **70.00** |
| ResNet101 | 42.51 | 162.76 | 15.73 | 17.85 | 63.75 |
| DenseNet121 | 6.96 | 27.13 | 5.79 | 15.58 | 58.75 |
| **EfficientNet-B0** | **4.01** | **15.60** | **0.83** | 8.31 | 60.00 |

### Summary

**EfficientNet-B0 was the most lightweight architecture** in terms of parameter count, model size, and FLOPs. It required only **4.01 million parameters**, **15.60 MB** of model storage, and **0.83 GFLOPs**.

**ResNet50 achieved the highest test accuracy of 70.00%** while requiring substantially fewer parameters and FLOPs than the VGG16 and VGG19 architectures.

**VGG19 had the largest computational footprint**, with **139.59 million parameters**, a **532.51 MB** model size, and **39.26 GFLOPs**.

The results demonstrate a trade-off between predictive performance and computational efficiency. EfficientNet-B0 provides a lightweight architecture, whereas ResNet50 provides the strongest classification accuracy in this experiment.

---

## Overall Findings

The experiments produced the following key findings:

- **ResNet50 achieved the highest transfer-learning test accuracy (70.00%).**
- **ResNet50 achieved the highest transfer-learning Precision (75.97%), Recall (70.00%), and F1-Score (67.64%).**
- **VGG19 achieved the highest transfer-learning AUC (92.85%).**
- **RBF-SVM achieved the highest classifier accuracy (65.00%), Precision (73.18%), Recall (65.00%), and F1-Score (60.21%)** when using VGG16 deep features.
- **Random Forest achieved the highest classifier AUC (93.29%).**
- **EfficientNet-B0 had the fewest parameters (4.01M), smallest model size (15.60 MB), and lowest FLOPs (0.83G).**
- **AlexNet had the lowest measured inference time (2.08 ms)** in this experiment.
- **VGG19 had the largest parameter count, model size, and FLOPs** among the architectures included in the computational-efficiency comparison.
- The results demonstrate a clear **performance-efficiency trade-off** between different CNN architectures.

---

## Best Results at a Glance

| Metric | Best Model / Classifier | Score |
| --- | --- | ---: |
| Transfer Learning Accuracy | ResNet50 | **70.00%** |
| Transfer Learning Precision | ResNet50 | **75.97%** |
| Transfer Learning Recall | ResNet50 | **70.00%** |
| Transfer Learning F1-Score | ResNet50 | **67.64%** |
| Transfer Learning AUC | VGG19 | **92.85%** |
| Classifier Accuracy | RBF-SVM | **65.00%** |
| Classifier Precision | RBF-SVM | **73.18%** |
| Classifier Recall | RBF-SVM | **65.00%** |
| Classifier F1-Score | RBF-SVM | **60.21%** |
| Classifier AUC | Random Forest | **93.29%** |
| Lowest Parameters | EfficientNet-B0 | **4.01 M** |
| Lowest Model Size | EfficientNet-B0 | **15.60 MB** |
| Lowest FLOPs | EfficientNet-B0 | **0.83 G** |
| Lowest Inference Time | AlexNet | **2.08 ms** |

---

## Conclusion

Overall, **ResNet50 provided the best predictive performance** among the evaluated transfer learning models, achieving **70.00% test accuracy**, **75.97% Precision**, **70.00% Recall**, and **67.64% F1-Score**.

For deep-feature-based classification, **RBF-SVM produced the strongest classification results by accuracy, precision, recall, and F1-score**, while **Random Forest achieved the highest AUC**.

From a computational perspective, **EfficientNet-B0 was the most lightweight architecture**, with the lowest parameter count, model size, and FLOPs among the evaluated models. Therefore, the preferred architecture depends on the deployment objective: **ResNet50 is preferable when predictive accuracy is prioritized, while EfficientNet-B0 is attractive when computational efficiency and model compactness are prioritized**.
