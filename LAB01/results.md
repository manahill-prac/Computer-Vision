# Experimental Results

## Overview

This section presents the experimental results of the transfer learning and deep feature classification experiments performed on the selected five-class skin lesion dataset.

The evaluation covers three main aspects:

1. **Transfer Learning Models** — comparison of eight pretrained CNN architectures.
2. **Classifiers on Deep Features** — comparison of traditional machine learning classifiers using extracted deep features.
3. **Computational Efficiency** — comparison of model size, parameter count, computational complexity, inference time, and classification accuracy.

---

## Table 1 — Comparison of Transfer Learning Models

The following table compares the performance of eight pretrained convolutional neural network architectures.

| Model           | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | AUC (%) |
| --------------- | -----------: | ------------: | ---------: | -----------: | ------: |
| AlexNet         |        57.50 |         60.97 |      57.50 |        54.29 |   85.86 |
| VGG16           |        63.75 |         66.82 |      63.75 |        59.71 |   89.80 |
| VGG19           |        65.00 |         68.10 |      65.00 |        60.82 |   92.85 |
| ResNet18        |        62.50 |         69.79 |      62.50 |        59.05 |   87.68 |
| **ResNet50**    |    **70.00** |     **75.97** |  **70.00** |    **67.64** |   88.89 |
| ResNet101       |        63.75 |         74.11 |      63.75 |        62.11 |   90.84 |
| DenseNet121     |        58.75 |         67.39 |      58.75 |        53.95 |   87.85 |
| EfficientNet-B0 |        60.00 |         73.55 |      60.00 |        57.01 |   90.61 |

### Summary

Among the evaluated transfer learning models, **ResNet50 achieved the highest classification accuracy of 70.00%**, along with the highest Precision (75.97%) and F1-Score (67.64%).

**VGG19 achieved the highest AUC of 92.85%**, indicating strong class-ranking capability despite having lower accuracy than ResNet50.

Overall, ResNet50 provided the strongest balance of classification performance among the evaluated models.

---

## Table 2 — Comparison of Different Classifiers

Deep features extracted from the selected feature extractor were evaluated using several traditional machine learning classifiers.

| Feature Extractor | Classifier                | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) |   AUC (%) |
| ----------------- | ------------------------- | -----------: | ------------: | ---------: | -----------: | --------: |
| Deep Features     | Logistic Regression       |        55.00 |         57.84 |      55.00 |        48.07 |     88.28 |
| Deep Features     | Decision Tree             |        60.00 |         58.12 |      60.00 |        55.05 |     74.99 |
| Deep Features     | Random Forest             |        56.25 |         59.77 |      56.25 |        49.48 | **93.29** |
| Deep Features     | K-Nearest Neighbors (KNN) |        58.75 |         63.82 |      58.75 |        53.60 |     84.78 |
| Deep Features     | Linear SVM                |        57.50 |         60.95 |      57.50 |        52.49 |     88.44 |
| **Deep Features** | **RBF-SVM**               |    **65.00** |     **73.18** |  **65.00** |    **60.21** |     92.54 |
| Deep Features     | XGBoost                   |        57.50 |         59.20 |      57.50 |        51.16 |     90.29 |

### Summary

Among the traditional classifiers, **RBF-SVM achieved the highest classification accuracy of 65.00%**, together with a Precision of 73.18% and F1-Score of 60.21%.

Interestingly, **Random Forest achieved the highest AUC of 93.29%**, suggesting strong class-ranking performance even though its classification accuracy was only 56.25%.

The results indicate that the choice of classifier has a substantial effect on the performance of deep feature representations. In this experiment, **RBF-SVM provided the strongest overall classification performance**.

---

## Table 3 — Computational Efficiency Comparison

This table compares the computational requirements and inference efficiency of the evaluated CNN architectures.

| Model               | Parameters (M) | Model Size (MB) | FLOPs (G) | Inference Time (ms) | Accuracy (%) |
| ------------------- | -------------: | --------------: | --------: | ------------------: | -----------: |
| AlexNet             |          57.02 |          217.54 |      1.42 |                2.08 |        57.50 |
| VGG16               |         134.28 |          512.25 |     30.93 |                9.82 |        63.75 |
| VGG19               |         139.59 |          532.51 |     39.26 |               11.49 |        65.00 |
| ResNet18            |          11.18 |           42.72 |      3.65 |                3.69 |        62.50 |
| **ResNet50**        |          23.52 |           90.02 |      8.26 |                8.65 |    **70.00** |
| DenseNet121         |           6.96 |           27.13 |      5.79 |               15.58 |        58.75 |
| **EfficientNet-B0** |       **4.01** |       **15.60** |  **0.83** |                8.31 |        60.00 |

### Summary

**EfficientNet-B0 is the most lightweight architecture**, requiring only 4.01 million parameters, 15.60 MB of storage, and 0.83 GFLOPs.

**ResNet50 achieved the highest accuracy (70.00%)** while maintaining substantially lower computational requirements than VGG16 and VGG19.

VGG19 has the largest computational footprint, with 139.59 million parameters, 532.51 MB model size, and 39.26 GFLOPs.

These results highlight the trade-off between predictive performance and computational efficiency. EfficientNet-B0 provides an attractive lightweight option, while ResNet50 provides the strongest classification performance.

---

## Overall Findings

The experiments reveal several important observations:

* **ResNet50 achieved the highest overall accuracy (70.00%)** among the evaluated transfer learning models.
* **VGG19 achieved the highest transfer-learning AUC (92.85%)**.
* **RBF-SVM achieved the highest accuracy (65.00%)** among the evaluated traditional classifiers using deep features.
* **Random Forest achieved the highest classifier AUC (93.29%)**.
* **EfficientNet-B0 was the most parameter-efficient model**, with only 4.01 million parameters.
* **VGG19 had the largest computational footprint**, requiring 139.59 million parameters and 39.26 GFLOPs.
* The results demonstrate a clear **performance-efficiency trade-off** between larger CNN architectures and lightweight architectures.

### Best Results at a Glance

| Metric                      | Best Model / Classifier |        Score |
| --------------------------- | ----------------------- | -----------: |
| Transfer Learning Accuracy  | ResNet50                |   **70.00%** |
| Transfer Learning Precision | ResNet50                |   **75.97%** |
| Transfer Learning Recall    | ResNet50                |   **70.00%** |
| Transfer Learning F1-Score  | ResNet50                |   **67.64%** |
| Transfer Learning AUC       | VGG19                   |   **92.85%** |
| Classifier Accuracy         | RBF-SVM                 |   **65.00%** |
| Classifier Precision        | RBF-SVM                 |   **73.18%** |
| Classifier Recall           | RBF-SVM                 |   **65.00%** |
| Classifier F1-Score         | RBF-SVM                 |   **60.21%** |
| Classifier AUC              | Random Forest           |   **93.29%** |
| Lowest Parameters           | EfficientNet-B0         |   **4.01 M** |
| Lowest Model Size           | EfficientNet-B0         | **15.60 MB** |
| Lowest FLOPs                | EfficientNet-B0         |   **0.83 G** |

---

## Conclusion

Overall, **ResNet50 provided the best predictive performance**, achieving 70.00% test accuracy and the highest Precision and F1-Score among the transfer learning models.

For deep-feature-based classification, **RBF-SVM produced the strongest accuracy**, while Random Forest produced the highest AUC.

From a computational perspective, **EfficientNet-B0 was the most lightweight architecture**, making it particularly attractive for resource-constrained deployment scenarios. Therefore, the final model selection depends on the intended priority: **ResNet50 for higher predictive performance or EfficientNet-B0 for computational efficiency**.

