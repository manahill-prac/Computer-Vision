# Computer Vision Lab 03 — Results & Analysis

## Edge Detection Techniques and Their Impact on Classification Performance

This document contains the complete results, observations, required tables, discussion answers, viva answers, and conclusion for Computer Vision Lab 03. It is intended to be readable independently of the notebook/code.

---

## 1. Lab Overview

### Objectives
- Understand basic edge detection concepts.
- Implement Sobel, Prewitt, Laplacian, LoG, and Canny edge detectors.
- Study the effect of noise on edge detection.
- Analyze the effect of smoothing before edge detection.
- Compare raw, filtered, and edge-based classification.
- Understand traditional edge features versus learned CNN features.

### Tasks Covered
1. Comparative Edge Detection
2. Effect of Noise on Edge Detection
3. Canny Parameter Analysis
4. Classification Using Edge Maps
5. Classification Performance Comparison
6. Visual Comparison of Classification Results
7. Discussion and Viva Questions

---

# 2. Experimental Setup

**Device:** CUDA-enabled GPU

### Selected Classes

| ID | Class |
|---:|---|
| 0 | Actinic keratosis |
| 1 | Basal cell carcinoma |
| 2 | Melanoma |
| 3 | Nevus |
| 4 | Pigmented benign keratosis |

### Dataset Split

| Set | Images |
|---|---:|
| Training | 1,572 |
| Validation | 175 |
| Official test | 80 |

### Training Images by Selected Class

| Class | Training Images |
|---|---:|
| Actinic keratosis | 114 |
| Basal cell carcinoma | 376 |
| Melanoma | 438 |
| Nevus | 357 |
| Pigmented benign keratosis | 462 |
| **Total shown by class** | **1,747** |

**Note:** The execution output reports 1,572 images in the actual training split after the train/validation split. The per-class table above is the supplied class-count output.

---

# 3. Edge Detection Methods

Implemented methods:

1. Sobel Gx
2. Sobel Gy
3. Sobel Gradient Magnitude
4. Prewitt
5. Laplacian
6. Laplacian of Gaussian (LoG)
7. Canny

Required visual progression:

**Original → Sobel → Prewitt → Laplacian → LoG → Canny**

Representative images were taken from multiple classes as required.

---

# 4. Task 1 — Comparative Edge Detection

### Sobel Gx
Detects intensity changes primarily along the x-direction.

### Sobel Gy
Detects intensity changes primarily along the y-direction.

### Sobel Gradient Magnitude
Combines horizontal and vertical gradient responses to represent overall edge strength.

### Prewitt
A first-order derivative detector similar to Sobel, with a simpler kernel weighting.

### Laplacian
A second-order derivative detector that responds strongly to rapid intensity changes.

### LoG
Laplacian of Gaussian: Gaussian smoothing is applied before Laplacian edge detection to reduce some high-frequency noise.

### Canny
A multi-stage edge detector involving smoothing, gradient estimation, non-maximum suppression, double thresholding, and hysteresis.

---

# 5. Task 2 — Effect of Noise on Edge Detection

Two artificial noise types were used:

- Gaussian noise
- Salt-and-Pepper noise

The required comparisons were:

1. Original image
2. Noisy image
3. Noisy image after Gaussian filtering
4. Noisy image after Median filtering

## Table 1 — Effect of Noise and Preprocessing on Edge Detection

The supplied execution output did not contain numerical edge-quality scores, so the qualitative entries below are observations rather than invented measurements.

| Edge Detector | Input Image | Noise Type | Preprocessing | Edge Quality | Noise Sensitivity | Observations |
|---|---|---|---|---|---|---|
| Sobel | Original | None | None | Clear | Low–Moderate | Clear first-order intensity boundaries. |
| Sobel | Noisy | Gaussian | None | Reduced | Moderate–High | Noise introduces additional intensity variations and false/rough edge responses. |
| Sobel | Noisy | Salt & Pepper | None | Reduced | High | Isolated impulse pixels can produce false edge responses. |
| Sobel | Noisy | Gaussian | Gaussian Filter | Improved | Lower | Smoothing reduces high-frequency noise before edge detection. |
| Sobel | Noisy | Salt & Pepper | Median Filter | Improved | Lower | Median filtering suppresses isolated impulse noise effectively. |
| Prewitt | Original | None | None | Clear | — | Produces a first-order edge representation similar to Sobel. |
| Laplacian | Original | None | None | Sharp | High | Second-order differentiation emphasizes rapid intensity changes and is noise-sensitive. |
| LoG | Noisy | Gaussian | Gaussian Filter | Improved | Reduced | Gaussian smoothing before Laplacian processing suppresses some noise responses. |
| Canny | Original | None | Built-in smoothing | Clear and continuous | Lower than direct derivative methods | Multi-stage processing produces thin, controlled edges. |
| Canny | Noisy | Gaussian | Gaussian Filter | Improved | Reduced | Pre-smoothing helps suppress Gaussian noise. |
| Canny | Noisy | Salt & Pepper | Median Filter | Improved | Reduced | Median preprocessing suppresses impulse noise before Canny. |

### Main observations

- Noise introduces additional intensity transitions that can be detected as false edges.
- Gaussian noise produces widespread random intensity variations.
- Salt-and-Pepper noise produces isolated extreme pixels.
- Gaussian filtering is useful for reducing Gaussian-type high-frequency variations.
- Median filtering is particularly useful for Salt-and-Pepper noise.
- Excessive smoothing can remove fine details as well as noise.
- The Laplacian is especially sensitive to noise because it uses a second-order derivative.
- Canny provides a controlled edge representation through its multi-stage process.

---

# 6. Task 3 — Canny Parameter Analysis

Four configurations were evaluated.

## Table 2 — Canny Parameter Analysis

| Configuration | Low Threshold | High Threshold | Kernel Size | Edge Quality | Number of Detected Edges | Edge Density (%) | Observation |
|---|---:|---:|---|---|---:|---:|---|
| Canny-1 | 30 | 100 | 3×3 | More dense | 9,416 | 18.765944 | Low thresholds detect many weak and strong transitions. |
| Canny-2 | 50 | 150 | 3×3 | Balanced | 4,399 | 8.767140 | Produces a moderately sparse edge representation. |
| Canny-3 | 100 | 200 | 3×3 | Sparse | 1,441 | 2.871891 | Higher thresholds retain fewer, stronger edges. |
| Canny-4 | 50 | 150 | 5×5 | Very sparse | 961 | 1.915258 | Larger smoothing kernel substantially reduces detected edges. |

### Observed trend

Increasing the thresholds reduced detected edge pixels:

- 30/100 → **9,416**
- 50/150 → **4,399**
- 100/200 → **1,441**

Changing the kernel from 3×3 to 5×5 at thresholds 50/150 reduced the count from **4,399 to 961**.

### Selected configuration

**Canny: Low = 50, High = 150, Kernel = 3×3**

This was used for the Lab 03 edge representation.

The selection provides a moderately sparse edge representation between the very dense 30/100 result and the much sparser 100/200 and 5×5 results.

---

# 7. Task 4 — Classification Dataset Representations

Three representations were prepared:

| Representation | Description |
|---|---|
| Raw | Original images |
| Filtered | Lab 02 Gaussian-filtered images |
| Edge | Canny edge images using 50/150 thresholds and 3×3 kernel |

The same train/validation/test structure was used for the representations.

---

# 8. Task 5 — Classification Results

Models:

- ResNet50 CNN
- Linear SVM
- Random Forest
- KNN

Metrics:

- Accuracy
- Precision
- Recall
- F1-score
- Training time
- Inference time

## 8.1 ResNet50 CNN

| Model | Representation | Accuracy | Precision | Recall | F1-Score | Training Time (s) | Inference Time (ms/image) |
|---|---|---:|---:|---:|---:|---:|---:|
| ResNet50 CNN | Raw | 0.7000 | 0.744012 | 0.7000 | 0.688127 | 255.064159 | 13.381857 |
| ResNet50 CNN | Filtered | 0.6000 | 0.661818 | 0.6000 | 0.573821 | 217.207354 | 11.609167 |
| ResNet50 CNN | Edge | 0.3750 | 0.348596 | 0.3750 | 0.313803 | 309.667261 | 11.919403 |

## 8.2 Classical Classifiers

| Model | Representation | Accuracy | Precision | Recall | F1-Score | Training Time (s) | Inference Time (ms/image) |
|---|---|---:|---:|---:|---:|---:|---:|
| Linear SVM | Raw | 0.5875 | 0.731128 | 0.5875 | 0.535262 | 0.920091 | 0.574955 |
| Random Forest | Raw | 0.6000 | 0.672222 | 0.6000 | 0.522484 | 8.561306 | 0.676024 |
| KNN | Raw | 0.5625 | 0.623938 | 0.5625 | 0.538473 | 0.001900 | 1.515150 |
| Linear SVM | Filtered | 0.5250 | 0.562353 | 0.5250 | 0.496508 | 1.285313 | 0.723833 |
| Random Forest | Filtered | 0.6000 | 0.680203 | 0.6000 | 0.552910 | 7.591070 | 0.810254 |
| KNN | Filtered | 0.4875 | 0.477612 | 0.4875 | 0.447337 | 0.001961 | 0.829172 |
| Linear SVM | Edge | 0.3625 | 0.337807 | 0.3625 | 0.338980 | 3.088004 | 1.068106 |
| Random Forest | Edge | 0.3875 | 0.297802 | 0.3875 | 0.332049 | 8.963535 | 0.702178 |
| KNN | Edge | 0.3000 | 0.282979 | 0.3000 | 0.280838 | 0.002229 | 0.790870 |

---

# 9. Table 3 — Cross-Lab Classification Performance

| Model / Classifier | Accuracy Raw (Lab 1) | Accuracy Filtered (Lab 2) | Accuracy Edge (Lab 3) |
|---|---:|---:|---:|
| KNN | 0.5625 | 0.4875 | 0.3000 |
| Linear SVM | 0.5875 | 0.5250 | 0.3625 |
| Random Forest | 0.6000 | 0.6000 | 0.3875 |
| ResNet50 CNN | 0.7000 | 0.6000 | 0.3750 |

## ResNet50 representation comparison

| Representation | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Raw | 0.7000 | 0.7440 | 0.7000 | 0.6881 |
| Filtered | 0.6000 | 0.6618 | 0.6000 | 0.5738 |
| Edge | 0.3750 | 0.3486 | 0.3750 | 0.3138 |

---

# 10. Task 6 — Visual Comparison of Classification Results

For ResNet50:

| Metric | Raw | Filtered | Edge |
|---|---:|---:|---:|
| Accuracy | 0.7000 | 0.6000 | 0.3750 |
| Precision | 0.7440 | 0.6618 | 0.3486 |
| Recall | 0.7000 | 0.6000 | 0.3750 |
| F1-score | 0.6881 | 0.5738 | 0.3138 |

The results show lower measured classification metrics for filtered images compared with raw images, and substantially lower metrics for edge-only images.

Raw images retain color, intensity, texture, boundaries and other spatial information. Gaussian filtering can suppress some high-frequency detail, while edge detection removes most color and intensity information and primarily retains boundary information.

---

# 11. Discussion Questions and Answers

## Q1. Which edge detector was most sensitive to noise?

**Answer:** The Laplacian is expected to be highly sensitive to noise because it is a second-order derivative operator. Noise creates rapid intensity variations, which can be amplified by differentiation. The use of Gaussian smoothing before the LoG operation also demonstrates the importance of controlling high-frequency noise before second-order edge detection.

Sobel and Prewitt can also respond to noise, but smoothing can reduce unwanted responses.

---

## Q2. How did Gaussian and Median filtering affect edge quality?

**Answer:** Gaussian filtering reduced high-frequency variations and helped produce cleaner edge maps in the presence of Gaussian noise. Median filtering was particularly useful for Salt-and-Pepper noise because it suppresses isolated extreme pixels while preserving many boundaries.

Both can improve edge continuity and reduce false edges, although excessive smoothing can remove fine details.

---

## Q3. How did changing Canny thresholds affect edge count and quality?

**Answer:** Increasing the thresholds reduced the number of detected edges:

| Thresholds | Edge Pixels |
|---|---:|
| 30 / 100 | 9,416 |
| 50 / 150 | 4,399 |
| 100 / 200 | 1,441 |

At 50/150, changing the kernel from 3×3 to 5×5 further reduced the count from **4,399 to 961**.

Lower thresholds detect more weak transitions, while higher thresholds produce a sparser representation containing stronger edges.

---

## Q4. Did edge-only images improve or reduce classification accuracy?

**Answer:** They reduced classification accuracy in this experiment.

For ResNet50:

- Raw: **70.00%**
- Filtered: **60.00%**
- Edge: **37.50%**

The edge representation removes information such as color, texture, intensity and internal-region appearance. Those features can be useful for distinguishing classes.

---

## Q5. What information may be lost in edge maps?

**Answer:** Edge maps can lose or greatly reduce:

- Color
- Absolute intensity
- Fine texture
- Surface appearance
- Shading
- Internal regions without strong boundaries

Therefore, an edge map preserves important boundary information but does not preserve the complete visual information contained in the original image.

---

## Q6. What are the advantages of allowing CNNs to learn features automatically?

**Answer:** A CNN can learn different levels of features directly from training images. Early layers can learn edge-like patterns, while deeper layers can combine them into shapes, textures and more complex visual representations.

Using raw images therefore allows the model to learn which combinations of visual features are useful instead of restricting the input to manually extracted edges.

The experiment supports this: ResNet50 achieved **70.00%** accuracy on raw images versus **37.50%** on edge images.

---

## Q7. Which representation produced the most useful classification results?

**Answer:** In this experimental run, the **raw representation produced the highest measured ResNet50 classification metrics**:

| Representation | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Raw | 0.7000 | 0.7440 | 0.7000 | 0.6881 |
| Filtered | 0.6000 | 0.6618 | 0.6000 | 0.5738 |
| Edge | 0.3750 | 0.3486 | 0.3750 | 0.3138 |

This result applies specifically to the reported experimental setup.

---

# 12. Additional Results Analysis

### Raw representation
ResNet50:
- Accuracy: 70.00%
- Precision: 74.40%
- Recall: 70.00%
- F1: 68.81%

### Filtered representation
ResNet50:
- Accuracy: 60.00%
- Precision: 66.18%
- Recall: 60.00%
- F1: 57.38%

### Edge representation
ResNet50:
- Accuracy: 37.50%
- Precision: 34.86%
- Recall: 37.50%
- F1: 31.38%

Classical edge accuracies:
- Linear SVM: 36.25%
- Random Forest: 38.75%
- KNN: 30.00%

---

# 13. Training and Inference Time

## ResNet50

| Representation | Training Time (s) | Inference Time (ms/image) |
|---|---:|---:|
| Raw | 255.064159 | 13.381857 |
| Filtered | 217.207354 | 11.609167 |
| Edge | 309.667261 | 11.919403 |

## Classical Models — Training Time

| Model | Raw | Filtered | Edge |
|---|---:|---:|---:|
| Linear SVM | 0.920091 | 1.285313 | 3.088004 |
| Random Forest | 8.561306 | 7.591070 | 8.963535 |
| KNN | 0.001900 | 0.001961 | 0.002229 |

## Classical Models — Inference Time

| Model | Raw | Filtered | Edge |
|---|---:|---:|---:|
| Linear SVM | 0.574955 | 0.723833 | 1.068106 |
| Random Forest | 0.676024 | 0.810254 | 0.702178 |
| KNN | 1.515150 | 0.829172 | 0.790870 |

---

# 14. Viva Questions and Answers

### 1. What is an edge in an image?
An edge is a location where there is a significant change in image intensity, often corresponding to a boundary between regions or objects.

### 2. Difference between first-order and second-order edge detection?
First-order methods use the first derivative and detect intensity gradients. Sobel and Prewitt are examples. Second-order methods use the second derivative; Laplacian is an example and is more sensitive to rapid intensity variations.

### 3. Difference between Sobel Gx and Gy?
Gx detects intensity changes in the x-direction, while Gy detects intensity changes in the y-direction.

### 4. Why is Laplacian more sensitive to noise?
It uses a second-order derivative, which strongly responds to rapid intensity changes caused by noise.

### 5. Purpose of Gaussian smoothing before edge detection?
It reduces high-frequency variations and noise before edge detection, helping reduce false responses.

### 6. Main advantage of Canny?
Canny combines smoothing, gradient calculation, non-maximum suppression, double thresholding and hysteresis to produce thin and relatively well-connected edges.

### 7. What are Canny's low and high thresholds?
They define the double-threshold stage. Strong responses above the high threshold are strong edges; weaker responses between the thresholds can be retained when connected to strong edges.

### 8. Difference between Gaussian and Salt-and-Pepper noise?
Gaussian noise produces random intensity variations, while Salt-and-Pepper noise produces isolated extreme bright and dark pixels.

### 9. Why is Median filtering useful for Salt-and-Pepper noise?
It replaces a pixel using the median of its neighborhood, making it effective for isolated extreme values while preserving many boundaries.

### 10. Why can edge detection reduce classification performance?
It removes information such as color, texture and intensity that may be useful for classification.

### 11. Can a CNN learn edge features automatically?
Yes. Early CNN layers can learn edge-like and gradient-like filters automatically from training data.

### 12. Why might raw images perform better than edge-only images?
Raw images retain much more information, including color, intensity, texture, shape and spatial appearance. A CNN can learn which of these features are useful.

---

# 15. Overall Discussion

The experiment demonstrates that edge detection is useful for representing object or region boundaries, but an edge-only representation is not necessarily sufficient for image classification.

Canny parameter analysis showed that threshold selection strongly affects edge density. The measured edge count changed from **9,416** at 30/100 to **4,399** at 50/150 and **1,441** at 100/200. Increasing the kernel size from 3×3 to 5×5 at 50/150 reduced the count to **961**.

Classification results showed:

- Raw ResNet50: **70.00%**
- Filtered ResNet50: **60.00%**
- Edge ResNet50: **37.50%**

The classical classifiers also showed lower accuracy on edge representations.

This illustrates the trade-off between handcrafted image representations and learned features. Edge detection explicitly represents boundaries, but it removes information that may be discriminative for classification.

---

# 16. Conclusion

This laboratory implemented Sobel, Prewitt, Laplacian, LoG and Canny edge detection methods and examined their behavior under noise and preprocessing.

Gaussian filtering helped reduce high-frequency variations, while Median filtering was useful for Salt-and-Pepper noise. Canny threshold and kernel changes substantially affected edge density.

The selected Lab 03 configuration was:

**Canny — Low = 50, High = 150, Kernel = 3×3**

The classification experiment compared raw, Gaussian-filtered and Canny edge representations using ResNet50, Linear SVM, Random Forest and KNN.

For the reported experiment, ResNet50 achieved:

**Raw:** 70.00% accuracy, 74.40% precision, 70.00% recall, 68.81% F1

**Filtered:** 60.00% accuracy, 66.18% precision, 60.00% recall, 57.38% F1

**Edge:** 37.50% accuracy, 34.86% precision, 37.50% recall, 31.38% F1

The results show that, for this dataset and experimental setup, retaining the original image information produced higher classification metrics than using the filtered or edge-only representations.

---

# 17. Learning Outcomes

After completing this laboratory, the following were demonstrated:

- Implementation of common edge detection techniques.
- Analysis of noise effects on edge detection.
- Use of Gaussian and Median preprocessing.
- Investigation of Canny thresholds and kernel size.
- Preparation of edge-based image representations.
- Classification using raw, filtered and edge images.
- Comparison of CNN and classical classifier performance.
- Evaluation using accuracy, precision, recall and F1-score.
- Measurement of training and inference time.
- Interpretation of handcrafted edge features versus learned CNN features.

---

# 18. Reproducibility Summary

| Item | Setting |
|---|---|
| Device | CUDA |
| Number of classes | 5 |
| Classes | Actinic keratosis; Basal cell carcinoma; Melanoma; Nevus; Pigmented benign keratosis |
| Training split | 1,572 |
| Validation split | 175 |
| Official test | 80 |
| Lab 02 filter | Gaussian |
| Lab 03 edge detector | Canny |
| Canny low threshold | 50 |
| Canny high threshold | 150 |
| Canny kernel | 3×3 |
| CNN | ResNet50 |
| Classical models | Linear SVM, Random Forest, KNN |
| Representations | Raw, Filtered, Edge |

---

## Note on Reported Values

All numerical values in this document are taken from the supplied execution output. No numerical confusion-matrix values are reported because they were not included in the supplied results. Qualitative observations in Table 1 are presented as observations of the behavior being evaluated, not as fabricated numerical measurements.
