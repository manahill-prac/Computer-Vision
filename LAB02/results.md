# Experimental Results — Lab 02

## Effect of Image Filtering on Skin-Lesion Classification

---

## 1. Objective

The objective of this experiment is to investigate how different **spatial-domain image-processing filters** affect the performance of pretrained deep-learning models for skin-lesion classification.

The experiment compares a baseline using the original images with five filtering conditions:

* **Average / Mean filter**
* **Gaussian filter**
* **Median filter**
* **Sharpening filter**
* **Sobel edge filter**

All model/filter combinations use the same dataset split, preprocessing, training procedure, and evaluation procedure. This ensures that the observed differences can primarily be attributed to the applied image-filtering operation.

A total of **18 experiments** were performed:

> **3 pretrained models × 6 image conditions = 18 runs**

---

# 2. Dataset

## 2.1 Dataset Used in the Notebook

The notebook continues the same **5-class ISIC skin-lesion subset used in Lab Activity 1** rather than downloading the original HAM10000 release separately.

> **Important:** The lab task mentions HAM10000, while the executed notebook uses the previously used 5-class ISIC subset. This distinction should be reported transparently rather than presenting the executed dataset as a separate original HAM10000 download.

## 2.2 Selected Classes

The experiment uses the following five classes:

1. Actinic keratosis
2. Basal cell carcinoma
3. Melanoma
4. Nevus
5. Pigmented benign keratosis

**Number of selected classes:** 5

## 2.3 Class Distribution

| Class                      |    Images |
| -------------------------- | --------: |
| Actinic keratosis          |       114 |
| Basal cell carcinoma       |       376 |
| Melanoma                   |       438 |
| Nevus                      |       357 |
| Pigmented benign keratosis |       462 |
| **Total**                  | **1,747** |

A fixed **stratified 90/10 train-validation split** was created using `random_state=42` and reused for every filter condition.

This resulted in approximately:

* **Training subset:** 1,572 images
* **Validation subset:** 175 images
* **Test subset:** Official `Test` directory, kept separate for final evaluation

Keeping the same split across all experiments provides a consistent basis for comparing the effect of different filters.

---

# 3. Three Best Pretrained Models

The three models used in Lab 02 were selected from the Lab Activity 1 results.

| Rank | Model         | Lab 1 Accuracy |
| ---: | ------------- | -------------: |
|    1 | **ResNet50**  |     **70.00%** |
|    2 | **VGG19**     |     **65.00%** |
|    3 | **ResNet101** |     **63.75%** |

The notebook selected **ResNet101 instead of VGG16** as the third model because both achieved 63.75% accuracy, while ResNet101 had the higher F1-score and AUC in the Lab 1 comparison.

Therefore, the three models evaluated in Lab 02 were:

* **ResNet50**
* **VGG19**
* **ResNet101**

---

# 4. Experimental Setup

## 4.1 Image Preprocessing

| Setting       | Value                  |
| ------------- | ---------------------- |
| Image size    | **224 × 224**          |
| Batch size    | **32**                 |
| Random seed   | **42**                 |
| Hardware      | **CUDA/GPU**           |
| Normalization | ImageNet normalization |

ImageNet normalization:

```text
Mean = [0.485, 0.456, 0.406]
Std  = [0.229, 0.224, 0.225]
```

---

## 4.2 Training Augmentation

The same training augmentation pipeline was retained for all experiments:

* Random horizontal flip with probability `0.5`
* Random vertical flip with probability `0.5`
* Random affine transformation:

  * Translation = `(0.2, 0.2)`
  * Shear = `10°`
  * Scale = `(0.8, 1.2)`

Using the same augmentation pipeline across all experiments helps isolate the effect of the filtering operation.

---

## 4.3 Training Strategy

Each pretrained model used a two-phase transfer-learning procedure.

### Phase 1 — Classification-Head Training

* Backbone frozen
* Classification head trained
* Epochs = **3**
* Optimizer = **Adam**
* Learning rate = `1e-3`

### Phase 2 — Fine-Tuning

* Entire network unfrozen
* Maximum epochs = **10**
* Optimizer = **Adam**
* Learning rate = `1e-4`
* Scheduler = **ReduceLROnPlateau**
* Early-stopping patience = **3**
* Best validation-loss weights restored

### Loss Function

**CrossEntropyLoss**

---

# 5. Image Filters

The following six image conditions were evaluated for each pretrained model.

| Condition      | Implementation                                              |
| -------------- | ----------------------------------------------------------- |
| **No Filter**  | Original RGB image                                          |
| **Average**    | 5 × 5 mean/box blur                                         |
| **Gaussian**   | 5 × 5 Gaussian blur                                         |
| **Median**     | 5 × 5 median filter                                         |
| **Sharpening** | 3 × 3 sharpening kernel                                     |
| **Sobel**      | Grayscale Sobel gradient magnitude replicated to 3 channels |

Therefore:

> **3 models × 6 conditions = 18 experimental runs**

---

# 6. Evaluation Metrics

The executed notebook evaluated each experiment using:

* **Accuracy**
* **Weighted Precision**
* **Weighted Recall**
* **Weighted F1-score**
* **Macro-F1**
* **Balanced Accuracy**
* **Macro one-vs-rest AUC**

These metrics provide complementary information. Accuracy measures overall correct classification, while Macro-F1 and Balanced Accuracy provide additional information about performance across classes. AUC evaluates the model's class-ranking performance.

---

# 7. Experimental Results

## 7.1 Complete Comparison of All 18 Experiments

| Model         | Filter        | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | Macro-F1 (%) | Balanced Accuracy (%) |   AUC (%) |
| ------------- | ------------- | -----------: | ------------: | ---------: | -----------: | -----------: | --------------------: | --------: |
| **ResNet50**  | **No Filter** |    **70.00** |         73.30 |  **70.00** |    **68.50** |    **68.50** |             **70.00** |     90.72 |
| ResNet50      | Average       |        61.25 |         66.51 |      61.25 |        59.40 |        59.40 |                 61.25 |     87.34 |
| ResNet50      | Gaussian      |        63.75 |         67.47 |      63.75 |        58.16 |        58.16 |                 63.75 |     90.66 |
| ResNet50      | Median        |        61.25 |     **75.79** |      61.25 |        56.75 |        56.75 |                 61.25 | **91.13** |
| ResNet50      | Sharpening    |        57.50 |         48.83 |      57.50 |        50.89 |        50.89 |                 57.50 |     89.45 |
| ResNet50      | Sobel         |        45.00 |         41.07 |      45.00 |        42.09 |        42.09 |                 45.00 |     81.66 |
| **VGG19**     | **No Filter** |    **66.25** |         72.84 |  **66.25** |    **63.47** |    **63.47** |             **66.25** | **93.05** |
| VGG19         | Average       |        62.50 |         62.00 |      62.50 |        56.94 |        56.94 |                 62.50 |     92.38 |
| VGG19         | Gaussian      |        62.50 |         65.42 |      62.50 |        57.18 |        57.18 |                 62.50 |     92.36 |
| VGG19         | Median        |        66.25 |         69.16 |      66.25 |        62.13 |        62.13 |                 66.25 |     90.90 |
| VGG19         | Sharpening    |        53.75 |         63.88 |      53.75 |        50.59 |        50.59 |                 53.75 |     86.56 |
| VGG19         | Sobel         |        52.50 |         62.88 |      52.50 |        49.98 |        49.98 |                 52.50 |     84.41 |
| **ResNet101** | **No Filter** |    **63.75** |         70.40 |  **63.75** |        60.54 |        60.54 |             **63.75** |     90.04 |
| ResNet101     | Average       |        58.75 |         64.03 |      58.75 |        56.67 |        56.67 |                 58.75 |     87.77 |
| ResNet101     | Gaussian      |        52.50 |     **72.34** |      52.50 |        45.27 |        45.27 |                 52.50 |     86.09 |
| ResNet101     | Median        |        60.00 |         67.39 |      60.00 |        55.06 |        55.06 |                 60.00 |     90.20 |
| ResNet101     | Sharpening    |        63.75 |     **72.67** |      63.75 |    **60.57** |    **60.57** |                 63.75 | **91.07** |
| ResNet101     | Sobel         |        42.50 |         52.47 |      42.50 |        41.13 |        41.13 |                 42.50 |     76.80 |

> **All values are percentages.**

### Main observations

* ResNet50 achieved the highest overall accuracy: **70.00%**.
* VGG19 achieved the highest AUC: **93.05%**.
* ResNet50 + Median achieved the highest weighted Precision: **75.79%**.
* ResNet50 + No Filter achieved the highest weighted Recall: **70.00%**.
* ResNet50 + No Filter achieved the highest F1-Score and Macro-F1: **68.50%**.
* Sobel produced the lowest accuracy for all three architectures.

---

# 8. Change in Accuracy Relative to the Unfiltered Baseline

The following table reports the change in accuracy relative to each model's own **No Filter** baseline.

| Filter     |      ResNet50 |       VGG19 |   ResNet101 |
| ---------- | ------------: | ----------: | ----------: |
| Average    |      −8.75 pp |    −3.75 pp |    −5.00 pp |
| Gaussian   |      −6.25 pp |    −3.75 pp |   −11.25 pp |
| Median     |      −8.75 pp | **0.00 pp** |    −3.75 pp |
| Sharpening |     −12.50 pp |   −12.50 pp | **0.00 pp** |
| Sobel      | **−25.00 pp** |   −13.75 pp |   −21.25 pp |

> **pp = percentage points**

The largest accuracy change occurred with the **Sobel filter** for every model:

| Model     | No Filter |  Sobel |        Change |
| --------- | --------: | -----: | ------------: |
| ResNet50  |    70.00% | 45.00% | **−25.00 pp** |
| VGG19     |    66.25% | 52.50% | **−13.75 pp** |
| ResNet101 |    63.75% | 42.50% | **−21.25 pp** |

---

# 9. Model-by-Model Analysis

## 9.1 ResNet50

The unfiltered ResNet50 model achieved a baseline accuracy of **70.00%**.

| Condition  |   Accuracy |        Change |
| ---------- | ---------: | ------------: |
| No Filter  | **70.00%** |             — |
| Average    |     61.25% |      −8.75 pp |
| Gaussian   |     63.75% |      −6.25 pp |
| Median     |     61.25% |      −8.75 pp |
| Sharpening |     57.50% |     −12.50 pp |
| Sobel      |     45.00% | **−25.00 pp** |

Every tested filter reduced ResNet50 accuracy.

The largest degradation occurred with Sobel filtering. Macro-F1 decreased from **68.50%** to **42.09%**, while Balanced Accuracy decreased from **70.00%** to **45.00%**.

An important observation is that Median filtering produced the highest weighted Precision for ResNet50 at **75.79%**, despite lower Accuracy, Recall, F1-Score, Macro-F1, and Balanced Accuracy than the unfiltered model.

This demonstrates that Precision alone does not provide a complete picture of overall classification performance.

---

## 9.2 VGG19

The unfiltered VGG19 model achieved a baseline accuracy of **66.25%**.

| Condition  |   Accuracy |      Change |
| ---------- | ---------: | ----------: |
| No Filter  | **66.25%** |           — |
| Average    |     62.50% |    −3.75 pp |
| Gaussian   |     62.50% |    −3.75 pp |
| Median     | **66.25%** | **0.00 pp** |
| Sharpening |     53.75% |   −12.50 pp |
| Sobel      |     52.50% |   −13.75 pp |

Median filtering preserved the baseline accuracy of **66.25%** and the same Balanced Accuracy.

However, its F1-Score and AUC were lower than those of the unfiltered condition.

Sobel filtering produced the largest reduction for VGG19, lowering accuracy by **13.75 percentage points**.

---

## 9.3 ResNet101

The unfiltered ResNet101 model achieved a baseline accuracy of **63.75%**.

| Condition  |   Accuracy |        Change |
| ---------- | ---------: | ------------: |
| No Filter  | **63.75%** |             — |
| Average    |     58.75% |      −5.00 pp |
| Gaussian   |     52.50% |     −11.25 pp |
| Median     |     60.00% |      −3.75 pp |
| Sharpening | **63.75%** |   **0.00 pp** |
| Sobel      |     42.50% | **−21.25 pp** |

Sharpening preserved the same accuracy as the unfiltered baseline.

However, other metrics were slightly different. Sharpening increased weighted Precision from **70.40% to 72.67%**, F1-Score from **60.54% to 60.57%**, Macro-F1 from **60.54% to 60.57%**, and AUC from **90.04% to 91.07%**.

Sobel filtering caused the largest degradation, reducing accuracy to **42.50%**.

---

# 10. Answers to the Required Questions

## Q1. Which three pretrained models performed best in Lab Activity 1?

The three models selected from Lab Activity 1 were:

| Rank | Model         | Lab 1 Accuracy |
| ---: | ------------- | -------------: |
|    1 | **ResNet50**  |     **70.00%** |
|    2 | **VGG19**     |     **65.00%** |
|    3 | **ResNet101** |     **63.75%** |

ResNet101 was selected over VGG16 because both achieved **63.75% accuracy**, while ResNet101 had the higher F1-score and AUC in the Lab 1 comparison.

Therefore, the three models evaluated in Lab 02 were **ResNet50, VGG19, and ResNet101**.

---

## Q2. How does filtering affect each of the three models?

Filtering generally reduced classification performance compared with the original-image baseline.

### ResNet50

All five filters reduced accuracy. The reduction ranged from:

* **Gaussian:** −6.25 pp
* **Sobel:** −25.00 pp

### VGG19

Median filtering preserved the baseline accuracy at **66.25%**. Average and Gaussian filtering produced smaller reductions, while Sharpening and Sobel produced larger reductions.

### ResNet101

Sharpening preserved the baseline accuracy at **63.75%**. Average, Gaussian, and Median filtering reduced accuracy, while Sobel produced the largest reduction.

Therefore, the models did not respond identically to every filter, although the overall trend was that filtering was usually detrimental under this experimental setup.

---

## Q3. Which filter produces the greatest change compared with the unfiltered baseline?

The **Sobel edge filter** produced the greatest accuracy change for all three models.

| Model     | Accuracy Change with Sobel |
| --------- | -------------------------: |
| ResNet50  |              **−25.00 pp** |
| VGG19     |              **−13.75 pp** |
| ResNet101 |              **−21.25 pp** |

It also produced the lowest accuracy for each model:

| Model     | Sobel Accuracy |
| --------- | -------------: |
| ResNet50  |     **45.00%** |
| VGG19     |     **52.50%** |
| ResNet101 |     **42.50%** |

---

## Q4. Does the effect of a filter remain consistent across all three models?

No.

The general direction was similar for many filters, but the **magnitude of the effect differed between architectures**.

For example, Median filtering produced:

| Model     | Accuracy Change |
| --------- | --------------: |
| ResNet50  |        −8.75 pp |
| VGG19     |     **0.00 pp** |
| ResNet101 |        −3.75 pp |

Similarly, Sharpening produced:

| Model     | Accuracy Change |
| --------- | --------------: |
| ResNet50  |       −12.50 pp |
| VGG19     |       −12.50 pp |
| ResNet101 |     **0.00 pp** |

Therefore, filter sensitivity is **model-dependent**.

---

## Q5. Does filtering improve or decrease Macro-F1 and Balanced Accuracy?

In this experiment, filtering generally **decreased both Macro-F1 and Balanced Accuracy** compared with the unfiltered baseline.

### Baseline vs. Sobel

| Model     | Baseline Macro-F1 | Sobel Macro-F1 | Baseline Balanced Accuracy | Sobel Balanced Accuracy |
| --------- | ----------------: | -------------: | -------------------------: | ----------------------: |
| ResNet50  |            68.50% |         42.09% |                     70.00% |                  45.00% |
| VGG19     |            63.47% |         49.98% |                     66.25% |                  52.50% |
| ResNet101 |            60.54% |         41.13% |                     63.75% |                  42.50% |

There were some preservation cases:

* **VGG19 + Median:** Balanced Accuracy remained at 66.25%.
* **ResNet101 + Sharpening:** Balanced Accuracy remained at 63.75%.

However, these conditions did not demonstrate a general improvement over the corresponding unfiltered baselines.

---

## Q6. Which lesion classes are most affected by filtering?

The per-class results show that filtering can affect individual lesion classes differently.

A particularly strong example is **ResNet101 + Sobel**:

| Class                      |  Precision |     Recall |   F1-Score |
| -------------------------- | ---------: | ---------: | ---------: |
| Actinic keratosis          |     66.67% | **12.50%** | **21.05%** |
| Basal cell carcinoma       | **90.91%** |     62.50% | **74.07%** |
| Melanoma                   |     37.50% | **18.75%** | **25.00%** |
| Nevus                      |     40.00% |     62.50% |     48.78% |
| Pigmented benign keratosis |     27.27% |     56.25% |     36.73% |

In this particular run, **actinic keratosis** and **melanoma** showed especially low recall and F1-Score.

These observations describe this specific dataset/model/filter experiment and should not be interpreted as universal properties of these medical classes.

---

## Q7. Why might smoothing remove useful lesion texture or morphological information?

Average, Gaussian, and Median filters reduce local image variations to varying degrees.

Skin-lesion classification can depend on visual characteristics such as:

* Fine texture
* Pigmentation patterns
* Lesion boundaries
* Irregular shapes
* Local color transitions
* Small morphological structures

Smoothing can suppress some of these details together with unwanted noise.

If the removed information is useful for distinguishing lesion classes, the CNN receives a less informative representation, which can reduce classification performance.

The experimental results support this interpretation because the smoothing filters generally reduced Macro-F1 and Balanced Accuracy relative to the unfiltered baseline.

---

## Q8. Why might sharpening or edge detection help or hurt classification?

### Sharpening

Sharpening increases local contrast and emphasizes boundaries.

It may help when:

* Lesion boundaries contain useful information.
* Fine structures are important.
* The original image contains mild blur.

However, sharpening can also:

* Amplify noise
* Create artificial high-frequency patterns
* Distort natural texture
* Change the image distribution presented to the pretrained network

The experimental results demonstrate this model-dependent behavior. Sharpening reduced ResNet50 and VGG19 accuracy but preserved ResNet101 accuracy at **63.75%**.

### Sobel Edge Detection

Sobel filtering emphasizes image gradients and produces an edge-oriented representation.

In this experiment, the image was converted to grayscale before computing the Sobel gradient magnitude, and the resulting representation was replicated to three channels.

This transformation removes much of the original RGB color and texture information.

That can be harmful when the CNN relies on:

* Color
* Pigmentation
* Texture
* Region appearance
* Rich RGB information

The substantial accuracy reductions observed for all three models are consistent with this explanation.

---

## Q9. What is the difference between convolution and correlation?

Both convolution and correlation involve sliding a kernel across an image and calculating local weighted sums.

The key difference is **kernel flipping**.

### Correlation

The kernel is applied without flipping:

$$
g(x,y)=\sum_m\sum_n f(x+m,y+n)h(m,n)
$$

### Convolution

The kernel is flipped horizontally and vertically:

$$
g(x,y)=\sum_m\sum_n f(x-m,y-n)h(m,n)
$$

Therefore:

| Operation       | Kernel                    |
| --------------- | ------------------------- |
| **Correlation** | Kernel is not flipped     |
| **Convolution** | Kernel is flipped by 180° |

For symmetric kernels, such as many Gaussian and mean kernels, convolution and correlation produce the same result because flipping the kernel does not change it.

In many deep-learning libraries, the operation commonly called a **convolution layer** is mathematically implemented as cross-correlation rather than strict mathematical convolution.

---

## Q10. Based on the results, explain the relationship between classical image processing and deep-learning-based feature extraction.

Classical image processing and deep learning can be viewed as two different stages of feature transformation.

Classical filters apply predefined transformations before the image reaches the CNN:

* **Mean / Gaussian / Median:** smoothing
* **Sharpening:** emphasizing local high-frequency information
* **Sobel:** emphasizing image gradients and edges

A CNN, in contrast, learns task-specific feature representations from the training data.

The results show that preprocessing is not automatically beneficial simply because it emphasizes a particular visual property. The pretrained CNNs were already capable of learning useful representations from the original RGB images. Removing or altering information before the CNN could therefore make classification more difficult.

The strongest example is the Sobel condition. Converting the original RGB lesion image into an edge-oriented representation resulted in substantial performance reductions across all three models.

At the same time, the response was model-dependent. ResNet101 maintained its baseline accuracy after sharpening, while VGG19 maintained its baseline accuracy after Median filtering.

Therefore, classical image processing should be treated as an **experimentally validated preprocessing choice** rather than an automatic improvement over the original images.

---

# 11. Key Findings

The main findings from the 18 experiments are:

1. **ResNet50, VGG19, and ResNet101** were the three pretrained models evaluated.
2. ResNet50 achieved the highest unfiltered accuracy at **70.00%**.
3. VGG19 achieved the highest AUC at **93.05%** under the unfiltered condition.
4. ResNet50 achieved the highest weighted Precision at **75.79%** with Median filtering.
5. The unfiltered ResNet50 condition achieved the highest F1-Score and Macro-F1 at **68.50%**.
6. **Sobel filtering produced the largest accuracy reduction for all three models.**
7. ResNet50 experienced the largest reduction with Sobel: **−25.00 pp**.
8. VGG19 maintained its baseline accuracy after Median filtering.
9. ResNet101 maintained its baseline accuracy after Sharpening.
10. Macro-F1 and Balanced Accuracy generally decreased after filtering.
11. Per-class results showed that the effect of filtering was not uniform across lesion classes.
12. The results demonstrate that classical image preprocessing interacts with learned CNN representations in a **model-dependent manner**.

---

# 12. Best Results at a Glance

| Metric                             | Model / Condition          |         Score |
| ---------------------------------- | -------------------------- | ------------: |
| Highest Accuracy                   | **ResNet50 — No Filter**   |    **70.00%** |
| Highest Weighted Precision         | **ResNet50 — Median**      |    **75.79%** |
| Highest Recall                     | **ResNet50 — No Filter**   |    **70.00%** |
| Highest F1-Score                   | **ResNet50 — No Filter**   |    **68.50%** |
| Highest Macro-F1                   | **ResNet50 — No Filter**   |    **68.50%** |
| Highest Balanced Accuracy          | **ResNet50 — No Filter**   |    **70.00%** |
| Highest AUC                        | **VGG19 — No Filter**      |    **93.05%** |
| Largest Accuracy Drop              | **ResNet50 — Sobel**       | **−25.00 pp** |
| Lowest Accuracy                    | **ResNet101 — Sobel**      |    **42.50%** |
| Accuracy Preserved After Filtering | **VGG19 — Median**         |    **66.25%** |
| Accuracy Preserved After Filtering | **ResNet101 — Sharpening** |    **63.75%** |

---

# 13. Best Results Within Each Model

| Model         | Best Accuracy Condition(s) |   Accuracy |            Macro-F1 | Balanced Accuracy |                 AUC |
| ------------- | -------------------------- | ---------: | ------------------: | ----------------: | ------------------: |
| **ResNet50**  | No Filter                  | **70.00%** |          **68.50%** |        **70.00%** |              90.72% |
| **VGG19**     | No Filter / Median         | **66.25%** |     63.47% / 62.13% |        **66.25%** |     93.05% / 90.90% |
| **ResNet101** | No Filter / Sharpening     | **63.75%** | 60.54% / **60.57%** |        **63.75%** | 90.04% / **91.07%** |

Where multiple conditions achieved the same accuracy, they are reported together rather than treating the tie as a unique result.

---

# 14. Required Analysis Checklist

The executed notebook contains the following analyses required for the experiment:

* [x] Selected-class distribution
* [x] Original and filtered image visual comparison
* [x] Confusion matrices
* [x] Training/validation accuracy curves
* [x] Training/validation loss curves
* [x] Per-class Precision, Recall, and F1-Score
* [x] Macro-F1
* [x] Balanced Accuracy
* [x] AUC/ROC metric
* [x] Comparative accuracy analysis
* [x] Change from each model's unfiltered baseline
* [x] Analysis of the effect of each filter
* [x] Answers to all required questions

The notebook also saves the numerical results as:

```text
results/lab02_filter_comparison.csv
results/lab02_per_class_metrics.csv
```

These CSV files contain the experimental results generated during execution.

---

# 15. Conclusion

This experiment evaluated three pretrained CNN architectures—**ResNet50, VGG19, and ResNet101**—under an unfiltered baseline and five spatial-domain image-filtering conditions.

The results show that the original RGB images generally provided stronger classification performance than the filtered images. ResNet50 achieved the highest baseline accuracy of **70.00%**, while VGG19 and ResNet101 achieved **66.25%** and **63.75%**, respectively.

Among the tested filters, **Sobel produced the largest performance reduction across all three models**. Accuracy decreased by:

* **25.00 percentage points** for ResNet50
* **13.75 percentage points** for VGG19
* **21.25 percentage points** for ResNet101

The results also demonstrate that filtering does not affect all architectures identically. **VGG19 maintained its baseline accuracy after Median filtering**, while **ResNet101 maintained its baseline accuracy after Sharpening**.

These observations indicate that the usefulness of a classical image-processing filter depends on the information it preserves or removes and how that information interacts with the learned feature representation of the CNN.

For this dataset and experimental configuration, preserving the **original RGB lesion information** was generally more effective than applying the tested spatial-domain filters before classification.

---

# 16. Reproducibility Notes

| Setting                | Value                                                             |
| ---------------------- | ----------------------------------------------------------------- |
| Dataset                | 5-class ISIC skin-lesion subset                                   |
| Number of classes      | 5                                                                 |
| Image size             | 224 × 224                                                         |
| Batch size             | 32                                                                |
| Random seed            | 42                                                                |
| Train/Validation split | Stratified 90/10                                                  |
| Test evaluation        | Official `Test` directory                                         |
| Models                 | ResNet50, VGG19, ResNet101                                        |
| Image conditions       | No Filter, Average, Gaussian, Median, Sharpening, Sobel           |
| Total experiments      | 18                                                                |
| Optimizer              | Adam                                                              |
| Loss function          | CrossEntropyLoss                                                  |
| Pretrained weights     | ImageNet                                                          |
| Hardware               | CUDA/GPU                                                          |
| Frameworks/Libraries   | PyTorch, torchvision, OpenCV, scikit-learn                        |
| Main metrics           | Accuracy, Precision, Recall, F1, Macro-F1, Balanced Accuracy, AUC |

---


