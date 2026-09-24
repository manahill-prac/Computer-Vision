# Task 02 --- Effect of Image Filtering on Skin-Lesion Classification

## 1. Objective

The objective of this experiment is to investigate how different
spatial-domain image-processing filters affect the performance of
pretrained deep-learning models for skin-lesion classification.

The experiment compares a baseline using original images with five
filtering conditions:

-   Average / Mean filter
-   Gaussian filter
-   Median filter
-   Sharpening filter
-   Sobel edge filter

All model/filter combinations use the same dataset split, preprocessing,
training procedure, and evaluation procedure so that the effect of
filtering can be compared fairly.

------------------------------------------------------------------------

## 2. Dataset

### Dataset used in the notebook

The notebook continues the same 5-class ISIC skin-lesion subset used in
Lab Activity 1 rather than downloading the original HAM10000 release
separately.

**Important:** The lab task mentions HAM10000, while the executed
notebook uses the instructor-approved/previously used ISIC subset. This
distinction should be mentioned in the submission rather than presenting
the dataset as the original HAM10000 dataset.

### Selected classes

1.  Actinic keratosis
2.  Basal cell carcinoma
3.  Melanoma
4.  Nevus
5.  Pigmented benign keratosis

**Number of selected classes:** 5

### Class distribution before the 90/10 split

  Class                             Images
  ---------------------------- -----------
  Actinic keratosis                    114
  Basal cell carcinoma                 376
  Melanoma                             438
  Nevus                                357
  Pigmented benign keratosis           462
  **Total**                      **1,747**

A fixed stratified 90/10 train-validation split was created with
`random_state=42` and reused for every filter condition. This gives
approximately:

-   Training subset: 1,572 images
-   Validation subset: 175 images

The separate `Test` directory was used for final evaluation.

------------------------------------------------------------------------

## 3. Best Three Pretrained Models

The three models were selected from Lab Activity 1 according to the
ranking recorded in the notebook:

  Rank   Model         Lab 1 Accuracy
  ------ ----------- ----------------
  1      ResNet50              70.00%
  2      VGG19                 65.00%
  3      ResNet101             63.75%

The notebook notes that ResNet101 was selected over VGG16, which also
achieved 63.75% accuracy, because ResNet101 had the higher F1-score and
AUC.

Therefore, the three models used in Task 02 are:

-   **Best Model 1: ResNet50**
-   **Best Model 2: VGG19**
-   **Best Model 3: ResNet101**

------------------------------------------------------------------------

## 4. Experimental Setup

### Image preprocessing

-   Image size: **224 × 224**
-   ImageNet normalization:
    -   Mean = `[0.485, 0.456, 0.406]`
    -   Standard deviation = `[0.229, 0.224, 0.225]`
-   Batch size: **32**
-   Random seed: **42**
-   Hardware during execution: **CUDA/GPU**

### Training augmentation

For training images, the same augmentation pipeline was retained across
all experiments:

-   Random horizontal flip, probability 0.5
-   Random vertical flip, probability 0.5
-   Random affine transformation:
    -   Translation = `(0.2, 0.2)`
    -   Shear = `10°`
    -   Scale = `(0.8, 1.2)`

### Training strategy

Each pretrained model used two-phase transfer learning:

**Phase 1 --- Feature extraction**

-   Backbone frozen
-   Classification head trained
-   3 epochs
-   Adam optimizer
-   Learning rate = `1e-3`

**Phase 2 --- Fine-tuning**

-   Entire network unfrozen
-   Up to 10 epochs
-   Adam optimizer
-   Learning rate = `1e-4`
-   ReduceLROnPlateau scheduler
-   Early stopping patience = 3
-   Best validation-loss model restored

Loss function: **CrossEntropyLoss**

### Filters

  Filter       Implementation
  ------------ --------------------------------------------------------------
  No Filter    Original RGB image
  Average      5 × 5 mean/box blur
  Gaussian     5 × 5 Gaussian blur
  Median       5 × 5 median filter
  Sharpening   3 × 3 sharpening kernel
  Sobel        Grayscale Sobel gradient magnitude, replicated to 3 channels

A total of **18 experiments** were performed:

**3 models × 6 conditions = 18 runs**

------------------------------------------------------------------------

# 5. Experimental Results

The following metrics were obtained from the executed notebook.

-   Accuracy
-   Weighted Precision
-   Weighted Recall
-   Weighted F1-score
-   Macro-F1
-   Balanced Accuracy
-   Macro one-vs-rest AUC

## 5.1 Complete comparison table

  ------------------------------------------------------------------------------------------------
  Model       Filter         Accuracy   Precision   Recall       F1   Macro-F1   Balanced      AUC
                                                                                 Accuracy 
  ----------- ------------ ---------- ----------- -------- -------- ---------- ---------- --------
  ResNet50    No Filter         70.00       73.30    70.00    68.50      68.50      70.00    90.72

  ResNet50    Average           61.25       66.51    61.25    59.40      59.40      61.25    87.34

  ResNet50    Gaussian          63.75       67.47    63.75    58.16      58.16      63.75    90.66

  ResNet50    Median            61.25       75.79    61.25    56.75      56.75      61.25    91.13

  ResNet50    Sharpening        57.50       48.83    57.50    50.89      50.89      57.50    89.45

  ResNet50    Sobel             45.00       41.07    45.00    42.09      42.09      45.00    81.66

  VGG19       No Filter         66.25       72.84    66.25    63.47      63.47      66.25    93.05

  VGG19       Average           62.50       62.00    62.50    56.94      56.94      62.50    92.38

  VGG19       Gaussian          62.50       65.42    62.50    57.18      57.18      62.50    92.36

  VGG19       Median            66.25       69.16    66.25    62.13      62.13      66.25    90.90

  VGG19       Sharpening        53.75       63.88    53.75    50.59      50.59      53.75    86.56

  VGG19       Sobel             52.50       62.88    52.50    49.98      49.98      52.50    84.41

  ResNet101   No Filter         63.75       70.40    63.75    60.54      60.54      63.75    90.04

  ResNet101   Average           58.75       64.03    58.75    56.67      56.67      58.75    87.77

  ResNet101   Gaussian          52.50       72.34    52.50    45.27      45.27      52.50    86.09

  ResNet101   Median            60.00       67.39    60.00    55.06      55.06      60.00    90.20

  ResNet101   Sharpening        63.75       72.67    63.75    60.57      60.57      63.75    91.07

  ResNet101   Sobel             42.50       52.47    42.50    41.13      41.13      42.50    76.80
  ------------------------------------------------------------------------------------------------

**All values are percentages.**

------------------------------------------------------------------------

# 6. Change in Accuracy Relative to the Unfiltered Baseline

  Filter          ResNet50       VGG19   ResNet101
  ------------ ----------- ----------- -----------
  Average         -8.75 pp    -3.75 pp    -5.00 pp
  Gaussian        -6.25 pp    -3.75 pp   -11.25 pp
  Median          -8.75 pp     0.00 pp    -3.75 pp
  Sharpening     -12.50 pp   -12.50 pp     0.00 pp
  Sobel          -25.00 pp   -13.75 pp   -21.25 pp

`pp` = percentage points.

The largest accuracy change for all three models occurs with the **Sobel
filter**:

-   ResNet50: **−25.00 percentage points**
-   VGG19: **−13.75 percentage points**
-   ResNet101: **−21.25 percentage points**

------------------------------------------------------------------------

# 7. Model-by-Model Analysis

## 7.1 ResNet50

Baseline accuracy was **70.00%**.

Filtering reduced accuracy under every tested condition:

-   Average: 61.25% → −8.75 pp
-   Gaussian: 63.75% → −6.25 pp
-   Median: 61.25% → −8.75 pp
-   Sharpening: 57.50% → −12.50 pp
-   Sobel: 45.00% → −25.00 pp

The Sobel filter caused the largest degradation. Macro-F1 decreased from
**68.50%** to **42.09%**, while balanced accuracy decreased from
**70.00%** to **45.00%**.

Interestingly, Median filtering produced a higher weighted precision of
75.79%, but its overall accuracy, recall, F1, macro-F1, and balanced
accuracy were lower than the unfiltered baseline. This shows why
precision alone should not be used to judge the overall classification
behavior.

## 7.2 VGG19

Baseline accuracy was **66.25%**.

-   Average: 62.50% → −3.75 pp
-   Gaussian: 62.50% → −3.75 pp
-   Median: 66.25% → 0.00 pp
-   Sharpening: 53.75% → −12.50 pp
-   Sobel: 52.50% → −13.75 pp

Median filtering produced the same accuracy and balanced accuracy as the
baseline, although its F1-score and AUC were lower.

Sobel again caused substantial degradation.

## 7.3 ResNet101

Baseline accuracy was **63.75%**.

-   Average: 58.75% → −5.00 pp
-   Gaussian: 52.50% → −11.25 pp
-   Median: 60.00% → −3.75 pp
-   Sharpening: 63.75% → 0.00 pp
-   Sobel: 42.50% → −21.25 pp

ResNet101 was comparatively unchanged by sharpening in terms of
accuracy: both the baseline and sharpening condition achieved
**63.75%**. However, this did not mean that every metric was identical;
the sharpening condition had slightly higher weighted precision, F1,
macro-F1 and AUC.

------------------------------------------------------------------------

# 8. Answers to the Required Questions

## Q1. Which three pretrained models performed best in Lab Activity 1?

The three models selected from Lab Activity 1 were:

1.  **ResNet50 --- 70.00%**
2.  **VGG19 --- 65.00%**
3.  **ResNet101 --- 63.75%**

ResNet101 was selected over VGG16 because both had 63.75% accuracy,
while ResNet101 had the higher F1-score and AUC according to the Lab 1
comparison recorded in the notebook.

------------------------------------------------------------------------

## Q2. How does filtering affect each of the three models?

Filtering generally reduced classification performance compared with the
original-image baseline.

### ResNet50

All five filters reduced accuracy. The reduction ranged from **−6.25 pp
with Gaussian filtering** to **−25.00 pp with Sobel filtering**.

### VGG19

Median filtering preserved the baseline accuracy at **66.25%**. Average
and Gaussian filtering caused smaller reductions, while sharpening and
Sobel caused larger reductions.

### ResNet101

Sharpening preserved the baseline accuracy at **63.75%**. Average,
Gaussian, and Median filtering reduced accuracy, with Gaussian producing
a larger decrease. Sobel produced the largest reduction.

Therefore, the models did not respond identically to every filter,
although the overall trend was that filtering was usually detrimental.

------------------------------------------------------------------------

## Q3. Which filter produces the greatest change compared with the unfiltered baseline?

The **Sobel edge filter** produced the greatest accuracy change for all
three models.

-   ResNet50: −25.00 pp
-   VGG19: −13.75 pp
-   ResNet101: −21.25 pp

It also produced the lowest accuracy for each model:

-   ResNet50: 45.00%
-   VGG19: 52.50%
-   ResNet101: 42.50%

------------------------------------------------------------------------

## Q4. Does the effect of a filter remain consistent across all three models?

No.

The direction of the effect is broadly similar because most filters
reduce performance, but the magnitude differs across models.

For example:

-   Median filtering reduced ResNet50 accuracy by 8.75 pp.
-   Median filtering had **0.00 pp** change for VGG19.
-   Median filtering reduced ResNet101 accuracy by 3.75 pp.

Similarly, sharpening reduced ResNet50 and VGG19 accuracy by 12.50 pp
but produced no accuracy change for ResNet101.

Therefore, filter sensitivity is **model-dependent**.

------------------------------------------------------------------------

## Q5. Does filtering improve or decrease macro-F1 and balanced accuracy?

In this experiment, filtering generally **decreased both Macro-F1 and
balanced accuracy** compared with the unfiltered baseline.

### ResNet50

Baseline:

-   Macro-F1 = 68.50%
-   Balanced Accuracy = 70.00%

Sobel:

-   Macro-F1 = 42.09%
-   Balanced Accuracy = 45.00%

### VGG19

Baseline:

-   Macro-F1 = 63.47%
-   Balanced Accuracy = 66.25%

Sobel:

-   Macro-F1 = 49.98%
-   Balanced Accuracy = 52.50%

### ResNet101

Baseline:

-   Macro-F1 = 60.54%
-   Balanced Accuracy = 63.75%

Sobel:

-   Macro-F1 = 41.13%
-   Balanced Accuracy = 42.50%

The only notable preservation cases were Median filtering for VGG19 and
Sharpening for ResNet101 in terms of accuracy/balanced accuracy. They
did not produce a general improvement over the baseline.

------------------------------------------------------------------------

## Q6. Which lesion classes are most affected by filtering?

The per-class classification analysis shows that filtering particularly
affects classes whose recognition depends on information that can be
weakened or distorted by smoothing or edge conversion.

The strongest degradation is visible in the edge-filtered conditions,
especially for **melanoma** and **actinic keratosis** in the reported
per-class results. For example, in the ResNet101 + Sobel condition:

  Class                          Precision   Recall       F1
  ---------------------------- ----------- -------- --------
  Actinic keratosis                 66.67%   12.50%   21.05%
  Basal cell carcinoma              90.91%   62.50%   74.07%
  Melanoma                          37.50%   18.75%   25.00%
  Nevus                             40.00%   62.50%   48.78%
  Pigmented benign keratosis        27.27%   56.25%   36.73%

This indicates that Sobel-based conversion can cause substantial
class-specific loss of recall and F1, particularly for actinic keratosis
and melanoma in this run.

The effect should not be interpreted as a universal property of these
diseases; it is a result observed on this experiment's selected dataset
and trained models.

------------------------------------------------------------------------

## Q7. Why might smoothing remove useful lesion texture or morphological information?

Average, Gaussian, and Median filters reduce high-frequency image
information to varying degrees.

Skin-lesion classification can depend on visual characteristics such as:

-   Fine texture
-   Small pigment structures
-   Lesion borders
-   Irregular shapes
-   Local color transitions
-   Small morphological details

Smoothing can suppress some of these details together with noise. If the
removed information is discriminative for a particular lesion class, the
CNN receives a less informative representation and classification
performance can decrease.

This experiment supports that interpretation because the smoothing
filters generally reduced Macro-F1 and balanced accuracy relative to the
unfiltered baseline.

------------------------------------------------------------------------

## Q8. Why might sharpening or edge detection help or hurt classification?

### Sharpening

Sharpening increases local contrast and emphasizes edges. This can help
if lesion boundaries or fine structures are useful classification
features.

However, excessive sharpening can also:

-   Amplify image noise
-   Create artificial high-frequency patterns
-   Distort natural texture
-   Make the image distribution different from what the pretrained model
    originally learned

The results demonstrate both possibilities. Sharpening reduced ResNet50
and VGG19 accuracy but left ResNet101 accuracy unchanged at 63.75%.

### Sobel edge detection

Sobel filtering explicitly converts the image into an edge-magnitude
representation. This removes much of the original color and texture
information.

This can hurt classification when the pretrained CNN depends on:

-   Color
-   Pigmentation
-   Texture
-   Region appearance
-   Rich RGB information

The strong decrease in performance for all three models, particularly
the large accuracy drops for ResNet50 and ResNet101, is consistent with
this explanation.

------------------------------------------------------------------------

## Q9. What is the difference between convolution and correlation?

Both convolution and correlation involve sliding a kernel across an
image and computing a local weighted sum.

The key difference is **kernel flipping**.

### Correlation

The kernel is applied directly without flipping:

\[ g(x,y)=`\sum`{=tex}\_m`\sum`{=tex}\_n f(x+m,y+n)h(m,n) \]

### Convolution

The kernel is flipped horizontally and vertically before the operation:

\[ g(x,y)=`\sum`{=tex}\_m`\sum`{=tex}\_n f(x-m,y-n)h(m,n) \]

Therefore:

-   **Correlation:** kernel is not flipped.
-   **Convolution:** kernel is flipped by 180°.

For symmetric kernels, such as many Gaussian or mean filters,
convolution and correlation give the same result because flipping the
kernel does not change it.

In many deep-learning libraries, the operation commonly called a
convolution layer is mathematically implemented as cross-correlation
rather than strict mathematical convolution.

------------------------------------------------------------------------

## Q10. Based on the results, explain the relationship between classical image processing and deep-learning-based feature extraction.

Classical image processing and deep learning can be viewed as two
different stages of feature transformation.

Classical filters impose a predefined transformation before the CNN sees
the image:

-   Mean/Gaussian/Median filters emphasize smoothing.
-   Sharpening emphasizes local high-frequency information.
-   Sobel emphasizes edges.

A CNN, in contrast, learns task-specific feature representations from
the training data.

The results show that preprocessing is not automatically beneficial
simply because it highlights a particular visual property. The
pretrained networks were already capable of learning useful
representations from the original RGB images. Removing or changing
information before the CNN could therefore make classification harder.

The strongest example is the Sobel condition. It converts the original
RGB lesion image into a grayscale edge representation and substantially
reduces performance across all three models.

At the same time, the response is model-dependent. ResNet101 maintained
its baseline accuracy after sharpening, while VGG19 maintained its
baseline accuracy after median filtering. This indicates that a
classical filter can interact differently with the learned
representation of different architectures.

Overall, the experiment suggests that classical image processing should
be treated as an experimentally validated preprocessing choice rather
than an automatic improvement over raw images.

------------------------------------------------------------------------

# 9. Key Findings

1.  The three evaluated pretrained models were **ResNet50, VGG19, and
    ResNet101**.
2.  The unfiltered baseline produced the strongest overall accuracy for
    ResNet50 and VGG19.
3.  ResNet101 achieved the same accuracy with sharpening as with no
    filtering.
4.  **Sobel filtering produced the largest accuracy reduction for all
    three models.**
5.  Filtering effects were **not identical across architectures**.
6.  Macro-F1 and balanced accuracy generally decreased when filtering
    was applied.
7.  Smoothing can remove useful texture, color transitions, and
    morphological details.
8.  Sharpening can preserve or emphasize useful boundaries but may also
    amplify unwanted patterns.
9.  Sobel filtering discards much of the original color and texture
    information, which is particularly harmful when those features are
    important for classification.
10. The experiment demonstrates that classical image-processing filters
    and learned CNN representations interact in a model-dependent way.

------------------------------------------------------------------------

# 10. Best Results Within Each Model

  ----------------------------------------------------------------------------
  Model       Best             Accuracy     Macro-F1     Balanced          AUC
              Accuracy                                   Accuracy 
              Condition                                           
  ----------- ------------ ------------ ------------ ------------ ------------
  ResNet50    No Filter          70.00%       68.50%       70.00%       90.72%

  VGG19       No Filter /        66.25%     63.47% /       66.25%     93.05% /
              Median                          62.13%                    90.90%

  ResNet101   No Filter /        63.75%     60.54% /       63.75%     90.04% /
              Sharpening                      60.57%                    91.07%
  ----------------------------------------------------------------------------

The table above reports the conditions with the highest accuracy within
each model. Where multiple conditions have equal accuracy, they are
shown together rather than treating the tie as a unique winner.

------------------------------------------------------------------------

# 11. Required Additional Analysis Checklist

The executed notebook contains code/output for the following required
analyses:

-   [x] HAM/ISIC selected class distribution
-   [x] Original and filtered image visual comparison
-   [x] Confusion matrices
-   [x] Training/validation accuracy curves
-   [x] Training/validation loss curves
-   [x] Per-class precision, recall, and F1-score
-   [x] Macro-F1
-   [x] Balanced accuracy
-   [x] AUC/ROC metric
-   [x] Comparative accuracy analysis
-   [x] Change from each model's unfiltered baseline

The notebook also saves:

-   `results/lab02_filter_comparison.csv`
-   `results/lab02_per_class_metrics.csv`

These CSV files contain the numerical experimental results generated by
the notebook.

------------------------------------------------------------------------

# 12. Conclusion

The experiment evaluated three pretrained CNN architectures under an
unfiltered baseline and five spatial-domain filtering conditions.

The results show that the original images generally provided stronger
classification performance than filtered images. The **Sobel edge filter
caused the largest performance reduction across all three models**, with
accuracy decreases of 25.00 percentage points for ResNet50, 13.75
percentage points for VGG19, and 21.25 percentage points for ResNet101.

The results also demonstrate that filtering does not affect all
architectures identically. Median filtering preserved VGG19's baseline
accuracy, while sharpening preserved ResNet101's baseline accuracy.
Therefore, the effect of classical preprocessing depends on the
interaction between the filter, the information retained in the image,
and the learned representation of the CNN.

For this experimental setup, the evidence indicates that preserving the
original RGB lesion information was generally more useful than applying
the tested spatial filters before classification.

------------------------------------------------------------------------

## 13. Reproducibility Notes

-   Random seed: `42`
-   Image size: `224 × 224`
-   Batch size: `32`
-   Number of classes: `5`
-   Models: `ResNet50`, `VGG19`, `ResNet101`
-   Conditions: `No Filter`, `Average`, `Gaussian`, `Median`,
    `Sharpening`, `Sobel`
-   Total runs: `18`
-   Validation split: stratified 10%
-   Evaluation: held-out `Test` directory
-   Pretrained weights: torchvision ImageNet pretrained weights
-   Frameworks/libraries: PyTorch, torchvision, OpenCV, scikit-learn

## 14. Submission Note

Before submitting, keep the notebook and this `results.md` file together
in the Lab 02 GitHub folder. The lab instructions also require a brief
README explaining how to run the experiments.

The dataset discrepancy should be reported transparently: the executed
notebook uses the same 5-class ISIC subset from Lab 1, not a separate
original HAM10000 download.
