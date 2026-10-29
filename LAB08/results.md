# Lab Mid Exam — HOG vs SIFT on Blood-Cell Images

**Name:** Manahil Mirza  |  **Registration no.:** FA23-BAI-011  |  **Course:** Computer Vision Lab

| Personal setting | Rule | My value |
|---|---|---|
| Random seed | last two digits of 011 | **11** |
| HOG cell size | last digit 1 is odd | **16 × 16** |
| SIFT grid size | last digit 1 is in 0–4 | **4 × 4 (16 keypoints)** |

**Data source:** official `medmnist` BloodMNIST, native size **64 × 64**. Classes: eosinophil (1), monocyte (5), neutrophil (6), platelet (7).
**Rule followed:** C was chosen on the validation set only. The test set was used only in Q4.
The notebook is `FA23_BAI_011_ManahilMirza_LabMid.ipynb` (all outputs visible); a PDF copy is submitted as well.

---

## Q1. Data setup

The subset was chosen randomly with `np.random.RandomState(11)`, without replacement.

| Split | X shape | y shape | eosinophil | monocyte | neutrophil | platelet |
|---|---|---|---|---|---|---|
| train | (400, 64, 64, 3) | (400,) | 100 | 100 | 100 | 100 |
| validation | (120, 64, 64, 3) | (120,) | 30 | 30 | 30 | 30 |
| test | (120, 64, 64, 3) | (120,) | 30 | 30 | 30 | 30 |

After preparation (grayscale, resize to 128 × 128) every split has shape `(N, 128, 128)` and dtype `uint8`
(train `(400, 128, 128)`, validation `(120, 128, 128)`, test `(120, 128, 128)`).

**4 example images per class:** shown as a 4 × 4 grid in the notebook (Q1 plot cell). The original colour images and the prepared 128 × 128 grayscale versions are also plotted there.

---

## Q2. HOG + SVM

### (a) Feature length

HOG settings: 9 orientations, **16 × 16** pixels per cell, 2 × 2 cells per block, L2-Hys block normalisation.

**Feature length printed by the code: 1764** (matrix shapes: train (400, 1764), validation (120, 1764)).

**Hand calculation:**
1. Cells per side = 128 / 16 = **8**, so 8 × 8 = 64 cells.
2. Blocks slide one cell at a time: blocks per side = 8 − 2 + 1 = **7**, so 7 × 7 = **49 blocks**.
3. Each block has 2 × 2 = 4 cells, and each cell has 9 bins, so 4 × 9 = **36 numbers per block**.
4. Feature length = 49 × 36 = **1764**. The code asserts that this equals the printed length.

### (b) HOG picture for one image per class

The HOG pictures (one image from each of the four classes) are plotted in the notebook, Q2 (b) plot cell.

### (c) LinearSVC — C chosen on the validation set

| C | validation accuracy | validation macro-F1 |
|---|---|---|
| **0.1** (chosen) | **0.8083** | **0.8056** |
| 1 | 0.7667 | 0.7632 |
| 10 | 0.7750 | 0.7737 |

**Chosen C = 0.1. Validation accuracy = 80.83 %, validation macro-F1 = 80.56 %.**

Confusion matrix (validation set, rows = true class):

| | pred eosinophil | pred monocyte | pred neutrophil | pred platelet |
|---|---|---|---|---|
| **true eosinophil** | 19 | 5 | 6 | 0 |
| **true monocyte** | 2 | 27 | 1 | 0 |
| **true neutrophil** | 5 | 2 | 22 | 1 |
| **true platelet** | 0 | 0 | 1 | 29 |

---

## Q3. SIFT + SVM

### (a) Normal SIFT detection on the 400 training images

`cv2.SIFT_create().detectAndCompute(image, None)`:

| Keypoints per training image | Value |
|---|---|
| average | **70.58** |
| minimum | **7** |
| maximum | **181** |
| images with zero keypoints | **0** |

The keypoints drawn on training image 0 are shown in the notebook, Q3 (a) plot cell.

**Why is a different number of keypoints per image a problem for an SVM?**
An SVM needs every image to be one feature vector of the same length, but normal SIFT returned anywhere from 7 to 181 descriptors of 128 numbers per image, so the descriptors cannot be stacked into one fixed-size feature matrix.

### (b) Fix: SIFT descriptors on a fixed 4 × 4 grid

Keypoints are placed on a regular 4 × 4 grid over the 128 × 128 image (centres at 16, 48, 80, 112 in x and y), keypoint size 32. The descriptor is computed at every grid point with `sift.compute` and the 16 descriptors are joined into one vector.
Every image now has exactly the same keypoints, so the feature vector has a fixed length.

The 4 × 4 grid of keypoints drawn on one image is shown in the notebook, Q3 (b) plot cell.

**Feature length printed by the code: 2048** (matrix shapes: train (400, 2048), validation (120, 2048)).

**Hand calculation:**
1. Grid points = 4 × 4 = **16 keypoints**.
2. Each SIFT descriptor has **128** numbers (4 × 4 spatial cells × 8 orientation bins).
3. Feature length = 16 × 128 = **2048**. The code asserts that this equals the printed length.

*Note on scaling:* OpenCV descriptors have a length of about 512, whereas HOG blocks have length 1. To keep both feature types on a comparable scale (so the same C values {0.1, 1, 10} mean the same thing for both models), every 128-number descriptor is divided by its own length before the descriptors are joined. This does not change the feature length.

### (c) LinearSVC — C chosen on the validation set

| C | validation accuracy | validation macro-F1 |
|---|---|---|
| 0.1 | 0.8083 | 0.8065 |
| **1** (chosen) | **0.8250** | **0.8265** |
| 10 | 0.8167 | 0.8187 |

**Chosen C = 1. Validation accuracy = 82.50 %, validation macro-F1 = 82.65 %.**

Confusion matrix (validation set, rows = true class):

| | pred eosinophil | pred monocyte | pred neutrophil | pred platelet |
|---|---|---|---|---|
| **true eosinophil** | 21 | 6 | 3 | 0 |
| **true monocyte** | 7 | 23 | 0 | 0 |
| **true neutrophil** | 4 | 1 | 25 | 0 |
| **true platelet** | 0 | 0 | 0 | 30 |

---

## Q4. Robustness

### (a) Prediction written before running the robustness test

- **Rotation by 20°:** I think **HOG + SVM will lose more accuracy**. HOG's 9 orientation bins are 20° wide (180° / 9), so a 20° rotation pushes the gradient strength of every edge into the neighbouring orientation bin, and the feature vector changes a lot. The SIFT descriptor spreads each gradient smoothly over neighbouring orientation/position bins (8 bins of 45°, with interpolation and Gaussian weighting), so it should change more gently.
- **Gaussian noise (σ = 0.05):** I think **HOG + SVM will lose more accuracy**. HOG takes gradients directly from the raw pixels, so pixel noise creates many random gradients that the block normalisation then amplifies, whereas SIFT first smooths the image with a Gaussian blur, which removes much of the pixel noise before the gradients are computed.

### (b) Frozen models evaluated on the test set (120 images, 1 image = 0.83 percentage points)

Both models are frozen (trained on the training set with the C chosen on validation; nothing is re-trained). The change is applied to the prepared 128 × 128 gray images **before feature extraction**, and both models receive exactly the same changed images.
Rotation: 20° about the image centre (`cv2.warpAffine`, reflected borders). Noise: Gaussian, σ = 0.05 on the image scaled to 0–1 (seeded with 11).

| Model | Clean accuracy | Rotated accuracy | Noisy accuracy |
|---|---|---|---|
| HOG + SVM | 71.67 % | 70.83 % | 28.33 % |
| SIFT + SVM | 74.17 % | 74.17 % | 75.83 % |

| Model | Drop under rotation (points) | Drop under noise (points) |
|---|---|---|
| HOG + SVM | 0.83 | 43.33 |
| SIFT + SVM | 0.00 | −1.67 |

Example images (clean, rotated, noisy) and a bar chart of the three accuracies per model are plotted in the notebook, Q4 (b) plot cell.

### (c) Was my prediction correct?

1. For noise my prediction was correct: HOG + SVM fell from 71.67 % to 28.33 % (a drop of 43.33 points, close to the 25 % chance level for four balanced classes), while SIFT + SVM stayed at 74.17 % → 75.83 %.
2. For rotation my prediction was only correct by a very small margin, because HOG dropped 0.83 points (1 image) and SIFT 0.00 points, which is too small to show that HOG is really more sensitive, since one test image is 0.83 points.
3. So in my experiment noise is the clear weakness of HOG, while a 20° rotation hardly hurt either model, which is much less than I expected.

---

## Q5. Short answers and AI log

### (a) Why does HOG use block normalisation?
The size of the gradients in a cell depends on lighting and contrast, so the same edge can give a big or a small histogram just because the picture is brighter or darker. HOG groups neighbouring cells into overlapping blocks and normalises each block's histogram (here L2-Hys), which makes the descriptor much less sensitive to illumination and contrast changes.

### (b) Why is it wrong to choose C using the test set?
The test set is only there to measure how well the finished model works on completely unseen data. If C is chosen by looking at test accuracy, the test images have influenced the model (information leakage), so the reported test accuracy becomes too optimistic. C must be chosen on the validation set, and the test set used only once at the end.

### (c) Which two classes were confused most?

Errors between each pair of classes on the validation set (both directions counted):

| class pair | HOG errors | SIFT errors | total errors |
|---|---|---|---|
| **eosinophil – monocyte** | 7 | 13 | **20** |
| eosinophil – neutrophil | 11 | 7 | 18 |
| monocyte – neutrophil | 3 | 1 | 4 |
| neutrophil – platelet | 2 | 0 | 2 |
| eosinophil – platelet | 0 | 0 | 0 |
| monocyte – platelet | 0 | 0 | 0 |

The two classes confused most were **eosinophil and monocyte** (20 errors in total: 7 for HOG and 13 for SIFT). Eosinophil – neutrophil follows closely with 18 errors, so eosinophil is the hardest class (recall on validation: HOG 19/30 = 63.3 %, SIFT 21/30 = 70.0 %), while platelet is almost never confused.

**Why grayscale makes this harder:** eosinophils are recognised mainly by their orange-red granules, and monocytes by their blue-grey cytoplasm. Grayscale turns these colours into similar gray levels, so the colour cue that separates these cells is lost, and only shape and texture remain, which look alike at 64 × 64 pixels.

### (d) AI log

| Prompt (one line) | What I changed | How I checked the result was correct |
|---|---|---|
| Build HOG+SVM and SIFT+SVM for BloodMNIST classes 1,5,6,7, seed 11, HOG cell 16, SIFT 4x4 grid, with a rotation/noise test. | Set my registration number 011 so the settings became seed 11, HOG 16 × 16 and SIFT 4 × 4; asked for automatic dataset loading (official `medmnist`, Kaggle copy as backup); executed the notebook myself; checked and rewrote my Q4(c) and Q5 answers using my own output numbers. | Printed shapes and class counts of every split; asserted that the HOG and SIFT feature lengths equal my hand calculations (1764 and 2048); a 10-line PASS/FAIL check of the exam rules (all PASS); looked at the example grid, HOG pictures, keypoint pictures and the rotated/noisy examples. |
