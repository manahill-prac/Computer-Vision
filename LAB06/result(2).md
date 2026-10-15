# Landmark Recognition Using SIFT: Results

A landmark recognition system built with **Python + OpenCV**. It matches SIFT features between a query photo and a database of reference images, identifies the landmark, rejects images it does not know, and localizes the landmark with a RANSAC homography.

The full code is in [`Landmark_Recognition_SIFT.ipynb`](Landmark_Recognition_SIFT.ipynb). This page summarizes what the notebook produced when it was run on Google Colab.

---

## 1. Setup

| Item | Value |
|---|---|
| Dataset | [Wonders of the World Image Classification](https://www.kaggle.com/datasets/balabaskar/wonders-of-the-world-image-classification) (12 classes, 3,846 images), downloaded with `kagglehub` |
| Database landmarks (6) | Eiffel Tower, Great Wall of China, Pyramids of Giza, Roman Colosseum, Statue of Liberty, Taj Mahal |
| Reference images | 8 per landmark (48 in total), stored in one folder per landmark |
| Test queries | 3 held-out images per landmark (18) that are **never** in the database |
| "Unknown" queries | 6 images from two classes excluded from the database (Stonehenge, Chichen Itza) |
| Image size | Longest side resized to at most 800 px |
| Random seed | 42 |

### Method (as run)

| Step | Setting |
|---|---|
| Features | SIFT on grayscale images, at most 2000 keypoints per image, 128-D descriptors |
| Matcher | Brute-Force, Euclidean (L2) distance, KNN with k = 2 |
| Ambiguity filter | Lowe's ratio test, ratio = 0.75 |
| Landmark score | Reliable matches of the best reference image of each landmark, then ranked |
| Geometric verification | RANSAC homography (reprojection threshold 5 px) on the top 3 candidates |
| Localization | The 4 corners of the reference image are projected into the query image. The polygon is drawn only if it is convex and has a sensible area. |
| Accept only if | reliable matches ≥ 15, **and** RANSAC inliers ≥ 10, **and** inlier ratio ≥ 0.25, **and** the polygon is valid |

The decision never relies on the largest match count alone. Without the geometric check, every query would be assigned to some landmark, including landmarks that are not in the database.

### Database

Average SIFT keypoints per reference image:

| Landmark | Reference images | Avg. keypoints / image |
|---|---|---|
| Eiffel Tower | 8 | 924 |
| Great Wall of China | 8 | 1092 |
| Pyramids of Giza | 8 | 657 |
| Roman Colosseum | 8 | 941 |
| Statue of Liberty | 8 | 412 |
| Taj Mahal | 8 | 992 |

---

## 2. Six test conditions (Task 6)

All six tests use the Eiffel Tower. Tests 1 to 4 start from one held-out photo, test 5 is a different held-out photo of the same landmark, and test 6 is an image of a landmark that is not in the database.

| # | Test condition | True landmark | Predicted | Keypoints | Reliable matches | RANSAC inliers | Decision | Time (s) | Correct |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Original | Eiffel Tower | Eiffel Tower | 270 | 23 | 17 | Recognized | 0.30 | yes |
| 2 | Rotated (45°) | Eiffel Tower | Eiffel Tower | 372 | 22 | 18 | Recognized | 0.43 | yes |
| 3 | Resized (50%) | Eiffel Tower | Statue of Liberty (rejected) | 69 | 2 | 0 | Rejected | 0.08 | **no** |
| 4 | Reduced brightness (×0.4) | Eiffel Tower | Eiffel Tower | 127 | 15 | 14 | Recognized | 0.14 | yes |
| 5 | Different viewpoint | Eiffel Tower | Eiffel Tower (rejected) | 296 | 6 | 0 | Rejected | 0.32 | **no** |
| 6 | Not in database | (not in database) | Pyramids of Giza (rejected) | 982 | 19 | 5 | Rejected | 1.08 | yes |

Overall, 4 of the 6 tests produced the correct outcome.

---

## 3. Evaluation on all held-out queries

Every held-out query was run through the full pipeline (18 known-landmark queries and 6 unknown-landmark queries).

| Metric | Result |
|---|---|
| Known landmarks correctly recognized | **4 / 18 (22.2%)** |
| Known landmarks wrongly labelled as another landmark | **0 / 18** |
| Known landmarks rejected (missed) | 14 / 18 |
| Unknown landmarks correctly rejected | **6 / 6 (100%)** |
| Overall accuracy | 10 / 24 (41.7%) |
| Mean processing time | 0.95 s per query |

**Top-ranked candidate before the accept/reject rule.** In 11 of the 18 known queries (61%) the landmark ranked first by match count was the correct one. The strict acceptance thresholds then rejected most of them. This means the ranking stage works reasonably well, but the thresholds are too strict for this dataset.

### Confusion matrix (final decision)

| True \ Predicted | Eiffel Tower | Great Wall | Roman Colosseum | Taj Mahal | UNKNOWN (rejected) |
|---|---|---|---|---|---|
| Eiffel Tower | 1 | 0 | 0 | 0 | 2 |
| Great Wall of China | 0 | 1 | 0 | 0 | 2 |
| Pyramids of Giza | 0 | 0 | 0 | 0 | 3 |
| Roman Colosseum | 0 | 0 | 1 | 0 | 2 |
| Statue of Liberty | 0 | 0 | 0 | 0 | 3 |
| Taj Mahal | 0 | 0 | 0 | 1 | 2 |
| UNKNOWN (Stonehenge, Chichen Itza) | 0 | 0 | 0 | 0 | 6 |

The Pyramids of Giza and the Statue of Liberty were never recognized. The Statue of Liberty has the fewest keypoints in the database (about 412 per image), and its queries had only 204 to 334 keypoints.

### Per-query measurements

| Query | True | Top candidate | Keypoints | Reliable matches | Inliers | Polygon valid | Final |
|---|---|---|---|---|---|---|---|
| chichen_itza_01 | UNKNOWN | Pyramids of Giza | 982 | 19 | 5 | no | UNKNOWN ✔ |
| chichen_itza_02 | UNKNOWN | Statue of Liberty | 499 | 9 | 4 | no | UNKNOWN ✔ |
| chichen_itza_03 | UNKNOWN | Great Wall of China | 329 | 10 | 5 | no | UNKNOWN ✔ |
| stonehenge_01 | UNKNOWN | Great Wall of China | 1343 | 71 | 8 | no | UNKNOWN ✔ |
| stonehenge_02 | UNKNOWN | Pyramids of Giza | 166 | 4 | 4 | no | UNKNOWN ✔ |
| stonehenge_03 | UNKNOWN | Roman Colosseum | 318 | 10 | 5 | no | UNKNOWN ✔ |
| Eiffel Tower 1 | Eiffel Tower | Eiffel Tower | 270 | 23 | 17 | yes | Eiffel Tower ✔ |
| Eiffel Tower 2 | Eiffel Tower | Eiffel Tower | 296 | 6 | 0 | no | UNKNOWN ✘ |
| Eiffel Tower 3 | Eiffel Tower | Eiffel Tower | 746 | 11 | 9 | yes | UNKNOWN ✘ |
| Great Wall 1 | Great Wall | Taj Mahal | 630 | 10 | 5 | no | UNKNOWN ✘ |
| Great Wall 2 | Great Wall | Great Wall | 2000 | 1115 | 1113 | yes | Great Wall ✔ |
| Great Wall 3 | Great Wall | Statue of Liberty | 1048 | 19 | 5 | no | UNKNOWN ✘ |
| Pyramids 1 | Pyramids | Pyramids | 425 | 21 | 6 | yes | UNKNOWN ✘ |
| Pyramids 2 | Pyramids | Taj Mahal | 309 | 8 | 4 | no | UNKNOWN ✘ |
| Pyramids 3 | Pyramids | Great Wall | 314 | 6 | 5 | no | UNKNOWN ✘ |
| Colosseum 1 | Colosseum | Colosseum | 2000 | 43 | 30 | yes | Colosseum ✔ |
| Colosseum 2 | Colosseum | Colosseum | 544 | 14 | 13 | yes | UNKNOWN ✘ |
| Colosseum 3 | Colosseum | Statue of Liberty | 2000 | 85 | 7 | no | UNKNOWN ✘ |
| Statue of Liberty 1 | Statue of Liberty | Statue of Liberty | 251 | 8 | 5 | no | UNKNOWN ✘ |
| Statue of Liberty 2 | Statue of Liberty | Pyramids | 204 | 9 | 6 | no | UNKNOWN ✘ |
| Statue of Liberty 3 | Statue of Liberty | Statue of Liberty | 334 | 11 | 6 | no | UNKNOWN ✘ |
| Taj Mahal 1 | Taj Mahal | Taj Mahal | 2001 | 70 | 6 | no | UNKNOWN ✘ |
| Taj Mahal 2 | Taj Mahal | Taj Mahal | 228 | 24 | 19 | yes | Taj Mahal ✔ |
| Taj Mahal 3 | Taj Mahal | Great Wall | 714 | 13 | 6 | no | UNKNOWN ✘ |

(The per-query rows list queries in the same order as the notebook's table. "Eiffel Tower 1", "Great Wall 2" and so on stand for `query_01.jpg`, `query_02.jpg` and so on of that landmark.)

### Threshold sensitivity

Re-applying the decision rule with different `MIN_INLIERS` values:

| MIN_INLIERS | Known recognized (%) | Unknown rejected (%) |
|---|---|---|
| 4 | 27.8 | 100 |
| 6 | 27.8 | 100 |
| 8 | 22.2 | 100 |
| **10 (used)** | **22.2** | **100** |
| 12 | 22.2 | 100 |
| 15 | 22.2 | 100 |
| 20 | 11.1 | 100 |
| 25 | 11.1 | 100 |
| 30 | 11.1 | 100 |

Lowering `MIN_INLIERS` only raised recognition from 22.2% to 27.8%, while every unknown image was still rejected. The other thresholds (`MIN_GOOD`, the inlier ratio and the polygon-validity check) also contribute to the rejections, because most missed queries have very few reliable matches in the first place.

---

## 4. Demo run (Task 3 and Task 5)

The query-processing and localization demo cells (Tasks 3 and 5) picked their image from the `_unknown` query folder, so these two demos used a **Chichen Itza** image, which is not in the database. It is also a useful example of correct rejection: the top candidate (Pyramids of Giza) had 19 reliable matches but only 5 RANSAC inliers, so no region is drawn.

---

## 5. Discussion

**What worked.**
- **Zero false labels.** No known landmark was ever labelled as a different landmark (0 / 18), and all 6 unknown-landmark images were rejected. The combination of RANSAC inliers, an inlier ratio and a polygon-validity check does what it is meant to do: random matches between unrelated buildings do not agree on one geometric transformation. The `Stonehenge` image with 71 reliable matches but only 8 inliers shows this clearly.
- **Strong matches are very strong.** One Great Wall query produced 1115 reliable matches with 1113 inliers, and one Colosseum query 43 matches with 30 inliers. The Great Wall result is so extreme that this query is probably a near-duplicate of one of the reference images, since the dataset contains many similar web photos. Treat it as an outlier, not as typical performance.
- **Rotation and illumination.** The 45° rotation (22 matches, 18 inliers) and the ×0.4 brightness reduction (15 matches, 14 inliers) were both recognized, as expected from SIFT's rotation invariance and normalized descriptors.
- **Speed.** 0.95 s per query on average with a brute-force matcher over 48 reference images.

**What did not work.**
- **Recall is low (22.2%).** The system is conservative: 14 of the 18 known landmarks were rejected, even though the correct landmark was the top-ranked candidate in 11 of 18 cases. These errors are missed detections, not wrong answers.
- **The resized image failed (test 3).** Halving the image left only 69 keypoints and 2 reliable matches. The cause is the lost detail in an already small, low-resolution photo, not a failure of SIFT's scale invariance in principle.
- **Different viewpoint failed (test 5).** Only 6 reliable matches and 0 inliers were found. SIFT is only partly invariant to 3D viewpoint changes.
- **The brightness test only just passed.** It had exactly 15 reliable matches, equal to `MIN_GOOD`, so it sits right on the acceptance boundary.

**Limitations.**
1. **Low-resolution, low-texture queries.** Many dataset images are small web thumbnails. Several queries gave only 200 to 330 keypoints, and weak or watermarked images (the Eiffel Tower query carries a stock-photo watermark) add keypoints that do not belong to the landmark.
2. **Large viewpoint and appearance changes.** The reference and query photos of the same landmark often differ strongly in angle, distance and lighting. Only 8 reference images per landmark cover a small part of that variation.
3. **Planar assumption.** A homography models a flat surface, but landmarks are 3D. The projected polygon is the outline of the whole reference *image*, so it can extend beyond the frame or be rotated (as in test 1), instead of tightly bounding the landmark.
4. **Repetitive structures** (Colosseum arches, Great Wall bricks, pyramid blocks) create ambiguous descriptors, which the ratio test removes. This also lowers the count for true matches.
5. **Thresholds.** The acceptance thresholds (`MIN_GOOD = 15`, `MIN_INLIERS = 10`, `MIN_INLIER_RATIO = 0.25`) were set in advance and tested on a small set of 24 queries. They favor rejecting uncertain images, which suits a system where a wrong answer is worse than no answer but costs recall.
6. **Scalability.** Brute-force matching grows linearly with the number of reference images. A real tourism system would need approximate nearest-neighbour search (FLANN) or a visual-vocabulary index for the first stage.

**Possible improvements.**
- Use more reference images per landmark, and keep only larger or higher-quality ones.
- Upscale small images or lower SIFT's `contrastThreshold` to get more keypoints from low-resolution queries.
- Relax `MIN_GOOD` / the inlier ratio, or score a landmark with the sum of matches over several reference images, then re-evaluate on a larger test set.
- Try RootSIFT, mutual (cross-checked) matching and `cv2.USAC_MAGSAC` for the homography.
- Combine SIFT with a learned (CNN) global descriptor for the first-stage retrieval, and keep SIFT + RANSAC only for verification and localization.

---

## 6. Reproducing the results

1. Open `Landmark_Recognition_SIFT.ipynb` in Google Colab.
2. Run all cells (**Runtime → Run all**). The dataset is downloaded automatically with `kagglehub`.
3. The notebook writes the organized database, query images, stored descriptors and result tables, and packages them as `landmark_submission.zip`.

Settings: `SEED = 42`, OpenCV 4.14.0. Because the reference and query images are chosen at random from the dataset, a different seed or dataset version can give different numbers.
