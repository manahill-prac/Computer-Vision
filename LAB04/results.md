# Experimental Results — Lab 04

## Skin Lesion Boundary Detection Using Canny Edge Detection

This document contains the complete results, required tables, and question answers for Lab 04,
compiled from the executed `SkinLesion_BoundaryDetection_Canny.ipynb`. All numbers below are
read directly from that run's outputs — nothing here is estimated or assumed.

---

## 1. Objective

Develop a simple computer vision pipeline that detects the boundary of a skin lesion from a
dermoscopic image using image filtering and Canny edge detection, and evaluate how effectively
edge detection separates the lesion from the surrounding skin.

## 2. Dataset and Setup

- **Source:** ISIC skin-lesion archive, via `nodoubttome/skin-cancer9-classesisic` (same
  dataset as Labs 1–3, downloaded once with `kagglehub`, no manual download).
- **Images used (5, one per class — same 5 classes as Labs 1–3):**

| Image | Class |
|---|---|
| Image 1 | actinic keratosis |
| Image 2 | basal cell carcinoma |
| Image 3 | melanoma |
| Image 4 | nevus |
| Image 5 | pigmented benign keratosis |

- **Image size:** 350×350.
- **Preprocessing pipeline:** grayscale → 5×5 Gaussian blur (σ=0, i.e. OpenCV auto-computed)
  before every Canny call, as required.

This lab is **standalone** — unlike Lab 2 (built on Lab 1's best models) or Lab 3's Task 4
(built on Lab 1 raw + Lab 2's best filter + Lab 3's best edge method), nothing here depends on
a prior lab's selected model or filter; it only reuses the same dataset and class selection for
consistency.

## 3. Task 1 — Original Images

All 5 images loaded and displayed successfully. Two images carry visible non-lesion artifacts
worth noting up front, since they affect later results: **Image 1** has ruler/measurement tick
marks along the left edge, and **Image 4** has dense hair coverage across the lesion area —
both are common dermoscopic-image artifacts, not lesion structure.

## 4. Task 2 — Grayscale and Gaussian Filtering

Grayscale conversion and a 5×5 Gaussian blur were applied to all 5 images. At this kernel size
the blur is mild — visually, lesion shape and the artifacts noted above (ruler marks, hair) are
still clearly present after filtering, not removed.

## 5. Task 3 — Canny at Three Threshold Settings

### Table — Threshold Analysis (averaged across all 5 images)

| Threshold | Low | High | Avg Edge Density (%) | Avg Contour Count |
|---|---:|---:|---:|---:|
| 50–100 | 50 | 100 | 1.147 | 50.4 |
| 100–200 | 100 | 200 | 0.349 | 7.0 |
| 150–250 | 150 | 250 | 0.179 | 5.2 |

Edge density drops by roughly **84%** between the lowest and highest threshold pair, and the
average contour count drops from 50.4 to 5.2 — confirming that raising the thresholds sharply
suppresses detected edges, as expected from Canny's hysteresis mechanism.

**Per-image behavior (visual inspection of the Task 3 figure) matters more than the average
here:** for Images 1, 2, 3, and 5, the 150–250 setting detected **essentially nothing** (fully
black edge maps), and 100–200 detected only a handful of short fragments. Only the 50–100
setting produced a usable edge map for these four images — Image 3 (melanoma) in particular
shows a visually near-complete closed oval outline at 50–100. **Image 4 (nevus) is the
exception**: because it's dominated by strong hair-strand edges, it still shows dense edge
content at *all three* threshold settings, including 150–250 — the hair edges are strong enough
to survive even the highest threshold, while the much weaker true lesion-boundary signal does
not.

## 6. Task 4 — Selected Threshold

**Selected: Canny 50–100.** This was the only setting that produced non-trivial lesion-boundary
edges for 4 of the 5 images; 100–200 and 150–250 were too conservative and discarded the
(comparatively weak) lesion-boundary gradient along with noise. The trade-off is that 50–100 is
also the noisiest setting (highest contour count), which becomes directly relevant to the
boundary-detection problem in Task 5 below.

## 7. Task 5 — Detected Lesion Boundary

**This step did not work well, and that's worth reporting plainly rather than glossing over.**
The pipeline (Canny 50–100 → 5×5 morphological closing → `cv2.findContours` →  largest contour
by area) was applied to all 5 images. Visually inspecting the result:

- In **no image** did the selected contour trace the full lesion outline. Even for Image 3
  (melanoma), where the Canny edge map shows a visually complete closed oval, the drawn
  "boundary" is a short arc covering only a small portion of that oval.
- For **Image 4 (nevus)**, the selected "largest contour" isn't on the lesion at all — it's a
  hair-strand/image-border artifact near the bottom edge of the frame.
- The common failure mode: at the 50–100 threshold, Canny edges are **not fully closed** around
  the lesion (there are gaps), and the 5×5 morphological closing kernel wasn't large enough to
  bridge those gaps into one connected loop. `cv2.findContours` then sees several disconnected
  fragments instead of one ring, and "largest contour by area" picks whichever fragment happens
  to enclose the most (often tiny) area — not necessarily anything resembling the lesion
  boundary.

## 8. Task 6 — Lesion Area and Perimeter

### Required Results Table

| Image | Class | Best Filter | Edge Method | Area (pixels) | Perimeter (pixels) |
|---|---|---|---|---:|---:|
| Image 1 | actinic keratosis | Gaussian | Canny (50–100) | 47.0 | 157.8 |
| Image 2 | basal cell carcinoma | Gaussian | Canny (50–100) | 237.5 | 185.9 |
| Image 3 | melanoma | Gaussian | Canny (50–100) | 243.0 | 172.4 |
| Image 4 | nevus | Gaussian | Canny (50–100) | 683.5 | 919.3 |
| Image 5 | pigmented benign keratosis | Gaussian | Canny (50–100) | 345.5 | 200.4 |

**These numbers should be read as measurements of the detected contour fragments, not of the
actual lesions.** Every area here is under 700 pixels, out of a 350×350 = 122,500-pixel image —
i.e. under 0.6% of the frame — while every lesion is visibly 15–40% of the frame in the Task 1
images. Image 4's unusually high perimeter-to-area ratio (919.3 : 683.5) is consistent with it
being a thin, wiggly hair-artifact fragment rather than a compact lesion outline. See Section 10
(Question 5/6) for why this happened and how to fix it.

## 9. Required Visualization

Generated successfully for all 5 images (`required_visualization.png`): Original → Grayscale →
Gaussian Filter → Canny → Lesion Boundary. This is also the figure that makes the Task 5
shortfall most visible — the Canny column clearly shows more complete lesion outlines (notably
Image 3) than what the final Lesion Boundary column actually traces.

## 10. Final Comparison — Preprocessing × Edge Method

### Computed Metrics (averaged across all 5 images)

| Method | Preprocessing | Edge Method | Avg Edge Density (%) | Avg Contour Count |
|---|---|---|---:|---:|
| Original + Sobel | Original | Sobel | 5.260 | 800.2 |
| Original + Canny | Original | Canny | 5.341 | 376.2 |
| Average + Sobel | Average | Sobel | 10.919 | 364.0 |
| Average + Canny | Average | Canny | 0.572 | 16.8 |
| Gaussian + Sobel | Gaussian | Sobel | 9.213 | 549.2 |
| Gaussian + Canny | Gaussian | Canny | 1.147 | 50.4 |
| Median + Sobel | Median | Sobel | 6.421 | 381.0 |
| Median + Canny | Median | Canny | 0.996 | 38.6 |

### Required Comparison Table

Filled using the metrics above **and** direct visual inspection of the comparison grid (not the
notebook's auto-drafted placeholders, which were explicitly left for manual confirmation):

| Method | Noise Handling | Edge Quality | Boundary Detection | Overall Performance |
|---|---|---|---|---|
| Original + Sobel | Poor | Very dense/noisy (800 contours avg.) | Unusable — texture noise swamps the boundary | Poor |
| Original + Canny | Moderate | Dense but structured | Fragmented, but most complete among Canny variants | Fair |
| Average + Sobel | Poor | Dense/noisy, worst edge density (10.9%) | Unusable | Poor |
| Average + Canny | Good | Very sparse (0.57%) — most true signal smoothed away | Too little signal to form a boundary | Poor |
| Gaussian + Sobel | Poor | Dense/noisy | Unusable | Poor |
| Gaussian + Canny | Good | Sparse but balanced — the setting actually used in Tasks 5–6 | Fragmented (see Section 7) | Fair |
| Median + Sobel | Poor | Dense/noisy | Unusable | Poor |
| Median + Canny | Good | Sparse (1.0%) | Fragmented, similar to Gaussian+Canny | Fair |

**Pattern:** Sobel is unusable for boundary detection under every preprocessing condition tested
— the visual comparison grid shows near-identical heavy speckle noise for all four
Sobel variants, regardless of which smoothing filter preceded it. Canny is consistently cleaner,
but every Canny variant still produced only fragmented edges, not closed boundaries — smoothing
filters (Average/Gaussian/Median) reduce noise before Canny but also suppress much of the
already-weak lesion-boundary signal, while skipping smoothing (Original+Canny) keeps more
boundary signal at the cost of more noise (including the ruler-mark artifact in Image 1).

---

## 11. Questions to Answer

**Q1. Why is Gaussian filtering applied before Canny detection?**
Canny's gradient-computation step is sensitive to high-frequency pixel-to-pixel intensity
variation — in a dermoscopic image that means camera sensor noise, fine skin texture, and
compression artifacts all produce spurious gradient responses that look like edges to the
algorithm. A Gaussian blur suppresses that high-frequency noise before gradients are computed,
so the hysteresis-thresholding stage downstream has a cleaner gradient map to work from and
produces fewer false-positive edges. (Canny's own internal implementation also does a smoothing
step, but applying it explicitly beforehand — as this lab requires — keeps that step visible
and controllable.)

**Q2. How did the three Canny threshold settings affect the result?**
Raising the thresholds sharply reduced detected edges: average edge density fell from 1.147%
(50–100) to 0.349% (100–200) to 0.179% (150–250) — an 84% drop from lowest to highest — and
average contour count fell from 50.4 to 7.0 to 5.2. For 4 of the 5 images, 150–250 detected
almost nothing at all (fully black edge maps), and 100–200 left only a couple of short
fragments; only 50–100 produced edge maps with enough content to work with. Image 4 (the
hair-heavy nevus image) was the exception — its edges stayed dense even at 150–250, because
hair-strand gradients are strong enough to clear even the highest threshold.

**Q3. Which threshold produced the best lesion boundary?**
**50–100**, by a clear margin — it was the only setting where most images produced any
meaningful boundary signal at all. Image 3 (melanoma) at 50–100 shows what's visually close to
a full closed oval outline in the Canny edge map itself. That said, "best threshold" here means
*least bad* rather than good in an absolute sense: 50–100 is also the noisiest of the three
settings, and as Section 7 shows, the downstream contour-extraction step still failed to turn
even this edge map into a correct closed boundary.

**Q4. Why are edges useful for detecting skin lesions?**
A lesion is visually distinguished from surrounding skin mainly by a change in color/intensity
at its border — edge detectors are built specifically to respond to exactly that kind of
intensity transition. In clinical terms, several of the standard ABCDE criteria for lesion
assessment (Border irregularity in particular) are directly about boundary shape, which is
precisely what an edge map captures while discarding less-structured information like overall
color or fine internal texture. In principle this makes edges a natural, lightweight feature for
isolating the lesion's extent for measurement (area, perimeter) without needing a full learned
segmentation model.

**Q5. What problems did you observe in detecting the lesion boundary?**
Several, all visible directly in this run's output:
- **Broken/non-closed edges.** Canny at 50–100 did not produce a single closed loop around any
  lesion — even where the edge map visually looked like a near-complete oval (Image 3), small
  gaps remained, and the 5×5 morphological closing kernel used wasn't large enough to bridge
  them into one connected contour.
- **Largest-contour selection picked the wrong thing.** Because the boundary was fragmented
  into several disconnected pieces, "largest contour by area" selected small arc fragments
  (47–345 pixels of area) rather than the lesion outline, and for Image 4 it selected a
  hair/border artifact entirely unrelated to the lesion.
- **Non-lesion artifacts competing with the real boundary.** Image 4's dense hair produced
  edges strong enough to survive every threshold tested, including the highest (150–250), and
  Image 1's ruler tick marks also appear as edges — both compete with the actual lesion boundary
  for the detector's attention.
- **Low-contrast lesions produced almost no signal.** Image 1 (actinic keratosis), which has
  subtle color contrast against the surrounding skin, produced very sparse edges even at the
  most permissive threshold — there often wasn't a strong boundary gradient to detect in the
  first place.

**Q6. How could your method be improved?**
- **Close the gaps more aggressively before contour extraction** — a larger morphological
  closing kernel (or a dilate→close→erode sequence) to merge nearby edge fragments into one
  loop before calling `findContours`, rather than relying on a single small (5×5) kernel.
- **Remove hair before edge detection.** Dermoscopy pipelines commonly use a blackhat
  morphological filter to detect hair strands and inpaint over them before any edge/contour
  step — this would directly fix the Image 4 failure case.
- **Select contours by shape, not just area.** Instead of "largest contour by area," filter
  candidate contours by compactness/circularity (lesions are roughly round/oval, hair artifacts
  are thin and elongated) or take the convex hull of several nearby contour fragments to
  approximate a closed boundary from broken pieces.
- **Use a segmentation method built for this, instead of raw edge detection.** Approaches like
  GrabCut, active contours ("snakes") seeded from a rough initial region, or Otsu/adaptive
  thresholding on the (color or grayscale) lesion-vs-skin contrast would likely produce a
  genuinely closed region directly, rather than relying on Canny's open edge fragments plus a
  fragile contour-merging step.
- **Use a per-image adaptive threshold** rather than one fixed pair for every image — Image 1's
  low-contrast lesion and Image 4's hair-dominated frame would likely each need a different
  threshold than the one that works reasonably for Images 2, 3, and 5.
