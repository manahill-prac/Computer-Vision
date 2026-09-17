# Lab Task 02 — Results Table & Questions

## Required Experimental Results

| Model | Filter | Accuracy | Precision | Recall | F1-score | Macro-F1 | AUC |
|---|---|---|---|---|---|---|---|
| Best Model 1 (VGG16) | No Filter | --- | --- | --- | --- | --- | --- |
| Best Model 1 (VGG16) | Average | --- | --- | --- | --- | --- | --- |
| Best Model 1 (VGG16) | Gaussian | --- | --- | --- | --- | --- | --- |
| Best Model 1 (VGG16) | Median | --- | --- | --- | --- | --- | --- |
| Best Model 1 (VGG16) | Sharpening | --- | --- | --- | --- | --- | --- |
| Best Model 1 (VGG16) | Sobel | --- | --- | --- | --- | --- | --- |
| Best Model 2 (ResNet18) | No Filter | --- | --- | --- | --- | --- | --- |
| Best Model 2 (ResNet18) | Average | --- | --- | --- | --- | --- | --- |
| Best Model 2 (ResNet18) | Gaussian | --- | --- | --- | --- | --- | --- |
| Best Model 2 (ResNet18) | Median | --- | --- | --- | --- | --- | --- |
| Best Model 2 (ResNet18) | Sharpening | --- | --- | --- | --- | --- | --- |
| Best Model 2 (ResNet18) | Sobel | --- | --- | --- | --- | --- | --- |
| Best Model 3 (AlexNet) | No Filter | --- | --- | --- | --- | --- | --- |
| Best Model 3 (AlexNet) | Average | --- | --- | --- | --- | --- | --- |
| Best Model 3 (AlexNet) | Gaussian | --- | --- | --- | --- | --- | --- |
| Best Model 3 (AlexNet) | Median | --- | --- | --- | --- | --- | --- |
| Best Model 3 (AlexNet) | Sharpening | --- | --- | --- | --- | --- | --- |
| Best Model 3 (AlexNet) | Sobel | --- | --- | --- | --- | --- | --- |

> Run `Lab02_ImageFiltering_SkinLesion.ipynb` in Colab and copy the values from `lab02_results.csv` (or the printed `results_df`) into this table.

## Questions to Answer

**1. Which three pretrained models performed best in Lab Activity 1?**

Based on Task 01's Table 1 results (Accuracy, with AUC as tiebreaker):
1. **VGG16** — 81.82% Accuracy, 97.32% AUC
2. **ResNet18** — 81.82% Accuracy, 89.73% AUC
3. **AlexNet** — 72.73% Accuracy, 96.28% AUC (beats VGG19's 94.4% AUC at the same accuracy)

**2. How does filtering affect each of the three models?**

*To be completed after running the notebook — compare each model's Accuracy/Macro-F1/AUC across the 6 rows in the table above against its own "No Filter" baseline.*

**3. Which filter produces the greatest change compared with the unfiltered baseline?**

*To be completed after running the notebook — for each model, find the filter row with the largest absolute difference from its "No Filter" row.*

**4. Does the effect of a filter remain consistent across all three models?**

*To be completed after running the notebook — check whether each filter moves Accuracy/Macro-F1 in the same direction (up or down) for VGG16, ResNet18, and AlexNet, or whether the effect varies by model.*

**5. Does filtering improve or decrease macro-F1 and balanced accuracy?**

*To be completed after running the notebook — average the Macro-F1 change across all three models, per filter, to see which filters help and which hurt on balance.*

**6. Which lesion classes are most affected by filtering?**

*To be completed after running the notebook — compare per-class F1-score (from the classification report) between "No Filter" and each filtered run; the classes with the largest F1 swings are the most affected.*

**7. Why might smoothing remove useful lesion texture or morphological information?**

Average, Gaussian, and Median filters all work by averaging or ranking pixel values within a local neighborhood. Skin lesions are often distinguished by fine-grained texture (e.g. pigment network, dots/globules, irregular borders) — exactly the kind of high-frequency detail that smoothing filters suppress. By blurring these details, smoothing can erase the subtle morphological cues (border irregularity, texture heterogeneity) that a CNN relies on to distinguish, say, melanoma from a benign nevus.

**8. Why might sharpening or edge detection help or hurt classification?**

Sharpening amplifies high-frequency content (edges, texture), which can help emphasize lesion borders and surface texture — potentially useful since border irregularity is a clinically relevant feature. But it can also amplify noise and artifacts (hair, lighting variation), which may mislead the model. Sobel goes further and discards color/intensity information entirely, keeping only edge magnitude — this can help if the model's decision truly hinges on shape/border, but it destroys pigmentation and color cues (often diagnostically important in dermoscopy), so it can also hurt classification substantially.

**9. What is the difference between convolution and correlation?**

Both operations slide a kernel over an image and compute a weighted sum, but they differ in kernel orientation: **cross-correlation** slides the kernel as-is, while **convolution** first flips the kernel 180° (both horizontally and vertically) before sliding it. For a symmetric kernel (like a Gaussian or mean filter), the two are identical. For an asymmetric kernel (like a Sobel operator), they give different results. In practice, most deep learning frameworks (including PyTorch's `nn.Conv2d`) implement cross-correlation but call it "convolution," since the kernel weights are learned anyway and the flip makes no practical difference for a trained network.

**10. Based on your results, explain the relationship between classical image processing and deep-learning-based feature extraction.**

Classical filters and deep CNN feature extraction both operate on the same core idea — convolving a kernel across an image — but classical filters use fixed, hand-designed kernels (mean, Gaussian, Sobel, etc.) chosen to isolate a specific property (smoothness, edges), while a CNN's convolutional layers *learn* their own kernels from data to isolate whatever features minimize classification error. Applying a classical filter *before* the CNN effectively edits the input distribution the network was pretrained on (ImageNet natural images), which can help if the filter happens to emphasize a feature the network already relies on, or hurt if it destroys information the network needs and can't recover in later layers. The results in the table above should show that filters are not universally good or bad — their effect depends on how well the hand-crafted transformation aligns with what the specific pretrained network already learned to look for.
