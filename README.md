# Efficient and Explainable Brain Tumor Detection from MRI Images Using Transfer Learning and Confidence Analysis

## Project Overview
This project aims to develop a **lightweight and explainable deep learning model** for detecting brain tumors from MRI images. The approach emphasizes **efficiency, interpretability, and reliability** using transfer learning, Grad-CAM visualizations, and confidence analysis.

---

## Dataset
- **Source:** Kaggle – [Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)
- **Classes:** Tumor (Yes) / No Tumor (No)
- **Split:** Train / Validation / Test (Stratified split)
- Images resized to **224x224** for model input.

---

## Methodology

### 1. Data Preprocessing & Augmentation
- **Training Augmentations:** rotation, horizontal flip, width/height shift, zoom, brightness changes.
- **Validation/Test:** only rescaling.
- Custom data generator implemented using `tf.keras.utils.Sequence`.

### 2. Model Architecture
- **Primary Model:** MobileNetV2 (transfer learning)
- **Optional Comparison:** ResNet50, EfficientNetB0
- Classification Head:
  - GlobalAveragePooling2D
  - Dropout (0.3)
  - Dense layer (sigmoid) for binary classification
- **Training Strategy:** Freeze base layers initially, fine-tune last layers, EarlyStopping, ReduceLROnPlateau.

### 3. Explainability
- Grad-CAM and Grad-CAM++ for visualizing important regions.
- Optional SHAP/LIME for additional interpretability.

### 4. Confidence & Uncertainty Analysis
- Softmax probabilities for prediction confidence.
- Entropy-based uncertainty metrics.
- Histograms of confidence distribution.

### 5. Evaluation Metrics
- Accuracy, Precision, Recall, F1-score
- Confusion Matrix
- ROC Curve and Precision-Recall Curve
- Model efficiency analysis: size (MB), inference time per image.

---

## Implementation
- **Platform:** Google Colab
- **Libraries:** TensorFlow/Keras, OpenCV, SHAP, Matplotlib, Seaborn, Scikit-learn
- **GPU:** T4 for fast training on small batch sizes.

---

## Results & Visualizations
- **Grad-CAM Heatmaps:** Highlight tumor regions in MRI images.
- **Accuracy/Loss Curves:** Track training and validation performance.
- **Confusion Matrix:** Evaluate binary classification performance.
- **ROC/PR Curves:** Visualize sensitivity vs specificity.
- **Confidence Histograms:** Analyze model certainty.

---

## Model Efficiency
- Lightweight architecture ensures fast inference.
- Comparison of model sizes and training times for MobileNetV2, ResNet50, and EfficientNetB0.

---

## Conclusion
- The model successfully detects brain tumors from MRI images with **high efficiency and interpretability**.
- Visual explainability via Grad-CAM helps clinicians understand predictions.
- Confidence analysis provides insights into model reliability.

---

## Future Work
- Extend to **multi-class tumor detection** (different tumor types).
- Explore **3D MRI volumes** for richer spatial context.
- Implement **LIME explanations** and integrated **report generation** for clinical use.
- Optimize deployment for **mobile/edge devices**.

---

## References
1. [Kaggle Brain MRI Dataset](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)
2. Selvaraju et al., “Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization,” ICCV 2017.
3. Lundberg & Lee, “A Unified Approach to Interpreting Model Predictions,” NIPS 2017.
4. Howard et al., “MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications,” 2017.
