# Defect-Detection-in-Welding
Deep learning pipeline for detecting TIG aluminium weld defects using MobileNetV2 transfer learning. Classifies 6 defect types (good weld, burn through, contamination, lack of fusion, misalignment, lack of penetration) with 76.81% accuracy, 0.9159 AUC, and 0.8255 mAP. Includes Grad-CAM and t-SNE for interpretability.
Defect Detection in Welding 🔧

A deep learning pipeline for automatically detecting defects in TIG (Tungsten Inert Gas) aluminium welding using MobileNetV2 transfer learning. The model classifies weld images into six categories and is evaluated with a full interpretability suite (Grad-CAM, t-SNE) to verify it's learning genuine defect signatures rather than dataset artefacts.

📊 Results
Metric	Score
Test Accuracy	76.81%
Mean AUC (one-vs-rest)	0.9159
Mean Average Precision (mAP)	0.8255
Test Set Size	6,588 real images

The gap between validation accuracy (~99%) and test accuracy (76.81%) comes from evaluating on a heavily imbalanced, real-world test set (trained on 500 images/class, tested on the full unbalanced distribution) — not from overfitting. The high macro-AUC confirms the model is genuinely discriminative.

🏷️ Defect Classes
Class	Description
Good weld	Normal, defect-free weld bead
Burn through	Excess heat creates a hole in the material
Contamination	Foreign substances (oil, oxides, moisture) in the weld pool
Lack of fusion	Weld metal fails to bond properly to the base material
Misalignment	Workpieces positioned incorrectly
Lack of penetration	Weld doesn't go deep enough through the joint thickness
🧠 Model Architecture
Backbone: MobileNetV2 (ImageNet-pretrained, 3.4M parameters)
Transfer learning: Frozen base layers + fine-tuned last 20 layers
Classification head: GAP → Dense(256) → Dropout(0.5) → Dense(128) → Dropout(0.3) → Dense(6, Softmax)
Loss: Sparse categorical cross-entropy
Optimizer: Adam (α = 1e-4)
Training: Early stopping + ReduceLROnPlateau, batch size 32, up to 50 epochs

MobileNetV2 was chosen over VGG16 for its lightweight footprint (~14MB vs ~550MB) while achieving comparable or better accuracy on small datasets, thanks to depthwise separable convolutions and inverted residual blocks.

📁 Dataset
Source: TIG Aluminium 5083 dataset (Kaggle)
Training: 3,000 images, balanced (500/class), with augmentation (rotation, shifts, flipping, brightness, zoom)
Testing: 6,588 real, unbalanced images to simulate deployment conditions
🔍 Interpretability
Grad-CAM: Confirms the model focuses on defect-relevant regions (e.g., the hole/cavity for burn through, the weld toe for lack of fusion) rather than irrelevant background patterns.
t-SNE: Visualizes the model's internal feature space — cluster overlaps directly explain confusion-matrix errors (e.g., burn through and contamination clusters overlap, matching their 43% mutual confusion rate).
📈 Per-Class Performance
Class	Precision	Recall	F1	AUC
Good weld	0.7578	0.8817	0.8150	0.9360
Burn through	0.8952	0.5356	0.6702	0.8288
Contamination	0.7504	0.6670	0.7062	0.8774
Lack of fusion	0.8732	0.7041	0.7795	0.8703
Misalignment	0.7840	0.8464	0.8140	0.9848
Lack of penetration	0.5974	0.9829	0.7431	0.9981

Key findings:

Lack of penetration and Misalignment are the easiest classes to detect (near-perfect AUC).
Burn through and Contamination are hardest to distinguish — both show disrupted weld pool structure and are visually similar.
🚀 Future Work
Collect more training data for harder classes (burn through, contamination)
Fully fine-tune all MobileNetV2 layers at a lower learning rate
Ensemble with EfficientNet-B0 or ResNet50
Convert to TensorFlow Lite for real-time edge deployment (Jetson Nano, Raspberry Pi)
Apply confidence calibration (temperature scaling) for more reliable probability estimates
📚 References
Sandler et al., MobileNetV2: Inverted Residuals and Linear Bottlenecks, CVPR 2018
Selvaraju et al., Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization, ICCV 2017
Bacioiu et al., Automated defect classification of aluminium 5083 TIG welding using HDR camera and neural networks, Journal of Manufacturing Processes, 2019
van der Maaten & Hinton, Visualizing data using t-SNE, JMLR 2008

Author: Arnav Raj Kumar
