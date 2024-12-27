# 6D-Gauss: Camera 6D Pose Estimation from a Single RGB Image using a 3D Gaussian Splatting Model

## Overview
This repository contains the implementation of **6D-Gauss**, a novel method for accurate and initialization-free camera pose estimation from a single RGB image. By leveraging a 3D Gaussian Splatting (3DGS) model and Radiant Ellicell ray casting, this approach provides efficient and robust 6DoF camera pose estimation, critical for applications in robotics, augmented reality, autonomous navigation, and scene understanding.

---

## Features
- **Initialization-Free Camera Pose Estimation**: Eliminates reliance on iterative processes or prior pose estimates.
- **3D Gaussian Splatting (3DGS) Model**: Encodes scenes as ellipsoids with color and opacity, optimized using Structure from Motion (SfM).
- **Radiant Ellicell Ray Casting**: Generates ray hypotheses through ellipsoid surfaces for efficient line-of-sight determination.
- **Enhanced Multi-Head Attention**: Aligns features between the image and ray data to predict the camera’s optical center with improved accuracy.
- **Robustness Through RANSAC**: Utilizes RANSAC for accurate intersection points, discarding noisy outliers.

---

## Methodology
1. **3D Gaussian Splatting (3DGS) Model**:
   - Encodes scenes as ellipsoids optimized to match input images.
   - Improves color and opacity representation for accurate rendering.

2. **Radiant Ellicell Ray Casting**:
   - Divides ellipsoid surfaces into equal-area cells.
   - Casts rays to generate hypotheses for the camera's line of sight.

3. **Feature Extraction and Attention**:
   - Image features extracted using DINOv2.
   - Ray features computed with MLP and aligned using an attention map.

4. **Pose Estimation**:
   - Weighted Least Squares for position estimation.
   - Orientation derived from ray directions.

5. **Robustness Enhancements**:
   - RANSAC-based line intersection for inlier selection.
   - Multi-head attention for richer feature alignment.

---

## Workflow
1. **Initialization**:
   - Load necessary libraries and define constants.
2. **Data Preprocessing**:
   - Process input images, rays, and masks.
3. **Gaussian Splat Creation**:
   - Represent scene geometry using ellipsoids.
4. **Ray Processing and Feature Extraction**:
   - Generate and extract ray-based features.
5. **Multi-Head Attention Alignment**:
   - Align image and ray features.
6. **Pose Prediction**:
   - Estimate camera position and orientation.
7. **Training and Evaluation**:
   - Perform training, inference, and testing.
8. **Results Visualization**:
   - Visualize key results and comparisons.

---

## Results
### Barn Dataset
- **Translational Error**:
  - Baseline: **0.162 m**
  - Proposed: **0.0609 m**
- **Angular Error**:
  - Baseline: **30.3°**
  - Proposed: **21.65°**

### Visualizations
- Comparisons of ground truth vs. rendered images.
- Demonstrated robustness against noisy real-world data.

---

## Demo
Watch the demo videos:
- [Demo Video 1](https://youtu.be/G67NHkrIV7s)
- [Demo Video 2](https://youtu.be/HV4mwipMKFI)

---