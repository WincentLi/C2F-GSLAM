<p align="center">
  <h1>C2F-GSLAM: Coarse-to-Fine Gaussian SLAM with Geometry-Switching Primitives</h1>
</p>

## 项目概述

The application of Gaussian splatting in SLAM systems has led to significant advancements in photorealistic mapping, benefiting both academic research and industrial applications. However, existing Gaussian-based SLAM frameworks often face the issue of overfitting to training trajectories, which results in reduced mapping quality when viewed from unseen viewpoints. Additionally, the direct use of anisotropic primitives at the outset often leads to unstable optimization and high computational costs, limiting scalability for large-scale mapping tasks.

To overcome these challenges, we propose a novel coarse-to-fine Gaussian SLAM framework featuring a geometry-switching strategy. In the coarse stage, isotropic Gaussians are utilized to facilitate rapid initialization and robust tracking. In the fine stage, these isotropic Gaussians are transformed into anisotropic primitives for high-fidelity reconstruction. Furthermore, we introduce a three-step Gaussian adjustment strategy that improves map quality: pruning low-quality primitives to clean the map, cloning high-quality ones to increase density in underrepresented regions, and adaptively adding new Gaussians to capture missing details.

Extensive experiments across several benchmark datasets show that our method delivers state-of-the-art photometric quality and competitive geometric accuracy, even performing well on untrained viewpoints. The proposed framework sets a new standard in photorealistic SLAM mapping.

<p align="center">
  <img src="main/img/overview.png" alt="项目概览">
</p>
