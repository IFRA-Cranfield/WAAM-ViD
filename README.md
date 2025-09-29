# WAAM-ViD
WAAM-ViD is an angle-invariant vision-based melt pool dimension analysis pipeline developed to perform real-time dimensional analysis of the melt pool in Wire Arc Additive Manufacturing (WAAM). By leveraging advanced image processing and machine learning, it ensures precise monitoring and control to enhance build quality, consistency, and sustainability in Additive Manufacturing.

The dataset can be found: [WAAM-ViD Dataset](https://doi.org/10.57996/cran.ceres-2763)

## Overview Diagram
![pipeline](https://github.com/user-attachments/assets/73dd3e39-6255-4be1-b4c5-199505bf5034)


## Contribution
Our contributions can be outlined as follows:
1. Introduced WAAM-ViD, a novel benchmark dataset for multi-angle melt pool dimension analysis in Wire Arc Additive Manufacturing (WAAM).
2. Developed an angle-invariant, vision-based pipeline for accurate melt pool dimensional analysis.
3. Proposed WAAM-ViDNet, a multi-modal deep learning model for robust and continuous melt pool width prediction.

We present a two-stage deep learning pipeline for angle-invariant melt pool dimension analysis. First, DeepLabv3 is employed for the semantic segmentation of the melt pool, fine-tuned through active learning to achieve a 96% Dice score, effectively identifying the melt pool region. Second, WAAM-ViDNet predicts the melt pool width with high accuracy, demonstrating no statistically significant difference from the ground truth. Together, these components form a robust vision-based process monitoring system for WAAM.

## Instruction
The model takes two inputs:
1. Video of WAAM
2. Camera calibration metadata (camera matrix, distortion coefficients, rotation matrix, and translation vector)

Here is a sample video of the prototype:
https://github.com/user-attachments/assets/c7d33783-456d-4ec2-88fe-294c009c18a0

## Sample Segmentation Results
![seg_presentation](https://github.com/user-attachments/assets/f6af48ed-e929-40b1-ae59-6809ce9e37b3)

## Citation
For the use of the WAAM-ViD dataset and/or developed codes, please cite our paper:
```
@article{Kim2025WAAMViD,
  title = {WAAM-ViD: Towards Universal Vision-based Monitoring for Wire Arc Additive Manufacturing},
  author = {Kim, Keun Woo and Kamerkar, Alexander and Chiu, Tzu-En and Abdi, Ibrahim and Qin, Jian and Suder, Wojciech and Asif, Seemal},
  journal = {Frontiers in Manufacturing Technology},
  volume={},
  number={},
  pages={},
  year={},
  publisher={Frontiers}
}
```
