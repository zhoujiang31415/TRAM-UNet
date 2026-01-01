# TRAM-UNet: Transformer and Region Attention Module based U-Net for Breast Ultrasound Image Segmentation
Official PyTorch implementation of TRAM-UNet, as presented at the [IEEE EMBC 2025](https://ieeexplore.ieee.org/document/11253082).


## 📖 Introduction
Segmentation of breast ultrasound images is crucial for the early and accurate diagnosis of breast cancer. In this study, we propose TRAM-UNet (Transformer and Region Attention Module-Based U-Net), a novel deep learning model that integrates Transformer blocks and a Region Attention Module (RAM) to improve segmentation performance. TRAM-UNet achieves average Dice scores of 88.56±0.91%, 84.68±1.20%, and 83.96±1.34% on the BUS-BRA, BUSI, and BLUI datasets, respectively, significantly outperforming both U-Net and U-Net+Transformer across all datasets. These results demonstrate TRAM-UNet’s ability to refine boundaries, enhance segmentation accuracy, and adapt to different lesion characteristics, underscoring its potential to advance breast ultrasound segmentation and clinical diagnosis.Clinical Relevance— This study is clinically relevant as it demonstrates the potential of deep learning in improving breast ultrasound image segmentation. With further research and optimization, this approach could contribute to more precise and automated breast cancer diagnosis in clinical practice.

