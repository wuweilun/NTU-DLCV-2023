# NTU-DLCV-2023
This repository contains the assignments and final project for the NTU Deep Learning for Computer Vision 2023 course. There are a total of four assignments and one final project. Detailed information is provided below.
## Assignment 1 [Report](https://github.com/wuweilun/NTU-DLCV-2023/blob/main/HW1/hw1_r12922075.pdf)

### Task 1: Image Classification
- **Validation Accuracy**:
  - Model A (ResNet-18): 74.96%
  - Model B (EfficientNetV2-S): 90.20%
  
- **PCA Visualization**: Limited class separation, with only some classes clearly distinguishable.
- **t-SNE Visualization**: Improved class separations compared to PCA, with better clustering.

### Task 2: Self-Supervised Learning (SSL) BYOL

| Setting | Pre-training            | Fine-tuning    | Validation Accuracy |
|---------|--------------------------|----------------|---------------------|
| A       | -                         | Train full model | 44.58%             |
| B       | w/ label (TA backbone)    | Train full model | 49.01%             |
| C       | w/o label (SSL backbone)  | Train full model | 50.74%             |
| D       | w/ label (TA backbone)    | Fix backbone     | 28.33%             |
| E       | w/o label (SSL backbone)  | Fix backbone     | 30.54%             |

### Task 3: Semantic Segmentation
- **mIoU (Mean Intersection over Union)**:
  - VGG16-FCN32s: 0.7253
  - DeepLabV3-ResNet50: 0.7597

### Score: 100/100

## Assignment 2 [Report](https://github.com/wuweilun/NTU-DLCV-2023/blob/main/HW2/hw2_r12922075.pdf)

### Task 1: DDPM

- **Inference Time**:
  - Initial: 20 minutes for 1000 images
  - After Disabling CFG: 10 minutes

- **Accuracy**:
  - With CFG: 99.5%
  - Without CFG: 96%

### Task 2: DDIM

- **Eta Values & Image Diversity**:
  - Eta = 0: Denoised images identical to originals
  - Eta ≥ 0.5: Increased diversity in denoised images

- **Interpolation Observations**:
  - Spherical Linear Interpolation (Slerp): Smooth transitions between facial features
  - Simple Linear Interpolation: Blurred intermediate images with less detail

### Task 3: Domain-Adversarial Neural Network (DANN)


| Setting                | MNIST-M → SVHN               | MNIST-M → USPS               |
|------------------------|------------------------------|------------------------------|
| Trained on Source       | 40.03% (6359/15887)          | 80.44% (1197/1488)           |
| Adaptation (DANN)       | 52.51% (8343/15887)          | 93.28% (1388/1488)           |
| Trained on Target       | 93.64% (14877/15887)         | 98.86% (1471/1488)           |

### Score: 100/105

## Assignment 3 [Report](https://github.com/wuweilun/NTU-DLCV-2023/blob/main/HW3/hw3_r12922075.pdf)

### Task 1: Zero-shot image classification with CLIP
- **Validation Accuracy**:
  - “This is a photo of {object}”: 67.48%
  - “This is not a photo of {object}”: 69.64%
  - “No {object}, no score.” 45.24%
  - CIFAR100 prompt templates: 82.48%

### Task 2: PEFT on Vision and Language Model for Image Captioning
| Method         | CIDEr | CLIPScore |
|----------------|-------|-----------|
| Adapter        | 0.964 | 0.733     |
| Lora           | 0.901 | 0.726     |
| Prefix Tuning  | 0.827 | 0.714     |

### Task 3: Visualization of attention in image captioning
- **Example Images**: 
  - Correctly identified objects in two out of three example images.
  - Uncertainty in distinguishing between a tree and a bicycle in the third image.

### Score: 99/100

## Assignment 4 [Report](https://github.com/wuweilun/NTU-DLCV-2023/blob/main/HW4/hw4_r12922075.pdf)

### Task 1: 3D Novel View Synthesis with NeRF

| Settings                          | PSNR  | SSIM   | LPIPS (vgg) |
|-----------------------------------|-------|--------|-------------|
| layers: 8, skips: 4, embedding: 256 | 43.40 | 0.9941 | 0.0986      |
| layers: 8, skips: 4, embedding: 512 | 43.73 | 0.9945 | 0.0991      |
| layers: 6, skips: 3, embedding: 256 | 43.82 | 0.9943 | 0.1004      |

### Score: 100/100

## Final Project: Situated Reasoning (Video Question Answering)

### Overview

This project builds upon the Flipped-VQA architecture to enhance video-text representation and question answering capabilities. The project introduces significant improvements by replacing key components in the model, such as the visual encoder and the underlying language model. The enhancements are designed to improve the accuracy and overall performance of the model in predicting answers (A), questions (Q), and video frames (V) given pairs of VQ, VA, and QA.

### Key Contributions

- **Visual Encoder Replacement**: The visual encoder is replaced with ViCLIP, a video CLIP model specifically designed for transferrable video-text representation.

- **Language Model Upgrade**: The project upgrades from LLAMA1-7B to LLAMA2-7B, a more powerful large language model (LLM).

### Result
- **Int_Acc (Interaction Accuracy)**: 65.12
- **Seq_Acc (Sequence Accuracy)**: 68.38
- **Pre_Acc (Prediction Accuracy)**: 58.10
- **Fea_Acc (Feasibility Accuracy)**: 50.78
- **Mean**: 60.60
