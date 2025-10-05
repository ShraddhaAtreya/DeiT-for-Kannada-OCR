# Kannada-Character-Recognition-using-DeiT

Deep learning model for recognizing 621 handwritten Kannada character classes (main aksharas + ottaksharas) using Data-efficient Image Transformer (DeiT) architecture with transfer learning.

## Overview

This project implements a DeiT-based character recognition system for Kannada script, achieving 99.90% validation accuracy on both basic and compound characters. The model uses pretrained weights from ImageNet and fine-tunes on a combined dataset of 155,056 handwritten Kannada character images.

## Results

* **Validation Accuracy**: 99.90%
* **Training Accuracy**: 100.00%
* **Precision/Recall/F1-Score**: 0.999
* **Error Rate**: 0.10% (31 errors out of 31,012 samples)
* **Training Time**: ~3 hours (7 epochs on GPU)
* **Model Size**: 21.9M parameters

### Character Type Performance
* **Main Aksharas**: 99.90% (587 classes)
* **Ottaksharas**: 99.88% (34 classes)

## Dataset

* **Source**: Handwritten Kannada Characters Dataset (Main + Ottaksharas)
* **Total Images**: 155,056
* **Classes**: 621 Kannada characters
  - Main Aksharas: 587 classes (146,556 images)
  - Ottaksharas: 34 classes (8,500 images)
* **Split**: 80% training (124,044), 20% validation (31,012)
* **Split Strategy**: Stratified sampling ensuring all classes in both sets
* **Image Size**: 224x224 (resized)

## Model Architecture

* **Base Model**: DeiT Small (`deit_small_patch16_224`)
* **Pretrained**: ImageNet-1K weights
* **Framework**: PyTorch with timm library
* **Input Resolution**: 224x224x3
* **Patch Size**: 16x16
* **Output**: 621 classes (softmax)

## Training Configuration
Optimizer: AdamW
Learning Rate: 1e-4 (initial)
Weight Decay: 0.05
Scheduler: CosineAnnealingLR
Batch Size: 32
Epochs: 7
Loss Function: CrossEntropyLoss
Device: CUDA (GPU)
