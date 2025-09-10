# Image Captioning with CNN-RNN Architecture

This repository implements a complete **image captioning system** using a **CNN encoder** to extract visual features and an **RNN decoder** (LSTM/GRU) to generate descriptive captions word-by-word.

---

## Architecture Overview

The model follows the classic **Encoder-Decoder** pattern:

- **Encoder**: Pre-trained CNN backbone (ResNet, MobileNet, Inception) extracts image features → projected to embedding space.
- **Decoder**: RNN (LSTM or GRU) generates captions conditioned on image features using teacher forcing during training.
- **Inference**: Supports both **greedy decoding** and **beam search** for caption generation.

---

## Modules

### 1. `encoder.py` — CNN Feature Extractor
- Uses pre-trained models from `torchvision`: `resnet18`, `resnet50`, `mobilenet_v2`, `inception_v3`.
- Removes final classifier layer and adds a projection head (`Linear → BatchNorm → ReLU → Dropout`).
- Option to freeze CNN weights during training.

### 2. `decoder.py` — RNN Caption Generator
- Supports both **LSTM** and **GRU** architectures.
- Word embedding layer + multi-layer RNN + output projection.
- Training with **teacher forcing**.
- Inference with:
  - **Greedy search** (fast, deterministic)
  - **Beam search** (higher quality, batch_size=1 only)

### 3. `caption_model.py` — End-to-End Pipeline
- Integrates encoder and decoder into a single trainable model.
- Maps CNN features to RNN hidden state space.
- Unified `forward()` for training and `generate_caption()` for inference.

---
