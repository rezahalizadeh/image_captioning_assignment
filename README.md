# Image Captioning with CNN-RNN Architecture

## Overview

This repository contains an implementation framework for an image captioning system using a combination of Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs). The system takes images as input and generates natural language descriptions of their content.

Image captioning sits at the intersection of computer vision and natural language processing, combining feature extraction from images with sequence modeling for text generation. This implementation follows the encoder-decoder architecture:

1. An **encoder** (CNN) extracts visual features from input images
2. A **decoder** (RNN/LSTM/GRU) generates captions word-by-word based on these features

## Project Structure

```
image_captioning_assignment/
├── data/
│   └── download_flickr.py     # Script to download and prepare Flickr8k dataset
├── models/
│   ├── encoder.py             # CNN encoder implementations 
│   ├── decoder.py             # RNN decoder implementations
│   └── caption_model.py       # Combined encoder-decoder model
├── utils/
│   ├── dataset.py             # Dataset and data loader utilities
│   ├── vocabulary.py          # Vocabulary building and text processing
│   ├── trainer.py             # Training loop and optimization
│   └── metrics.py             # Evaluation metrics (BLEU, etc.)
├── notebooks/
│   ├── 1_Data_Exploration.ipynb       # Dataset exploration
│   ├── 2_Feature_Extraction.ipynb     # CNN feature extraction
│   ├── 3_Model_Training.ipynb         # Model training
│   └── 4_Evaluation_Visualization.ipynb # Results analysis
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

- The implementation is inspired by the "Show and Tell" paper by Vinyals et al.
- Pre-trained models are provided by torchvision
- Flickr8k dataset from the University of Illinois
