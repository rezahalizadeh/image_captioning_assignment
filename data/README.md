# Flickr8k Dataset Downloader & Preprocessor for Image Captioning

This script automates the download, extraction, and preprocessing of the **Flickr8k dataset** for use in **image captioning** tasks. It organizes images and captions into clean directories and splits the data into train, validation, and test sets according to the official Flickr8k splits.

---
Run the script to download and preprocess the dataset:

```bash
python flickr8k_downloader.py --data_dir ./data
```
--data_dir (optional): Base directory to store the dataset. Defaults to ./data.
