# Beyond Spectral Baselines: Hybrid Feature Engineering and Advanced Modeling for Environmental Sound Classification


> This repository serves as the version control platform for the final-semester project for subject DS200 - Big Data at University of Information Technology, Vietnam National University Ho Chi Minh city. 
>
> Feature selection fundamentally dictates the success of audio classification systems, with spectral features like mel-scaled spectrograms and mel-frequency cepstral coefficients (MFCC) traditionally outperforming rhythm-based representations when processed through Deep Convolutional Neural Networks (CNNs). This project first establishes a robust baseline by replicating established methodologies to systematically evaluate individual spectral and rhythm features on the ESC-50 environmental sound dataset. To move beyond the limitations of single-feature representations and standard CNN architectures, this research then introduces novel hybrid feature sets that fuse spectral and temporal data points. Finally, by deploying and comparing advanced modeling architectures against the CNN baseline, this study aims to optimize feature-model synergy and push the boundaries of classification accuracy and robustness for real-world acoustic environments.

## 1. Project details

## 2. Repository structure

## 3. How to reproduce this project?
Follow the instruction below to reproduce the paper.
### 3.1. Data preparation
This stage prepares the dataset for subsequent experiments and analysis. It involves downloading the data and organizing it according to the specified directory structure. The data is then split into different folds (train, validation, test), and finally, various spectrogram features are extracted before proceeding with further tasks.
#### 3.1.1. Data preparation
Access this [link](https://drive.google.com/drive/folders/15Bd4AoVXwEvjfukuliPj6HrnIdokXFNL?usp=sharing) to download dataset. 
##### Instructions:
1. Download the entire folder from the Google Drive link.
2. Extract (if needed) and place all `.wav` files into `data/audio/`.
3. Place the `esc50.csv` metadata file into `data/meta/`.
4. Ensure your YAML configuration file points to the correct paths:
    - `audio_dir: 'data/audio/'`
    - `metadata_filepath: 'data/meta/esc50.csv'`

```txt
project/data/
├── audio
│   ├── 1-137-A-32.wav
│   ├── 1-977-A-39.wav
│   └── ... # Very many files, 2000 files in total.
├── meta
│   ├── esc50-human.xlsx
│   └── esc50.csv
├── esc50.gif
├── LICENSE
└── README.md
```
> P/s: This dataset is ESC-50 (Environmental Sound Classification).
> - **Author**: Karol J. Piczak
> - **License**: [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
> - **Citation requirement**: If used for research or publication, please cite the original paper:
> 
> `Piczak, K. J. (2015). ESC: Dataset for Environmental Sound Classification. In Proceedings of the 23rd ACM international conference on Multimedia (pp. 1015-1018).`

#### 3.1.2. Data splitting
To split the dataset, run the script below:
```bash
python -m src.dataset.data_splitter -d configs/datasets/data_splitter.yml
```
where:
- `-m src.dataset.data_splitter`: Run the `src/dataset/data_splitter.py` as a module.
- `-d configs/datasets/data_splitter.yml`: Read the configuration file from `configs/datasets/data_splitter.yml`.
> P/s: If you want to change the hyperparameters while splitting, just change it in the `configs/datasets/data_splitter.yml` file.

#### 3.1.3. Feature extraction
After data splitting completion, run the script below to extract 06 features from each audio file. 

```bash
python -m src.dataset.feature_extractor --config configs/datasets/feature_extractor.yml
```
where:
- `-m src.dataset.feature_extractor`: Run the `src/dataset/feature_extractor.py` as a module.
- `--config configs/datasets/feature_extractor.yml`: Read the configuration file from `configs/datasets/feature_extractor.yml`.
> P/s: If you want to change the hyperparameters while splitting, just change it in the `configs/datasets/feature_extractor.yml` file.