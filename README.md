# Beyond Spectral Baselines: Hybrid Feature Engineering and Advanced Modeling for Environmental Sound Classification


> This repository serves as the version control platform for the final-semester project for subject DS200 - Big Data at University of Information Technology, Vietnam National University Ho Chi Minh city. 
>
> Feature selection fundamentally dictates the success of audio classification systems, with spectral features like mel-scaled spectrograms and mel-frequency cepstral coefficients (MFCC) traditionally outperforming rhythm-based representations when processed through Deep Convolutional Neural Networks (CNNs). This project first establishes a robust baseline by replicating established methodologies to systematically evaluate individual spectral and rhythm features on the ESC-50 environmental sound dataset. To move beyond the limitations of single-feature representations and standard CNN architectures, this research then introduces novel hybrid feature sets that fuse spectral and temporal data points. Finally, by deploying and comparing advanced modeling architectures against the CNN baseline, this study aims to optimize feature-model synergy and push the boundaries of classification accuracy and robustness for real-world acoustic environments.

## 1. Project details

## 2. Repository structure

## 3. How to reproduce this project?

### 3.1. Data preparation
#### 3.1.1. Data preparation

#### 3.1.2. Data splitting
To split the dataset, run the script below:
```bash
python -m src.dataset.data_splitter -d configs/pretraining/data_splitter.yml
```
where:
- `-m src.dataset.data_splitter`: Run the `src/dataset/data_splitter.py` as a module.
- `-d configs/pretraining/data_splitter.yml`: Read the configuration file from `configs/pretraining/data_splitter.yml`.
> P/s: If you want to change the hyperparameters while splitting, just change it in the `configs/pretraining/data_splitter.yml` file.

#### 3.1.3. Feature extraction
After data splitting completion, run the script below to extract 06 features from each audio file. 

```bash
python -m src.dataset.feature_extractor --config configs/pretraining/feature_extractor.yml
```
where:
- `-m src.dataset.feature_extractor`: Run the `src/dataset/feature_extractor.py` as a module.
- `--config configs/pretraining/feature_extractor.yml`: Read the configuration file from `configs/pretraining/feature_extractor.yml`.
> P/s: If you want to change the hyperparameters while splitting, just change it in the `configs/pretraining/feature_extractor.yml` file.