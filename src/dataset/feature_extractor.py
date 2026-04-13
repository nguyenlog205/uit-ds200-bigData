from pathlib import Path
import pandas as pd
from tqdm import tqdm 

import numpy as np
import pandas as pd
import librosa
from pathlib import Path

def extract_feature(config_path: str):
    # ==========================================================
    # --- Load configuration and tools ---
    from src.utils.config_loader import load_config
    config = load_config(config_path=config_path)

    from src.dataset.feature_extractors.cens_chromagram import CENSChromagram
    from src.dataset.feature_extractors.chroma import STFTChromagram
    from src.dataset.feature_extractors.ctq_chromagram import CQTChromagram
    from src.dataset.feature_extractors.melspectrogram import MelScaleSpectrogram
    from src.dataset.feature_extractors.mfcc import MFCC
    from src.dataset.feature_extractors.tempo import CyclicTempogram

    tool_01 = CENSChromagram(**config['feature_extractor_params']['cens_chromagram'])
    tool_02 = STFTChromagram(**config['feature_extractor_params']['chroma_stft'])
    tool_03 = CQTChromagram(**config['feature_extractor_params']['cqt_chromagram'])
    tool_04 = MelScaleSpectrogram(**config['feature_extractor_params']['melspectrogram'])
    tool_05 = MFCC(**config['feature_extractor_params']['mfcc'])
    tool_06 = CyclicTempogram(**config['feature_extractor_params']['cyclic_tempogram'])
    tools = {
        'cens_chromagram': tool_01,
        'chroma_stft': tool_02,
        'cqt_chromagram': tool_03,
        'melspectrogram': tool_04,
        'mfcc': tool_05,
        'cyclic_tempogram': tool_06,
    }

    subdataset_paths = config['directories']['audio_dir']   # list of 3 paths
    sub_metadata_paths = config['directories']['metadata_dir']  # list of 3 paths
    output_base = Path(config['directories']['output_dir'])
    output_base.mkdir(parents=True, exist_ok=True)

    extract_ctrl = config['extraction_control']
    target_sr = extract_ctrl['target_sr']
    duration = extract_ctrl['duration']
    mono = extract_ctrl['mono']
    save_format = extract_ctrl['save_format']   # 'npy'
    group_by_feature = extract_ctrl['group_by_feature']  # false → flat structure

    # ==========================================================
    # --- Start extracting each sub-dataset ---
    for dataset_path_str, metadata_path_str in zip(subdataset_paths, sub_metadata_paths):
        subset_name = Path(dataset_path_str).name   # 'train', 'val', or 'test'
        print(f"\nProcessing {subset_name} ...")

        dataset_path = Path(dataset_path_str)
        filepaths = [str(p) for p in dataset_path.glob("*.wav")]
        metadata_df = pd.read_csv(metadata_path_str)

        # Prepare output directory for this subset
        subset_output_dir = output_base / subset_name
        subset_output_dir.mkdir(parents=True, exist_ok=True)

        # List to collect new metadata rows
        new_metadata_rows = []

        for idx, filepath in enumerate(tqdm(filepaths, desc=f"Extracting {subset_name}")):
            meta_row = metadata_df.iloc[idx]

            # Load audio
            y, sr = librosa.load(filepath, sr=target_sr, mono=mono, duration=duration)
            # Pad or truncate to exact duration (librosa.load with duration already does that)
            # but ensure exact length: if sr * duration samples
            expected_len = int(target_sr * duration)
            if len(y) < expected_len:
                y = np.pad(y, (0, expected_len - len(y)))
            else:
                y = y[:expected_len]

            filename = Path(filepath).stem   # e.g., "1-100032-A-0"

            # Extract features with each tool
            for tool_name, tool in tools.items():
                feature = tool.transform(y)   # numpy array
                # Save feature
                if group_by_feature:
                    # output_dir/feature_name/subset/filename.npy
                    feat_dir = output_base / tool_name / subset_name
                    feat_dir.mkdir(parents=True, exist_ok=True)
                    save_path = feat_dir / f"{filename}.{save_format}"
                else:
                    # output_dir/subset/filename_featurename.npy
                    save_path = subset_output_dir / f"{filename}_{tool_name}.{save_format}"

                if save_format == 'npy':
                    np.save(save_path, feature)
                else:
                    raise ValueError(f"Unsupported save_format: {save_format}")

                # Build new metadata row
                new_row = {
                    'file_path': filepath,           # original audio path
                    'target': meta_row.get('target', None),
                    'category': meta_row.get('category', None),
                    'esc10': meta_row.get('esc10', None),
                    'src_file': meta_row.get('src_file', None),
                    'take': meta_row.get('take', None),
                    'feature_name': tool_name,
                    'feature_path': str(save_path)
                }
                new_metadata_rows.append(new_row)

        # Save metadata CSV for this subset
        new_metadata_df = pd.DataFrame(new_metadata_rows)
        csv_path = output_base / f"{subset_name}.csv"
        new_metadata_df.to_csv(csv_path, index=False)
        print(f"Saved {subset_name} metadata to {csv_path}")

    print("\nAll subsets processed.")
            
        
    


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', 
        '--config', 
        type=str, 
        default='configs/pretraining/feature_extractor.yml'
    )
    args = parser.parse_args()
    # extract_feature(args.config)

# python -m src.dataset.feature_extractor