import pandas as pd
import shutil
from pathlib import Path
from tqdm import tqdm
from typing import List

def split_dataset(
    config_path: str,
):
    # =========================================================
    try:
        from src.utils.config_loader import load_config
        config = load_config(config_path=config_path)
        metadata_filepath = config['directories']['metadata_filepath']
        audio_dir = config['directories']['audio_dir']
        output_dir = config['directories']['output_dir']
        
        use_official_folds = config['splitting']['use_official_folds']
        test_fold = config['splitting']['test_fold']
        val_fold = config['splitting']['val_fold']
        
        seed = config['other']['seed']
    except Exception as e:
        # --- Dir hyperparameters ---
        metadata_filepath = 'data/meta/esc50.csv',
        audio_dir = 'data/audio/',
        output_dir = 'data/splitted/',

        # --- Data splitting hyperparameters ---
        use_official_folds = True,
        test_fold = 5,
        val_fold = 4,

    try:
        metadata_df = pd.read_csv(metadata_filepath)
    except Exception as e:
        raise FileNotFoundError(f"Could not find metadata at {metadata_filepath}: {e}")
    
    # =========================================================
    # Make splitted directory structure
    output_path = Path(output_dir)
    
    def create_folders(base_path: Path):
        base_path.mkdir(parents=True, exist_ok=True)
        for subset in ['train', 'val', 'test']:
            (base_path / subset).mkdir(parents=True, exist_ok=True)
        print(f"Created folder structure at {base_path.absolute()}")

    create_folders(output_path)
    
    # =========================================================
    # Lists to store metadata for each split
    train_list: List[pd.Series] = []
    val_list: List[pd.Series] = []
    test_list: List[pd.Series] = []

    # =========================================================
    # Copy files and collect split metadata
    audio_source_path = Path(audio_dir)
    print("Starting the data splitting process...")
    
    success_count = 0
    for index, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Splitting"):
        filename = row['filename']
        fold = int(row['fold'])
        
        # Determine the subset
        if use_official_folds:
            if fold == test_fold:
                subset = 'test'
            elif fold == val_fold:
                subset = 'val'
            else:
                subset = 'train'
        else:
            subset = 'train'
            
        # Define source and target paths
        source_file = audio_source_path / filename
        target_file = output_path / subset / filename
        
        # Perform copy
        try:
            if source_file.exists():
                shutil.copy2(source_file, target_file)
                success_count += 1
                
                # Update filename to relative path for the CSV
                # Example: '1-100032-A-0.wav' -> 'train/1-100032-A-0.wav'
                row_copy = row.copy()
                row_copy['filename'] = f"{subset}/{filename}"
                
                # Append to the correct list
                if subset == 'train':
                    train_list.append(row_copy)
                elif subset == 'val':
                    val_list.append(row_copy)
                elif subset == 'test':
                    test_list.append(row_copy)
        except Exception as e:
            print(f"Error copying file {filename}: {e}")

    # =========================================================
    # Save split metadata to CSV files in output_dir
    print("Saving split metadata CSVs with relative paths...")
    
    pd.DataFrame(train_list).to_csv(output_path / 'train.csv', index=False)
    pd.DataFrame(val_list).to_csv(output_path / 'val.csv', index=False)
    pd.DataFrame(test_list).to_csv(output_path / 'test.csv', index=False)

    print(f"Metadata CSVs saved to: {output_path.absolute()}")
    print(f"Finished! Successfully copied {success_count} files.")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Load configuration file.")
    parser.add_argument(
        '-d', 
        '--config', 
        type=str, 
        default='configs/datasets/data_splitter.yml',
        help='Path to the configuration file (default: configs/pretraining/data_splitter.yml)'
    )
    args = parser.parse_args()
    config_path = Path(args.config)

    split_dataset(config_path=config_path)