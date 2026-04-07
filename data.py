from torch.utils.data import Dataset, DataLoader
import torch
from pathlib import Path
import pytorch_lightning as pl
import pandas as pd
import numpy as np


from torch.nn.utils.rnn import pad_sequence
def collate_fn(batch):
    x = [item['x'] for item in batch]
    y = [item['y'] for item in batch]
    v = [item['v'] for item in batch]
    x_len = [len(item['x']) for item in batch]
    x_padded = pad_sequence(x, batch_first=True, padding_value=0)
    y_padded = pad_sequence(y, batch_first=True, padding_value=0)
    return {
        "x": x_padded,
        "y": y_padded,
        "v": torch.stack(v, dim=0),
        "x_len": x_len
    }


class TorqueForceDataset(Dataset):
    def __init__(self, metadata_df, global_mean_std_file, window_size=500, step_size=125):
        """
        Args:
            metadata_df: DataFrame with metadata containing file paths and parameters
            global_mean_std_file: Path to global_mean_std.csv
        """
        self.metadata_df = metadata_df.reset_index(drop=True)
        self.global_mean_std = pd.read_csv(global_mean_std_file).set_index('column')
        self.window_size = window_size
        self.step_size = step_size

    def __len__(self):
        return len(self.metadata_df)
    
    def __getitem__(self, idx):
        row = self.metadata_df.iloc[idx]
        sample_id = row['id']
        
        # Use paths from metadata (they include full paths or relative paths)
        torque_file = row['input_path']
        force_file = row['output_path']
        
        # Load torque and power data from input_path
        torque_data = pd.read_csv(torque_file)
        # Select columns Torque and Power
        torque_data = torque_data[['Torque', 'Power']]
        
        # Load force data from output_path
        force_data = pd.read_csv(force_file)
        # Select columns Fy, Fz, Fx (columns 1, 2, 3)
        force_data = force_data[['Fy', 'Fz', 'Fx']]
        
        # Align by length (use minimum)
        min_len = min(len(torque_data), len(force_data))
        torque_data = torque_data.iloc[:min_len].reset_index(drop=True)
        force_data = force_data.iloc[:min_len].reset_index(drop=True)
        
        # Combine into single dataframe
        df = pd.concat([torque_data[['Torque', 'Power']], force_data[['Fy', 'Fz', 'Fx']]], axis=1)
        
        # drop rows with zero values
        df = df[(df['Torque'] != 0) & (df['Fx'] != 0) & (df['Fy'] != 0) & (df['Fz'] != 0)]
        
        # normalize the data using the global mean and std
        for column in ['Torque', 'Power', 'Fx', 'Fy', 'Fz']:
            df[column] = (df[column] - self.global_mean_std.loc[column, 'global_mean']) / self.global_mean_std.loc[column, 'global_std']
        
        rolling_mean = df[['Torque', 'Power', 'Fx', 'Fy', 'Fz']].rolling(window=self.window_size, min_periods=self.window_size, step=int(self.step_size)).mean()
        rolling_mean = rolling_mean.dropna().reset_index(drop=True)

        # Build vector from metadata: [Vc, ap, fn, D_or_L, HT_or_NHT, hardness]
        # Extract numeric values from columns with units
        vc = float(str(row['Vc (m/min)']).split()[0]) if isinstance(row['Vc (m/min)'], str) else row['Vc (m/min)']
        ap = float(str(row['ap (mm)']).split()[0]) if isinstance(row['ap (mm)'], str) else row['ap (mm)']
        fn = float(str(row['fn (mm/Teeth)']).split()[0]) if isinstance(row['fn (mm/Teeth)'], str) else row['fn (mm/Teeth)']
        
        # Encode categorical labels as numbers (or keep as is if already numeric)
        d_or_l = 1.0 if str(row['label(Dry/Lubricant)']).strip().upper() == 'D' else 0.0
        ht_or_nht = 1.0 if str(row['label2']).strip().upper() == 'HT' else 0.0
        
        hardness = row['hardness']
        
        vector = np.array([
            vc, ap, fn, d_or_l, ht_or_nht, hardness
        ], dtype=np.float32)

        return {
            "x": torch.from_numpy(rolling_mean[['Torque', 'Power']].values).float(),       # (2,W)
            "y": torch.from_numpy(rolling_mean[['Fx', 'Fy', 'Fz']].values).float(),            # (3,W)
            "v": torch.from_numpy(vector).float(),                                      # (6,)
        }

class TorqueForceDataModule(pl.LightningDataModule):
    def __init__(self, data_root_dir, batch_size=2, num_workers=2, window_size=500, step_size=125, train_ratio=0.8, val_ratio=0.1):
        """
        Args:
            data_root_dir: Path to data_root folder containing metadata.csv, trimmedTorquePower/, processedForce/
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for dataloaders
            window_size: Window size for rolling mean
            step_size: Step size for rolling mean
            train_ratio: Ratio of data for training
            val_ratio: Ratio of data for validation
        """
        super().__init__()
        self.data_root_dir = Path(data_root_dir)
        self.metadata_file = self.data_root_dir / "metadata.csv"
        self.global_mean_std_file = self.data_root_dir / "global_mean_std.csv"
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.window_size = window_size
        self.step_size = step_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        
        # Read metadata
        self.metadata_df = pd.read_csv(self.metadata_file)
        # Ensure column names are standardized (remove whitespace)
        self.metadata_df.columns = self.metadata_df.columns.str.strip()
        
        n = len(self.metadata_df)
        assert n >= 3, f"Need >=3 samples in metadata.csv. Found {n}."
        
        n_train = int(self.train_ratio * n)
        n_val = max(1, int(self.val_ratio * n))
        
        self.train_metadata = self.metadata_df.iloc[:n_train].reset_index(drop=True)
        self.val_metadata = self.metadata_df.iloc[n_train:n_train + n_val].reset_index(drop=True)
        self.test_metadata = self.metadata_df.iloc[n_train + n_val:].reset_index(drop=True)

    def setup(self, stage=None):
        self.train_dataset = TorqueForceDataset(
            self.train_metadata, self.global_mean_std_file, self.window_size, self.step_size
        )
        self.val_dataset = TorqueForceDataset(
            self.val_metadata, self.global_mean_std_file, self.window_size, self.step_size
        )
        self.test_dataset = TorqueForceDataset(
            self.test_metadata, self.global_mean_std_file, self.window_size, self.step_size
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=collate_fn)
    def val_dataloader(self): # used training data to test learning ability of the model
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=collate_fn)
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=collate_fn)

# main as test
if __name__ == "__main__":
    data_root_dir = "data_root"
    data_module = TorqueForceDataModule(data_root_dir)
    data_module.setup()
    dataset = data_module.train_dataset
    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"x shape: {sample['x'].shape}, y shape: {sample['y'].shape}, v shape: {sample['v'].shape}")