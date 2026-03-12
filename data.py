from torch.utils.data import Dataset, DataLoader
import torch
from pathlib import Path
import pytorch_lightning as pl
import pandas as pd


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
    def __init__(self, signal_files, vector_files, global_mean_std_file, window_size=500, step_size=125):
        self.global_mean_std = pd.read_csv(global_mean_std_file).set_index('column')
        self.window_size = window_size
        self.step_size = step_size
        self.signal_files = signal_files
        self.vector_files = vector_files

    def __len__(self):
        return len(self.signal_files)
    
    def __getitem__(self, idx):
        df = pd.read_csv(self.signal_files[idx])
        # drop rows with either df['Torque'] == 0 or df['FX'] == 0 or df['FY'] == 0 or df['FZ'] == 0
        df = df[(df['Torque'] != 0) & (df['FX'] != 0) & (df['FY'] != 0) & (df['FZ'] != 0)]
        # normalize the data using the global mean and std
        for column in ['Torque', 'FX', 'FY', 'FZ']:
            df[column] = (df[column] - self.global_mean_std.loc[column, 'global_mean']) / self.global_mean_std.loc[column, 'global_std']
        
        rolling_mean = df[['Torque', 'FX', 'FY', 'FZ']].rolling(window=self.window_size, min_periods=self.window_size, step=int(self.step_size)).mean()
        rolling_mean = rolling_mean.dropna().reset_index(drop=True)

        # load the corresponding vector file TFVC40.csv -> vector40.csv
        vector_file = self.vector_files[idx]
        vector_df = pd.read_csv(vector_file, header=None)

        return {
            "x": torch.from_numpy(rolling_mean[['Torque']].values).float(),       # (1,W)
            "y": torch.from_numpy(rolling_mean[['FX', 'FY', 'FZ']].values).float(),            # (3,W)
            "v": torch.from_numpy(vector_df.values).squeeze().float(),                                      # (6,)
        }

class TorqueForceDataModule(pl.LightningDataModule):
    def __init__(self, signals_dir, vector_dir, global_mean_std_file, batch_size=2, num_workers=2, window_size=500, step_size=125):
        super().__init__()
        self.signals_dir = Path(signals_dir)
        self.vector_dir = Path(vector_dir)
        self.global_mean_std_file = global_mean_std_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.window_size = window_size
        self.step_size = step_size
        
        self.signal_files = sorted(self.signals_dir.glob('*.csv'))

        n = len(self.signal_files)
        assert n >= 3, f"Need >=3 TFVC*.csv files in {self.signals_dir}. Found {n}."
        n_train = int(0.8 * n)
        n_val = max(1, int(0.1 * n))
        
        self.train_signal_files = self.signal_files[:n_train]
        self.val_signal_files = self.signal_files[n_train:n_train + n_val]
        self.test_signal_files = self.signal_files[n_train + n_val:]

        self.train_vector_files = [self.vector_dir / (file.stem.replace('TFVC', 'vector') + '.csv') for file in self.train_signal_files]
        self.val_vector_files = [self.vector_dir / (file.stem.replace('TFVC', 'vector') + '.csv') for file in self.val_signal_files]
        self.test_vector_files = [self.vector_dir / (file.stem.replace('TFVC', 'vector') + '.csv') for file in self.test_signal_files]



    def setup(self, stage=None):
        self.train_dataset = TorqueForceDataset(self.train_signal_files, self.train_vector_files, self.global_mean_std_file, self.window_size, self.step_size)
        self.val_dataset = TorqueForceDataset(self.val_signal_files, self.val_vector_files, self.global_mean_std_file, self.window_size, self.step_size)
        self.test_dataset = TorqueForceDataset(self.test_signal_files, self.test_vector_files, self.global_mean_std_file, self.window_size, self.step_size)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=collate_fn)
    def val_dataloader(self): # used training data to test learning ability of the model
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=collate_fn)
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=collate_fn)

# main as test
if __name__ == "__main__":
    signals_dir = "data_root/signals"
    vector_dir = "data_root/vectors"
    global_mean_std_file = "data_root/global_mean_std.csv"
    data_module = TorqueForceDataModule(signals_dir, vector_dir, global_mean_std_file)
    data_module.setup()
    dataset = data_module.train_dataset
    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"x shape: {sample['x'].shape}, y shape: {sample['y'].shape}, v shape: {sample['v'].shape}")