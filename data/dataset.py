import os
import glob
import re
import pandas as pd
import numpy as np 
import torch
from torch.utils.data import Dataset

class PendulumDataset(Dataset):
    def __init__(self, data_dir):
        self.files = glob.glob(os.path.join(data_dir, "*.csv"))
        if not self.files:
            raise FileNotFoundError(f"No CSV files found in {data_dir}")
            
        print(f"Dataset found {len(self.files)} files in {data_dir}")
        
        self.regex_k1 = re.compile(r"K1=(\d+\.\d+)")
        self.regex_k2 = re.compile(r"K2=(\d+\.\d+)")
        self.regex_n  = re.compile(r"N=(\d+\.\d+)")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filepath = self.files[idx]
        filename = os.path.basename(filepath)
        
        # 1. 解析标签
        k1, k2, n_val = 0.0, 0.0, 0.0
        try:
            if match := self.regex_k1.search(filename): k1 = float(match.group(1))
            if match := self.regex_k2.search(filename): k2 = float(match.group(1))
            if match := self.regex_n.search(filename):  n_val = float(match.group(1))
        except Exception:
            pass 
        
        label = torch.tensor([k1, k2, n_val], dtype=torch.float32)
        
        # 2. 读取两列数据 
        try:
            df = pd.read_csv(filepath)
            if 'Angle_rad' not in df.columns:
                raise ValueError(f"Column 'Angle_rad' missing in {filename}")
            angle_seq = df['Angle_rad'].values.astype(np.float32)
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            angle_seq = np.zeros(72000, dtype=np.float32)
        seq_tensor = torch.tensor(angle_seq).unsqueeze(-1)
        
        return seq_tensor, label