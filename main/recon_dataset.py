import os
import re
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset
from tqdm import tqdm
from utils import normalize


class ReconstructionDataset(Dataset):
    def __init__(self, root, config):
        self.data = glob(os.path.join(root, "**npz"), recursive=True)
        self.min_level_db = config.min_level_db
        self.split_frame = config.split_frame
        self.train_frame = config.train_frame

    def __getitem__(self, idx):
        npz = np.load(self.data[idx])
        r_start = np.random.randint(0, self.split_frame - self.train_frame)
        r_end = r_start + self.train_frame
        mels, pianorolls = npz["m"], npz["p"]
        mel = mels[:, r_start:r_end]
        pianoroll = pianorolls[:, r_start:r_end]

        p_max = np.max(pianoroll)
        if p_max > 1:
            pianoroll /= np.max(pianoroll)

        mel = normalize(mel, self.min_level_db)
        r = np.random.randint(0, 1)
        # random choice inpainting or generation
        if r == 1:
            start = np.random.randint(0, 32)
            end = start + np.random.randint(32, self.segment_frame - 32)
            cond = np.copy(mel)
            cond[:, start:end] = pianoroll[:, start:end]
        else:
            cond = pianoroll

        mel = torch.from_numpy(mel).unsqueeze(0).float()
        cond = torch.from_numpy(cond).unsqueeze(0).float()
        return mel, cond

    def __len__(self):
        return len(self.data)
