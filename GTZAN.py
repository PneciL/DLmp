import os, torch

import soundfile as sf
import torch.nn.functional as F

from torch.utils.data import Dataset

class GTZAN(Dataset):
    def __init__(self, root_dir, segment_length=66150):
        self.root_dir = root_dir
        self.segment_length = segment_length
        self.filepaths = []
        self.labels = []
        self.genres = sorted(os.listdir(root_dir))

        for i, genre in enumerate(self.genres):
            genre_path = os.path.join(root_dir, genre)

            for filename in os.listdir(genre_path):
                if filename.endswith('.wav'):
                    self.filepaths.append(os.path.join(genre_path, filename))
                    self.labels.append(i)

    def __len__(self):

        return len(self.filepaths)

    def __getitem__(self, idx):
        try:
            data, sample_rate = sf.read(self.filepaths[idx])
            waveform = torch.from_numpy(data).float()
    
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            else:
                waveform = waveform.T
    
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
    
            if waveform.shape[1] > self.segment_length:
                max_start = waveform.shape[1] - self.segment_length
                start = torch.randint(0, max_start, (1,)).item()
                waveform = waveform[:, start:start + self.segment_length]
    
            else:
                padding = self.segment_length - waveform.shape[1]
                waveform = F.pad(waveform, (0, padding))
    
            return waveform, self.labels[idx]
            
        except Exception as e:
            print(f"Skippping corrupt file {self.filepaths[idx]}: {e}")

            new_idx = torch.randint(0, len(self, (1,)).item())

            return self.__getitem__(new_idx)