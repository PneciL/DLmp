import os, torch

import soundfile as sf
import torch.nn.functional as F
import torchaudio.transforms as T

from torch.utils.data import Dataset

N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
SAMPLE_RATE = 22050
N_FRAMES = 128  # 66150 / 512 ≈ 130 frames, cropped to 128

class GTZAN(Dataset):
    def __init__(self, root_dir, segment_length=66560):
        self.root_dir = root_dir
        self.segment_length = segment_length
        self.filepaths = []
        self.labels = []

        self.genres = sorted([g for g in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, g)) and not g.startswith('.')])

        for i, genre in enumerate(self.genres):
            genre_path = os.path.join(root_dir, genre)
            for filename in os.listdir(genre_path):
                if filename.endswith('.wav') and not filename.startswith('.'):
                    self.filepaths.append(os.path.join(genre_path, filename))
                    self.labels.append(i)

        self.mel_transform = T.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_fft=N_FFT,
            hop_length=HOP_LENGTH, n_mels=N_MELS, power=2.0
        )
        self.amp_to_db = T.AmplitudeToDB(stype='power', top_db=80)
        self.resample_cache = {}

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        try:
            data, sr = sf.read(self.filepaths[idx])
            waveform = torch.from_numpy(data).float()

            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            else:
                waveform = waveform.T
            
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            if sr != SAMPLE_RATE:
                if sr not in self.resample_cache:
                    self.resample_cache[sr] = T.Resample(sr, SAMPLE_RATE)
                waveform = self.resample_cache[sr](waveform)
                
            if waveform.shape[1] >= self.segment_length:
                max_start = waveform.shape[1] - self.segment_length
                start = torch.randint(0, max_start, (1,)).item()
                waveform = waveform[:, start:start + self.segment_length]
            else:
                padding = self.segment_length - waveform.shape[1]
                waveform = F.pad(waveform, (0, padding))

            mel = self.mel_transform(waveform) # [1, N_MELS, T]
            mel = self.amp_to_db(mel) # dB scale
            mel = mel - mel.max() # ref to max -> [-80, 0]
            mel = mel / 40 + 1 # normalize to [-1, 1]

            t = mel.shape[2]
            if t > N_FRAMES:
                mel = mel[:, :, :N_FRAMES]
            elif t < N_FRAMES:
                mel = F.pad(mel, (0, N_FRAMES - t))

            return mel, self.labels[idx]

        except Exception as e:
            print(f"Skipping idx {idx}: {e}")
            return self.__getitem__(torch.randint(0, len(self), (1,)).item())