import torch
import torch.nn.functional as F
import torchaudio.transforms as T
from torch.utils.data import Dataset
from datasets import load_dataset

N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
SAMPLE_RATE = 22050
N_FRAMES = 128  # 66150 / 512 ≈ 130 frames, cropped to 128

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']

class GTZAN(Dataset):
    def __init__(self, split="train", segment_length=66150):
        self.segment_length = segment_length
        # downloads automatically from HuggingFace on first run
        self.data = load_dataset("marsyas/gtzan", "all", split=split, trust_remote_code=True)

        self.mel_transform = T.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_fft=N_FFT,
            hop_length=HOP_LENGTH, n_mels=N_MELS, power=2.0
        )
        self.amp_to_db = T.AmplitudeToDB(stype='power', top_db=80)
        self.resample_cache = {}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            sample = self.data[idx]
            audio = sample["audio"]
            waveform = torch.tensor(audio["array"], dtype=torch.float32).unsqueeze(0)
            sample_rate = audio["sampling_rate"]
            g = sample["genre"]
            label = g if isinstance(g, int) else GENRES.index(g)

            # resample if needed
            if sample_rate != SAMPLE_RATE:
                if sample_rate not in self.resample_cache:
                    self.resample_cache[sample_rate] = T.Resample(sample_rate, SAMPLE_RATE)
                waveform = self.resample_cache[sample_rate](waveform)

            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            if waveform.shape[1] >= self.segment_length:
                start = torch.randint(0, waveform.shape[1] - self.segment_length + 1, (1,)).item()
                waveform = waveform[:, start:start + self.segment_length]
            else:
                waveform = F.pad(waveform, (0, self.segment_length - waveform.shape[1]))

            mel = self.mel_transform(waveform) # [1, N_MELS, T]
            mel = self.amp_to_db(mel) # dB scale
            mel = mel - mel.max() # ref to max -> [-80, 0]
            mel = mel / 40 + 1 # normalize to [-1, 1]

            t = mel.shape[2]
            mel = mel[:, :, :N_FRAMES] if t >= N_FRAMES else F.pad(mel, (0, N_FRAMES - t))

            return mel, label

        except Exception as e:
            print(f"Skipping idx {idx}: {e}")
            return self.__getitem__(torch.randint(0, len(self), (1,)).item())
