import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
class SpectrogramDataset(Dataset):
    """
    Dataset pour vocal separation.
    - __init__ :
    - __len__ :
    - __getitem__ :
        1. Tirer une track aléatoirement
        2. Tirer une position de départ valide
        3. Charger la window (memory-mapped)
        4. Retourner (mix_window, vocal_window) en tensors
    """

    def __init__(self, data_train_dir, frame_size=128,stride=1):



        self.frame_size = frame_size
        self.stride = stride
        self.tracks=[]
        for track_folder in data_train_dir.iterdir():
            length = np.load(track_folder / "mixture_linear.npy",mmap_mode="r").shape[1]
            track = {
                "mix": track_folder / "mixture_linear.npy",
                "vocals": track_folder / "vocals_linear.npy",
                "nb_frames": 1 + (length - self.frame_size) // self.stride
            }
            self.tracks.append(track)

    def __len__(self):
        return 2**31 #for ex

    def __getitem__(self, idx):
        track_idx = torch.randint(0, len(self.tracks), (1,)).item()
        track = self.tracks[track_idx]

        pos = torch.randint(0, track["nb_frames"], (1,)).item()
        start = pos * self.stride
        end = start + self.frame_size

        mix = np.load(track["mix"], mmap_mode="r")[:-1, start:end].copy()
        voc = np.load(track["vocals"], mmap_mode="r")[:-1, start:end].copy()

        # (freq, time) -> tensor float32
        mix_t = torch.from_numpy(mix).float()
        voc_t = torch.from_numpy(voc).float()

        return mix_t.unsqueeze(0), voc_t.unsqueeze(0)




