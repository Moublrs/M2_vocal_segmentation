from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import UNet
from dataloader import SpectrogramDataset


# =========================
# GPU
# =========================
GPU_ID = 6
DEVICE = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(GPU_ID)


# =========================
# Hyperparams
# =========================
BATCH_SIZE = 50
LR = 1e-6
NUM_ITERATIONS = 1000000

DATA_DIR = Path.home() / "TP SON UNET" / "data" / "spec_data_linear_ytb_not_resampled"
FRAME_SIZE = 128
STRIDE = 1


# =========================
# Dataset / Loader
# =========================
dataset = SpectrogramDataset(
    data_train_dir=DATA_DIR,
    frame_size=FRAME_SIZE,
    stride=STRIDE
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,          # sampling déjà aléatoire
    num_workers=4,
    pin_memory=True
)


# =========================
# Model / Optim
# =========================
model = UNet().to(DEVICE)
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=LR)


# =========================
# Training loop + tqdm
# =========================
step = 0
pbar = tqdm(total=NUM_ITERATIONS, desc="Training", dynamic_ncols=True)

for mix, voc in loader:
    if step >= NUM_ITERATIONS:
        break

    mix = mix.to(DEVICE)
    voc = voc.to(DEVICE)

    # Forward
    mask = model(mix)
    voc_pred = mask * mix

    # Loss

    loss = torch.mean(torch.abs(voc_pred - voc))

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # tqdm update
    pbar.update(1)
    pbar.set_postfix(loss=f"{loss.item():.6f}")

    step += 1

pbar.close()


# =========================
# Save
# =========================
torch.save(model.state_dict(), "unet_final.pth")
print("Training finished.")
