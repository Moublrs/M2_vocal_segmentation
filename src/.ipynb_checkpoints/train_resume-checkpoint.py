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
BATCH_SIZE = 512
LR = 1e-4
NUM_ITERATIONS = 50000

DATA_DIR = Path.home() / "TP SON UNET" / "data" / "spec_data_linear" / "train"
FRAME_SIZE = 128
STRIDE = 1

# Plateau
PATIENCE = PATIENCE = 10_000
LR_FACTOR = 0.5
MIN_LR = 1e-7

# Resume
LOAD_PATH = Path("./models/unet_final.pth")  # None pour partir de zéro

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
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# =========================
# Model / Optim
# =========================
model = UNet().to(DEVICE)

if LOAD_PATH and LOAD_PATH.exists():
    model.load_state_dict(torch.load(LOAD_PATH, map_location=DEVICE))
    print(f"Loaded: {LOAD_PATH}")

model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# =========================
# Training loop
# =========================
step = 0
best_loss = float("inf")
steps_no_improve = 0
loss_history = []

pbar = tqdm(total=NUM_ITERATIONS, desc="Training", dynamic_ncols=True)

for mix, voc in loader:
    if step >= NUM_ITERATIONS:
        break
    
    mix = mix.to(DEVICE)
    voc = voc.to(DEVICE)

    # Forward
    mask = model(mix)
    voc_pred = mask * mix
    loss = torch.mean(torch.abs(voc_pred - voc))

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Track loss (moyenne glissante)
    loss_val = loss.item()
    loss_history.append(loss_val)
    if len(loss_history) > 1000:
        loss_history.pop(0)
    avg_loss = sum(loss_history) / len(loss_history)

    # Plateau detection
    if avg_loss < best_loss * 0.99:
        best_loss = avg_loss
        steps_no_improve = 0
    else:
        steps_no_improve += 1

    if steps_no_improve >= PATIENCE:
        current_lr = optimizer.param_groups[0]["lr"]
        if current_lr > MIN_LR:
            new_lr = max(current_lr * LR_FACTOR, MIN_LR)
            for g in optimizer.param_groups:
                g["lr"] = new_lr
            tqdm.write(f"[Plateau] LR: {current_lr:.1e} -> {new_lr:.1e}")
            steps_no_improve = 0
        else:
            tqdm.write("[Plateau] Early stopping")
            break

    # tqdm
    lr = optimizer.param_groups[0]["lr"]
    pbar.update(1)
    pbar.set_postfix(loss=f"{loss_val:.6f}", avg=f"{avg_loss:.6f}", lr=f"{lr:.1e}")
    step += 1

pbar.close()

# =========================
# Save
# =========================
torch.save(model.state_dict(), "unet_final_fine_tuned.pth")
print("Training finished.")