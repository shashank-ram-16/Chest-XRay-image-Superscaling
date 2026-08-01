"""
Phase 1 - DIV2K pretraining (SRResNet -> SRGAN)
================================================
Produces the generator weights that Phase 2 (ChestX-ray14 fine-tuning) starts from.

Recipe : MSE-only 10 epochs -> adversarial 20 epochs
         g_loss = pixel_MSE + 0.006 * VGG19 + 1e-4 * BCE, 4x downsample
Target : local Windows machine, RTX 3050 6 GB, Intel i5 12th gen

Usage
-----
    python -u train_phase1.py                  # full run, default settings
    python -u train_phase1.py --smoke          # 1+1 epochs, quick sanity check
    python -u train_phase1.py --workers 4      # more decode parallelism
    python -u train_phase1.py --no-cache       # disable RAM cache (low-memory machines)
    python -u train_phase1.py --resume         # continue Stage B from last checkpoint

IMPORTANT (Windows)
-------------------
Everything that spawns DataLoader workers must run under `if __name__ == "__main__":`.
That is why all executable code lives inside main(). Module level holds only
constants, classes and functions - worker processes re-import this file.

Do NOT instantiate models or touch CUDA at module level: each worker would
initialise its own CUDA context and exhaust 6 GB of VRAM.
"""

# %%
import os
import sys
import json
import math
import time
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm


# =====================================================================
# Configuration (module level - workers need to import these)
# =====================================================================

SEED = 42

ROOT    = r"D:\Capstone Project"
HR_DIR  = os.path.join(ROOT, "DIV2K", "DIV2K_train_HR")
VAL_DIR = os.path.join(ROOT, "DIV2K", "DIV2K_valid_HR")
OUT_DIR = os.path.join(ROOT, "checkpoints", "phase1_div2k")

SCALE       = 4
PATCH_SIZE  = 96        # HR patch (SRGAN paper)
BATCH_SIZE  = 16        # fits 6 GB with AMP at 96x96
REPEAT      = 8         # random patches drawn per image per epoch

MSE_EPOCHS  = 10        # Stage A: SRResNet warm-up
GAN_EPOCHS  = 20        # Stage B: adversarial

LR_G_MSE    = 1e-4
LR_G        = 1e-4
LR_D        = 1e-4

VGG_WEIGHT  = 0.006     # perceptual (VGG19 relu5_4, pre-activation)
GAN_WEIGHT  = 1e-4      # adversarial

VAL_N       = 50        # fixed val crops, progress monitoring only
VAL_CROP    = 512       # HR centre-crop size for val PSNR

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp")


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def seed_worker(worker_id):
    ws = SEED + worker_id
    random.seed(ws)
    np.random.seed(ws)
    torch.manual_seed(ws)

def add_noise(x, std=0.02):
    """Instance noise on D inputs — blurs the real/fake boundary so D
    can't separate perfectly and saturate its gradients."""
    return x + torch.randn_like(x) * std


def list_images(d):
    """Extension-filtered listing. Prevents Thumbs.db / desktop.ini from
    crashing a worker mid-epoch with an opaque error."""
    assert os.path.isdir(d), f"Directory not found: {d}"
    files = sorted(f for f in os.listdir(d) if f.lower().endswith(IMG_EXT))
    assert files, f"No images found in: {d}"
    return files


# =====================================================================
# Dataset
# =====================================================================

class DIV2KDataset(Dataset):
    """DIV2K HR patches with bicubic-downsampled LR pairs.

    LR is produced by bicubic /4 of the HR patch, matching the Phase 3A
    evaluation protocol exactly (HR -> bicubic /4 -> LR -> G -> SR).

    cache_images:
        Decoding a full 2040x1356 PNG to take one 96x96 crop costs ~150 ms and
        makes training CPU-bound (~2.7 s/it on a 3050). Caching decoded uint8
        arrays makes every epoch after the first essentially free (~0.1 s/it).
        Costs ~6.5 GB RAM for the full 800-image DIV2K train set.

        Set max_cache lower on constrained machines - uncached images simply
        decode on demand, so a partial cache degrades gracefully.

        Disable entirely when num_workers > 0: each worker process gets its own
        copy of the cache, so N workers means N x 6.5 GB.
    """

    def __init__(self, hr_dir, scale=SCALE, patch_size=PATCH_SIZE, repeat=1,
                 augment=True, cache_images=True, max_cache=800):
        self.hr_dir       = hr_dir
        self.images       = list_images(hr_dir)
        self.scale        = scale
        self.patch_size   = patch_size
        self.repeat       = repeat
        self.augment      = augment
        self.cache_images = cache_images
        self.max_cache    = max_cache
        self.to_tensor    = transforms.ToTensor()
        self._cache       = {}

    def __len__(self):
        return len(self.images) * self.repeat

    def _get_array(self, i):
        if i in self._cache:
            return self._cache[i]

        img = Image.open(os.path.join(self.hr_dir, self.images[i])).convert("RGB")
        w, h = img.size
        img = img.crop((0, 0, w - w % self.scale, h - h % self.scale))
        arr = np.asarray(img)

        if self.cache_images and len(self._cache) < self.max_cache:
            self._cache[i] = arr
        return arr

    def __getitem__(self, idx):
        arr = self._get_array(idx % len(self.images))

        H, W, _ = arr.shape
        assert H >= self.patch_size and W >= self.patch_size, \
            f"Image smaller than patch size: {self.images[idx % len(self.images)]}"

        x = random.randint(0, W - self.patch_size)
        y = random.randint(0, H - self.patch_size)
        hr = Image.fromarray(arr[y:y + self.patch_size, x:x + self.patch_size])

        if self.augment:
            if random.random() < 0.5:
                hr = hr.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() < 0.5:
                hr = hr.transpose(Image.ROTATE_90)

        lr = hr.resize((self.patch_size // self.scale,
                        self.patch_size // self.scale), Image.BICUBIC)

        return self.to_tensor(lr), self.to_tensor(hr)


def build_loader(num_workers, use_cache, batch_size=BATCH_SIZE):
    # cache and multiprocessing do not mix - each worker would hold a full copy
    cache = use_cache and num_workers == 2
    if use_cache and num_workers > 0:
        print("[warn] RAM cache disabled: incompatible with num_workers > 0")

    ds = DIV2KDataset(HR_DIR, SCALE, PATCH_SIZE, repeat=REPEAT,
                      augment=True, cache_images=cache)

    g = torch.Generator()
    g.manual_seed(SEED)

    kwargs = dict(batch_size=batch_size, shuffle=True, drop_last=True,
                  num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                  generator=g)

    if num_workers > 0:
        kwargs.update(worker_init_fn=seed_worker,
                      persistent_workers=True,
                      prefetch_factor=4)

    return ds, DataLoader(ds, **kwargs)


# =====================================================================
# Models
# =====================================================================

class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, num_blocks=16):
        super().__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, 9, 1, 4),
            nn.PReLU(),
        )

        self.residuals = nn.Sequential(
            *[ResidualBlock() for _ in range(num_blocks)]
        )

        self.mid = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
        )

        self.upsample = nn.Sequential(
            nn.Conv2d(64, 256, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.PReLU(),
            nn.Conv2d(64, 256, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.PReLU(),
        )

        self.final = nn.Conv2d(64, 3, 9, 1, 4)

    def forward(self, x):
        x0 = self.initial(x)
        x = self.residuals(x0)
        x = self.mid(x)
        x = x + x0
        x = self.upsample(x)
        return self.final(x)


class Discriminator(nn.Module):
    """Returns LOGITS (no sigmoid) - pair with BCEWithLogitsLoss.

    The sigmoid + BCELoss form used in the original notebook is numerically
    unsafe under mixed precision and can produce NaNs in fp16.
    """

    def __init__(self):
        super().__init__()

        def block(in_c, out_c, stride):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, stride, 1),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),

            block(64, 64, 2),
            block(64, 128, 1),
            block(128, 128, 2),
            block(128, 256, 1),
            block(256, 256, 2),
            block(256, 512, 1),
            block(512, 512, 2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(1024, 1, 1),
        )

    def forward(self, x):
        return self.model(x).view(-1)


class VGGFeatureExtractor(nn.Module):
    """VGG19 features for the perceptual loss.

    Two corrections vs. the original notebook:
      * inputs are ImageNet-normalised (VGG was trained on normalised data;
        feeding raw [0,1] tensors gives a mis-scaled feature space)
      * features taken at index 35 = relu5_4 pre-activation, which is the
        phi_5,4 the SRGAN paper actually specifies
    """

    def __init__(self, layer_idx=35):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features[:layer_idx]
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg.eval()

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        return self.vgg((x - self.mean) / self.std)

    def train(self, mode=True):
        return super().train(False)      # stay frozen in eval mode


# =====================================================================
# Validation
# =====================================================================

def build_val_set(val_dir, n=VAL_N, crop=VAL_CROP, scale=SCALE):
    """Fixed, deterministic centre crops. Progress monitoring only -
    this is NOT the Phase 3A evaluation protocol."""
    to_tensor = transforms.ToTensor()
    files = list_images(val_dir)[:n]

    pairs = []
    for f in files:
        img = Image.open(os.path.join(val_dir, f)).convert("RGB")
        w, h = img.size
        if w < crop or h < crop:
            continue
        left, top = (w - crop) // 2, (h - crop) // 2
        hr = img.crop((left, top, left + crop, top + crop))
        lr = hr.resize((crop // scale, crop // scale), Image.BICUBIC)
        pairs.append((to_tensor(lr), to_tensor(hr)))

    assert pairs, f"Validation set is empty. Check VAL_DIR: {val_dir}"
    return pairs


def psnr(sr, hr):
    mse = torch.mean((sr - hr) ** 2).item()
    return 100.0 if mse == 0 else 10 * math.log10(1.0 / mse)


@torch.no_grad()
def validate(model, val_pairs, device):
    model.eval()
    total = 0.0
    for lr, hr in val_pairs:
        lr = lr.unsqueeze(0).to(device)
        hr = hr.unsqueeze(0).to(device)
        total += psnr(model(lr).clamp(0, 1), hr)
    model.train()
    return total / len(val_pairs)


# =====================================================================
# Stage A - SRResNet warm-up (MSE only)
# =====================================================================

def train_mse(G, loader, val_pairs, device, epochs, use_amp):
    """Pure pixel MSE. Skipping this and going straight to adversarial
    training is the classic way to get a GAN that never converges - the
    discriminator wins immediately."""
    opt = optim.Adam(G.parameters(), lr=LR_G_MSE, betas=(0.9, 0.999))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    criterion = nn.MSELoss()

    history = []
    G.train()

    for epoch in range(epochs):
        t0, running = time.time(), 0.0
        loop = tqdm(loader, desc=f"[SRResNet] Epoch {epoch+1}/{epochs}", leave=False)

        for lr_imgs, hr_imgs in loop:
            lr_imgs = lr_imgs.to(device, non_blocking=True)
            hr_imgs = hr_imgs.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = criterion(G(lr_imgs), hr_imgs)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running += loss.item()
            loop.set_postfix(mse=f"{loss.item():.5f}")

        avg = running / len(loader)
        vp = validate(G, val_pairs, device)
        history.append({"epoch": epoch + 1, "mse": avg, "val_psnr": vp,
                        "sec": round(time.time() - t0, 1)})

        print(f"[SRResNet Pretrain] Epoch [{epoch+1}/{epochs}] | "
              f"Pixel MSE: {avg:.5f} | Val PSNR: {vp:.3f} dB | "
              f"{history[-1]['sec']}s", flush=True)

        torch.save(G.state_dict(), os.path.join(OUT_DIR, "srresnet_div2k.pth"))

    return history


# =====================================================================
# Stage B - Adversarial training
# =====================================================================

def train_gan(G, D, VGG, loader, val_pairs, device, epochs, use_amp,
              history_mse, start_epoch=0, opt_G=None, opt_D=None):
    """g_loss = pixel_MSE + 0.006 * VGG19 + 1e-4 * BCE"""
    opt_G = opt_G or optim.Adam(G.parameters(), lr=LR_G, betas=(0.9, 0.999))
    opt_D = opt_D or optim.Adam(D.parameters(), lr=LR_D, betas=(0.9, 0.999))

    scaler_G = torch.amp.GradScaler("cuda", enabled=use_amp)
    scaler_D = torch.amp.GradScaler("cuda", enabled=use_amp)

    criterion_pixel = nn.MSELoss()
    criterion_content = nn.MSELoss()
    criterion_GAN = nn.BCEWithLogitsLoss()

    history = []
    G.train()
    D.train()

    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        g_sum = d_sum = px_sum = vgg_sum = adv_sum = 0.0

        loop = tqdm(loader, desc=f"[SRGAN] Epoch {epoch+1}/{epochs}", leave=False)

        for lr_imgs, hr_imgs in loop:
            lr_imgs = lr_imgs.to(device, non_blocking=True)
            hr_imgs = hr_imgs.to(device, non_blocking=True)

            bs = hr_imgs.size(0)
            d_real  = torch.full((bs,), 0.9, device=device)   # smoothed, for D
            d_fake  = torch.full((bs,), 0.1, device=device)   # smoothed, for D
            g_valid = torch.ones(bs, device=device)           # NOT smoothed, for G

            # -------- Generator --------
            opt_G.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                sr_imgs = G(lr_imgs)
                pixel_loss = criterion_pixel(sr_imgs, hr_imgs)
                vgg_loss = criterion_content(VGG(sr_imgs.clamp(0, 1)), VGG(hr_imgs))
                gan_loss = criterion_GAN(D(sr_imgs), g_valid)
                g_loss = pixel_loss + VGG_WEIGHT * vgg_loss + GAN_WEIGHT * gan_loss

            scaler_G.scale(g_loss).backward()
            scaler_G.step(opt_G)
            scaler_G.update()

            # -------- Discriminator --------
            opt_D.zero_grad(set_to_none=True)
            opt_D.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                real_loss = criterion_GAN(D(add_noise(hr_imgs)), d_real)
                fake_loss = criterion_GAN(D(add_noise(sr_imgs.detach())), d_fake)
                d_loss = (real_loss + fake_loss) / 2

            scaler_D.scale(d_loss).backward()
            scaler_D.step(opt_D)
            scaler_D.update()

            g_sum += g_loss.item()
            d_sum += d_loss.item()
            px_sum += pixel_loss.item()
            vgg_sum += vgg_loss.item()
            adv_sum += gan_loss.item()

            loop.set_postfix(G=f"{g_loss.item():.4f}", D=f"{d_loss.item():.4f}")

        n = len(loader)
        vp = validate(G, val_pairs, device)
        history.append({
            "epoch": epoch + 1,
            "g_loss": g_sum / n, "d_loss": d_sum / n,
            "pixel": px_sum / n, "vgg": vgg_sum / n, "adv": adv_sum / n,
            "val_psnr": vp, "sec": round(time.time() - t0, 1),
        })

        print(f"Epoch [{epoch+1}/{epochs}] | Avg G: {g_sum/n:.4f} | "
              f"Avg D: {d_sum/n:.4f} | Val PSNR: {vp:.3f} dB | "
              f"{history[-1]['sec']}s", flush=True)

        # resumable per-epoch checkpoint - matters on a laptop-class GPU
        torch.save({
            "epoch": epoch + 1,
            "G": G.state_dict(), "D": D.state_dict(),
            "opt_G": opt_G.state_dict(), "opt_D": opt_D.state_dict(),
            "history_mse": history_mse, "history_gan": history,
        }, os.path.join(OUT_DIR, "srgan_last.pth"))

        with open(os.path.join(OUT_DIR, "phase1_history.json"), "w") as f:
            json.dump({"mse": history_mse, "gan": history}, f, indent=2)

    return history


# =====================================================================
# Plots
# =====================================================================

def plot_curves(history_mse, history_gan, mse_epochs, out_dir):
    import matplotlib
    matplotlib.use("Agg")           # no display needed when run as a script
    import matplotlib.pyplot as plt

    ep_m = [h["epoch"] for h in history_mse]
    ep_g = [h["epoch"] + mse_epochs for h in history_gan]

    fig, ax = plt.subplots(1, 3, figsize=(16, 4))

    ax[0].plot(ep_m, [h["mse"] for h in history_mse], marker="o")
    ax[0].set_title("Stage A: pixel MSE")
    ax[0].set_xlabel("epoch")
    ax[0].grid(alpha=.3)

    ax[1].plot(ep_g, [h["g_loss"] for h in history_gan], label="G loss")
    ax[1].plot(ep_g, [h["d_loss"] for h in history_gan], label="D loss")
    ax[1].set_title("Stage B: adversarial losses")
    ax[1].set_xlabel("epoch")
    ax[1].legend()
    ax[1].grid(alpha=.3)

    ax[2].plot(ep_m, [h["val_psnr"] for h in history_mse], marker="o", label="MSE stage")
    ax[2].plot(ep_g, [h["val_psnr"] for h in history_gan], marker="o", label="GAN stage")
    ax[2].axvline(mse_epochs + 0.5, ls="--", c="gray")
    ax[2].set_title("DIV2K val PSNR")
    ax[2].set_xlabel("epoch")
    ax[2].set_ylabel("dB")
    ax[2].legend()
    ax[2].grid(alpha=.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "phase1_training_curves.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print("Curves ->", path)


# =====================================================================
# Main
# =====================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Phase 1 DIV2K pretraining")
    p.add_argument("--workers", type=int, default=0,
                   help="DataLoader workers. >0 requires running as a script, "
                        "not from a notebook or the VS Code interactive window.")
    p.add_argument("--no-cache", action="store_true",
                   help="Disable the RAM image cache (~6.5 GB for DIV2K train).")
    p.add_argument("--smoke", action="store_true",
                   help="1 MSE + 1 GAN epoch at REPEAT=1, for a quick sanity check.")
    p.add_argument("--resume", action="store_true",
                   help="Resume Stage B from srgan_last.pth.")
    p.add_argument("--skip-mse", action="store_true",
                   help="Skip Stage A and load existing srresnet_div2k.pth.")
    return p.parse_args()


def main():
    global REPEAT

    args = parse_args()
    set_seed(SEED)
    torch.backends.cudnn.benchmark = True        # fixed patch size -> big speedup

    os.makedirs(OUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    print("=" * 62)
    print("Phase 1 - DIV2K pretraining")
    print("=" * 62)
    print("Torch  :", torch.__version__)
    print("Device :", device)
    if device.type == "cuda":
        print("GPU    :", torch.cuda.get_device_name(0))
        print("VRAM   : %.1f GB" % (torch.cuda.get_device_properties(0).total_memory / 1e9))

    try:
        import psutil
        print("RAM    : %.1f GB available" % (psutil.virtual_memory().available / 1e9))
    except ImportError:
        pass

    mse_epochs = 1 if args.smoke else MSE_EPOCHS
    gan_epochs = 1 if args.smoke else GAN_EPOCHS
    if args.smoke:
        REPEAT = 1
        print("[smoke test] REPEAT=1, 1 epoch per stage")

    # ---------------- data ----------------
    print("\nTrain HR:", HR_DIR)
    print("Valid HR:", VAL_DIR)

    train_ds, loader = build_loader(args.workers, use_cache=not args.no_cache)
    val_pairs = build_val_set(VAL_DIR)

    print(f"Train images  : {len(train_ds.images)}")
    print(f"Samples/epoch : {len(train_ds)}  |  Iters/epoch: {len(loader)}")
    print(f"Val crops     : {len(val_pairs)}")
    print(f"Workers       : {args.workers}  |  RAM cache: {train_ds.cache_images}")

    # ---------------- models (inside main - never at module level) ----------------
    G = Generator().to(device)
    D = Discriminator().to(device)
    VGG = VGGFeatureExtractor().to(device)

    n_params = lambda m: sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"\nGenerator params     : {n_params(G):,}")
    print(f"Discriminator params : {n_params(D):,}\n")

    srresnet_path = os.path.join(OUT_DIR, "srresnet_div2k.pth")
    last_path = os.path.join(OUT_DIR, "srgan_last.pth")

    # ---------------- Stage A ----------------
    history_mse = []
    if args.resume and os.path.exists(last_path):
        print("Resuming - Stage A skipped")
    elif args.skip_mse:
        assert os.path.exists(srresnet_path), f"Not found: {srresnet_path}"
        G.load_state_dict(torch.load(srresnet_path, map_location=device))
        print("Stage A skipped, loaded", srresnet_path)
    else:
        print("-" * 62)
        print(f"STAGE A - SRResNet warm-up ({mse_epochs} epochs, MSE only)")
        print("-" * 62)
        history_mse = train_mse(G, loader, val_pairs, device, mse_epochs, use_amp)
        print("Saved ->", srresnet_path)

    # ---------------- Stage B ----------------
    start_epoch, opt_G, opt_D = 0, None, None

    if args.resume and os.path.exists(last_path):
        ckpt = torch.load(last_path, map_location=device)
        G.load_state_dict(ckpt["G"])
        D.load_state_dict(ckpt["D"])
        opt_G = optim.Adam(G.parameters(), lr=LR_G, betas=(0.9, 0.999))
        opt_D = optim.Adam(D.parameters(), lr=LR_D, betas=(0.9, 0.999))
        opt_G.load_state_dict(ckpt["opt_G"])
        opt_D.load_state_dict(ckpt["opt_D"])
        start_epoch = ckpt["epoch"]
        history_mse = ckpt.get("history_mse", [])
        print(f"Resumed from epoch {start_epoch}")
    elif not args.skip_mse or history_mse:
        # start the adversarial stage from the MSE-warmed generator
        if os.path.exists(srresnet_path):
            G.load_state_dict(torch.load(srresnet_path, map_location=device))

    print("-" * 62)
    print(f"STAGE B - Adversarial ({gan_epochs} epochs)")
    print(f"g_loss = pixel_MSE + {VGG_WEIGHT} * VGG19 + {GAN_WEIGHT} * BCE")
    print("-" * 62)

    history_gan = train_gan(G, D, VGG, loader, val_pairs, device, gan_epochs,
                            use_amp, history_mse, start_epoch, opt_G, opt_D)

    # ---------------- hand-off to Phase 2 ----------------
    final_path = os.path.join(OUT_DIR, "srgan_div2k_ready.pth")
    torch.save(G.state_dict(), final_path)
    torch.save(D.state_dict(), os.path.join(OUT_DIR, "srgan_div2k_D.pth"))

    print("\n" + "=" * 62)
    print("Generator     ->", final_path)
    print("Discriminator ->", os.path.join(OUT_DIR, "srgan_div2k_D.pth"))
    if history_gan:
        print("Final val PSNR: %.3f dB" % history_gan[-1]["val_psnr"])
    print("=" * 62)

    if history_mse and history_gan:
        plot_curves(history_mse, history_gan, mse_epochs, OUT_DIR)


if __name__ == "__main__":
    main()
