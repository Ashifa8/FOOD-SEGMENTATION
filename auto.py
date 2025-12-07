import os
import sys
import numpy as np
import pandas as pd
import torch
torch.set_float32_matmul_precision('medium')

import pytorch_lightning as pl
import albumentations as A
from PIL import Image
import argparse
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# -----------------------------
# CONFIG
# -----------------------------
n_classes = 1
img_size = 256
batch_size = 4

# -----------------------------
# ARGUMENTS
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train UNet model on fusion CSV.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--csv_path", type=str, default="/content/drive/MyDrive/Colab_Files/fusion_results.csv")
    parser.add_argument("--img_dir", type=str, default="/content/drive/MyDrive/FoodSeg103-256/Images/img_dir/train/")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--encoder", type=str, default="mobilenet_v2")
    parser.add_argument("--checkpoint_dir", type=str, default="/content/drive/MyDrive/fusion_checkpoints/")
    return parser.parse_args()

# -----------------------------
# HANDLE COLAB OR NORMAL SCRIPT
# -----------------------------
if "google.colab" in sys.modules:
    class Args:
        epochs = 50
        csv_path = "/content/drive/MyDrive/Colab_Files/fusion_results.csv"
        img_dir = "/content/drive/MyDrive/FoodSeg103-256/Images/img_dir/train/"
        limit = 1000
        encoder = "mobilenet_v2"
        checkpoint_dir = "/content/drive/MyDrive/fusion_checkpoints/"
    args = Args()
else:
    args = parse_args()

# -----------------------------
# DEVICE
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# -----------------------------
# DATASET
# -----------------------------
class FusionDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None, limit=None):
        df = pd.read_csv(csv_path)
        if limit:
            df = df.head(limit)

        # Get all filenames in folder
        img_files_set = set(os.listdir(img_dir))

        # Keep only rows that have a corresponding image
        valid_rows = []
        for _, row in df.iterrows():
            image_id_str = f"{int(row['image_id']):08d}.bmp"
            if image_id_str in img_files_set:
                valid_rows.append({
                    "image_id": image_id_str,
                    "fusion_score": row["fusion_score"]
                })

        self.df = pd.DataFrame(valid_rows)
        print(f"Dataset size after removing missing images: {len(self.df)}")
        if len(self.df) == 0:
            raise ValueError("No matching images found!")

        self.img_dir = img_dir
        self.augment = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(),
        ]) if transform is None else transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["image_id"])
        image = np.array(Image.open(img_path).convert("RGB"))
        target = np.float32(row["fusion_score"])

        image = self.augment(image=image)["image"]
        image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)

        return image, target.unsqueeze(0)

# -----------------------------
# DATALOADERS
# -----------------------------
train_dataset = FusionDataset(args.csv_path, args.img_dir, limit=args.limit)
train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
val_dl   = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

# -----------------------------
# MODEL
# -----------------------------
class UNetFusion(pl.LightningModule):
    def __init__(self, encoder_name="mobilenet_v2", learning_rate=3e-4):
        super().__init__()
        self.save_hyperparameters()
        self.unet = smp.create_model(
            arch="unet",
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=3,
            classes=1
        )
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, x):
        return self.unet(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).mean(dim=(2, 3))
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).mean(dim=(2, 3))
        loss = self.loss_fn(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

# -----------------------------
# CHECKPOINTS
# -----------------------------
os.makedirs(args.checkpoint_dir, exist_ok=True)

checkpoint_callback = ModelCheckpoint(
    dirpath=args.checkpoint_dir,
    filename="fusion_best",
    save_top_k=1,
    monitor="val_loss",
    mode="min",
    save_last=True
)

early_stop_callback = EarlyStopping(
    monitor="val_loss",
    patience=10,
    mode="min"
)

# -----------------------------
# TRAINER
# -----------------------------
model = UNetFusion(encoder_name=args.encoder)

trainer = pl.Trainer(
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
    max_epochs=args.epochs,
    precision="16-mixed",
    logger=False,
    enable_checkpointing=True,
    enable_progress_bar=True,
    callbacks=[checkpoint_callback, early_stop_callback]
)

# -----------------------------
# RESUME OR START
# -----------------------------
last_ckpt_path = os.path.join(args.checkpoint_dir, "last.ckpt")

if os.path.exists(last_ckpt_path):
    print(f"Resuming training from checkpoint: {last_ckpt_path}")
    trainer.fit(model, train_dl, val_dl, ckpt_path=last_ckpt_path)
else:
    print(f"Starting training from scratch for {args.epochs} epochs")
    trainer.fit(model, train_dl, val_dl)
