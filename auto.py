import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
import torch
import pytorch_lightning as pl
import albumentations as A
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import argparse

torch.set_float32_matmul_precision('medium')
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# ----------------------------- CONFIG -----------------------------
img_size = 256
batch_size = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ----------------------------- ARGUMENTS -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train UNet for regression or segmentation with fusion scores.")
    parser.add_argument("--mode", type=str, default="regression", choices=["regression", "segmentation"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--csv_path", type=str, default="/content/drive/MyDrive/Colab_Files/fusion_results.csv")
    parser.add_argument("--fusion_pred_csv", type=str, default="/content/drive/MyDrive/fusion_predictions.csv")
    parser.add_argument("--img_dir", type=str, default="/content/drive/MyDrive/FoodSeg103-256/Images/img_dir/train/")
    parser.add_argument("--ann_dir", type=str, default="/content/drive/MyDrive/FoodSeg103-256/Images/ann_dir/train/")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--max_samples", type=int, default=2000)
    parser.add_argument("--encoder", type=str, default="mobilenet_v2")
    parser.add_argument("--checkpoint_dir", type=str, default="/content/drive/MyDrive/fusion_checkpoints/")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n_classes_seg", type=int, default=104)
    return parser.parse_args()

# ----------------------------- HANDLE COLAB OR SCRIPT -----------------------------
if "google.colab" in sys.modules:
    class Args:
        mode = "regression"
        epochs = 50
        csv_path = "/content/drive/MyDrive/Colab_Files/fusion_results.csv"
        fusion_pred_csv = "/content/drive/MyDrive/fusion_predictions.csv"
        img_dir = "/content/drive/MyDrive/FoodSeg103-256/Images/img_dir/train/"
        ann_dir = "/content/drive/MyDrive/FoodSeg103-256/Images/ann_dir/train/"
        limit = 1000
        max_samples = 2000
        encoder = "mobilenet_v2"
        checkpoint_dir = "/content/drive/MyDrive/fusion_checkpoints/"
        lr = 3e-4
        n_classes_seg = 104
    args = Args()
else:
    args = parse_args()

os.makedirs(args.checkpoint_dir, exist_ok=True)

# ----------------------------- REGRESSION DATASET -----------------------------
class FusionDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None, limit=None):
        df = pd.read_csv(csv_path)
        if limit:
            df = df.head(limit)
        img_files_set = set(os.listdir(img_dir))
        valid_rows = []
        for _, row in df.iterrows():
            image_id_str = f"{int(row['image_id']):08d}.bmp"
            if image_id_str in img_files_set:
                valid_rows.append({"image_id": image_id_str, "fusion_score": row["fusion_score"]})
        self.df = pd.DataFrame(valid_rows)
        print(f"Regression dataset size after filtering missing images: {len(self.df)}")
        if len(self.df) == 0:
            raise ValueError("No matching images found!")
        self.img_dir = img_dir
        self.augment = A.Compose([A.Resize(img_size, img_size), A.Normalize()]) if transform is None else transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["image_id"])
        image = np.array(Image.open(img_path).convert("RGB"))
        target = np.float32(row["fusion_score"])
        image = self.augment(image=image)["image"]
        image = torch.tensor(image.transpose(2,0,1), dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)
        return image, target.unsqueeze(0)

# ----------------------------- SEGMENTATION DATASET -----------------------------
class FusionSegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, fusion_dict, transform=None, max_samples=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.fusion_dict = fusion_dict
        self.transform = transform
        self.image_ids = [f.split(".")[0] for f in os.listdir(img_dir) if f.endswith(".bmp")]
        if max_samples:
            self.image_ids = self.image_ids[:max_samples]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.img_dir, img_id + ".bmp")
        mask_path = os.path.join(self.mask_dir, img_id + ".bmp")
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))
        if self.transform:
            aug = self.transform(image=image, mask=mask)
            image, mask = aug["image"], aug["mask"]
        image = A.Normalize()(image=image)["image"].transpose(2,0,1)
        fusion_score = self.fusion_dict.get(img_id + ".bmp", 1.0)
        return (
            torch.tensor(image, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.long),
            torch.tensor(fusion_score, dtype=torch.float32)
        )

# ----------------------------- MODELS -----------------------------
class UNetFusion(pl.LightningModule):
    def __init__(self, encoder_name="mobilenet_v2", learning_rate=3e-4):
        super().__init__()
        self.save_hyperparameters()
        self.unet = smp.create_model(arch="unet", encoder_name=encoder_name, encoder_weights="imagenet", in_channels=3, classes=1)
        self.loss_fn = torch.nn.MSELoss()
    def forward(self, x):
        return self.unet(x)
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).mean(dim=(2,3))
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).mean(dim=(2,3))
        loss = self.loss_fn(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

class FusionWeightedUNet(pl.LightningModule):
    def __init__(self, num_classes=args.n_classes_seg, encoder_name="efficientnet-b0", lr=args.lr):
        super().__init__()
        self.unet = smp.create_model(arch="unet", encoder_name=encoder_name, encoder_weights="imagenet", in_channels=3, classes=num_classes)
        self.lr = lr
        self.num_classes = num_classes
        self.loss_fn = torch.nn.CrossEntropyLoss()
    def forward(self, x):
        return self.unet(x)
    def training_step(self, batch, batch_idx):
        images, masks, fusion_scores = batch
        masks = masks.long().to(images.device)
        logits = self(images)
        loss = self.loss_fn(logits, masks)
        weighted_loss = (loss * fusion_scores.mean()).mean()
        self.log("train_loss", weighted_loss, prog_bar=True)
        return weighted_loss
    def validation_step(self, batch, batch_idx):
        images, masks, fusion_scores = batch
        masks = masks.long().to(images.device)
        logits = self(images)
        loss = self.loss_fn(logits, masks)
        weighted_loss = (loss * fusion_scores.mean()).mean()
        self.log("val_loss", weighted_loss, prog_bar=True)

        preds = torch.argmax(logits, dim=1)
        preds_onehot = torch.nn.functional.one_hot(preds, num_classes=self.num_classes).permute(0,3,1,2)
        masks_onehot = torch.nn.functional.one_hot(masks, num_classes=self.num_classes).permute(0,3,1,2)

        tp, fp, fn, tn = smp.metrics.get_stats(preds_onehot, masks_onehot, mode="multiclass", num_classes=self.num_classes)
        self.log("val_miou_unweighted", smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro"))
        self.log("val_f1_unweighted", smp.metrics.f1_score(tp, fp, fn, tn, reduction="macro"))
        self.log("val_miou_weighted", smp.metrics.iou_score(tp, fp, fn, tn, reduction="weighted"))
        self.log("val_f1_weighted", smp.metrics.f1_score(tp, fp, fn, tn, reduction="weighted"))
    def validation_epoch_end(self, outputs):
        print(f"\n--- Epoch {self.current_epoch} Validation Metrics ---")
        for key in ["val_loss","val_miou_unweighted","val_f1_unweighted","val_miou_weighted","val_f1_weighted"]:
            if key in self.trainer.logged_metrics:
                print(f"{key}: {self.trainer.logged_metrics[key]:.4f}")
        print("-------------------------------------------\n")
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

# ----------------------------- MAIN -----------------------------
if __name__ == "__main__":
    if args.mode == "regression":
        dataset = FusionDataset(args.csv_path, args.img_dir, limit=args.limit)
        train_dl = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        val_dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

        model = UNetFusion(encoder_name=args.encoder)
        checkpoint_callback = ModelCheckpoint(dirpath=args.checkpoint_dir, filename="fusion_best", save_top_k=1, monitor="val_loss", mode="min", save_last=True)
        early_stop_callback = EarlyStopping(monitor="val_loss", patience=10, mode="min")

        trainer = pl.Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu",
                             devices=1, max_epochs=args.epochs,
                             precision="16-mixed",
                             enable_checkpointing=True,
                             callbacks=[checkpoint_callback, early_stop_callback])
        last_ckpt_path = os.path.join(args.checkpoint_dir, "last.ckpt")
        if os.path.exists(last_ckpt_path):
            print(f"Resuming training from checkpoint: {last_ckpt_path}")
            trainer.fit(model, train_dl, val_dl, ckpt_path=last_ckpt_path)
        else:
            print(f"Starting regression training from scratch for {args.epochs} epochs")
            trainer.fit(model, train_dl, val_dl)

    elif args.mode == "segmentation":
        fusion_pred_df = pd.read_csv(args.fusion_pred_csv)
        fusion_dict = {f"{i:08d}.bmp": row.get("fusion_score", row.get("pred", 1.0))
                       for i, row in fusion_pred_df.iterrows()}

        transform = A.Compose([A.Resize(img_size, img_size)])
        train_dataset = FusionSegmentationDataset(args.img_dir, args.ann_dir, fusion_dict, transform, args.max_samples)
        val_dataset = FusionSegmentationDataset(args.img_dir, args.ann_dir, fusion_dict, transform, args.max_samples)

        train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

        model = FusionWeightedUNet(num_classes=args.n_classes_seg, encoder_name=args.encoder)
        checkpoint_callback = ModelCheckpoint(dirpath=args.checkpoint_dir, filename="best_model", save_top_k=1, monitor="val_miou_weighted", mode="max", save_last=True)
        early_stop_callback = EarlyStopping(monitor="val_miou_weighted", patience=10, mode="max")

        trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=args.epochs, precision="16-mixed",
                             callbacks=[checkpoint_callback, early_stop_callback], log_every_n_steps=10)
        print(f"Starting segmentation training for {args.epochs} epochs")
        trainer.fit(model, train_dl, val_dl)

