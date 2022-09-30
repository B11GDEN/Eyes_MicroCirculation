import argparse
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks import RichProgressBar
from segmentation_models_pytorch import Unet
from segmentation_models_pytorch.losses import DiceLoss, SoftBCEWithLogitsLoss
from sklearn.model_selection import KFold
from torch.optim import AdamW
from torch.utils.data import DataLoader, ConcatDataset
from torchmetrics.functional import dice as dice_metric

from dataset import Dataset, get_train_transform, get_test_transform


def parse_args():
    parser = argparse.ArgumentParser(description='kfold train U-net on the eyes data')

    # data parameters
    parser.add_argument('-src', help='path to data', default='../datasets/EYES_MICRO_AUX')
    parser.add_argument('-num_folds', help='how many folds use in cross-validation', type=int, default=5)

    # dataset and pipeline parameters
    parser.add_argument('-crop', help='crop size in train transform', type=int, default=512)
    parser.add_argument('-bs', help='batch size', type=int, default=4)
    parser.add_argument('-accumulate', help='how many batches accumulate before optimizer step', type=int, default=1)
    parser.add_argument('-workers', type=int, default=0)
    parser.add_argument('-epochs', type=int, default=50)

    # model parameters from segmentation models pytorch
    parser.add_argument('-encoder_name', default='efficientnet-b3')
    parser.add_argument('-encoder_weights', default='imagenet')

    return parser.parse_args()


# define the LightningModule
class LitSegmenter(pl.LightningModule, Unet):
    def __init__(self, loss, aux_loss, **kwargs):
        pl.LightningModule.__init__(self)
        Unet.__init__(self, **kwargs)
        self.loss = loss
        self.aux_loss = aux_loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)
        loss = self.loss(pred, y.long().unsqueeze(dim=1))
        aux_loss = self.aux_loss(pred, y.long().unsqueeze(dim=1))
        total_loss = loss + 0.3 * aux_loss
        self.log("train_loss", total_loss)
        return total_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)
        dice = dice_metric(
            torch.sigmoid(pred), y,
            average='samples',
        )
        self.log("val_dice", dice, prog_bar=True)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=1e-3)
        sheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=5, max_epochs=self.trainer.max_epochs)
        return [optimizer], [sheduler]


def main(args):

    data_path = Path(args.src)
    names = np.array([f.name for f in (data_path / 'img_dir' / 'train').glob('*.png')])

    kf = KFold(n_splits=args.num_folds, random_state=42, shuffle=True)
    for fold, (train_index, test_index) in enumerate(kf.split(names)):

        train_names, test_names = names[train_index], names[test_index]

        # Datasets
        train_dataset = Dataset(data_path, train_names, get_train_transform(args.crop))
        test_dataset = Dataset(data_path, test_names, get_test_transform())

        # Add Auxiliary Datasets
        train_datasets = [train_dataset]
        for d in (data_path / 'ann_dir').glob('auxiliary*'):
            subdir = d.name
            _names = [f.name for f in (data_path / 'ann_dir' / subdir).glob('*.png')]
            train_datasets.append(
                Dataset(data_path, _names, get_train_transform(args.crop), subdir=subdir)
            )

        # Dataloaders
        train_loader = DataLoader(ConcatDataset(train_datasets), batch_size=args.bs, shuffle=True, num_workers=args.workers)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.workers)

        # Basic Unet model
        model = LitSegmenter(
            encoder_name=args.encoder_name,
            encoder_weights=args.encoder_weights,
            classes=1,
            activation=None,
            loss=SoftBCEWithLogitsLoss(smooth_factor=1e-3, ignore_index=-1),
            aux_loss=DiceLoss(mode='binary', smooth=1e-3, ignore_index=-1),
        )

        checkpoint_callback = ModelCheckpoint(
            monitor='val_dice',
            filename=f'fold{fold}_' + '{epoch}-{val_dice:.3f}',
            save_top_k=1,
            mode='max',
            save_weights_only=True
        )
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        rich_progress = RichProgressBar()

        # Trainer
        trainer = pl.Trainer(
            check_val_every_n_epoch=5,
            log_every_n_steps=10,
            max_epochs=args.epochs,
            accumulate_grad_batches=args.accumulate,
            callbacks=[rich_progress, lr_monitor, checkpoint_callback],
            accelerator="gpu", devices=1
        )

        # Fit
        trainer.fit(model, train_loader, test_loader)


if __name__ == "__main__":
    main(parse_args())
