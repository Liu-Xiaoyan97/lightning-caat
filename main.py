import argparse

import lightning
from lightning import Trainer
from omegaconf import OmegaConf

from modules.datamodules import DataModules
from modules.lightningmodules import GenerateModule


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_len', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--is_train', type=str, default='Y')
    args = parser.parse_args()
    dm = DataModules(file_path_dir=r"./Data/zh", max_len=args.max_len, batch_size=args.batch_size)
    lightning.seed_everything(1)
    """load config.yml"""
    config = OmegaConf.load("config.yml")
    model = GenerateModule(**config, max_len=args.max_len, num_layers=args.num_layers, batch_size=args.batch_size)
    trainer = Trainer(
        callbacks=[
            lightning.pytorch.callbacks.ModelCheckpoint(
                monitor="val_blue",
                save_top_k=2,
                filename='{epoch}-{val_loss:.4f}-{val_blue:.4f}',
                mode='max'
            )
        ],
        accelerator="auto",
        devices='auto',
        max_epochs=args.max_epoch,
        # limit_train_batches=0.1,
        # limit_val_batches=0.1,
        check_val_every_n_epoch=5,
        log_every_n_steps=1
    )
    if args.is_train == "Y":
        dm.setup('fit')
        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader()
        if args.ckpt:
            trainer.fit(model, train_loader, val_loader, ckpt_path=args.ckpt)
        else:
            trainer.fit(model, train_loader, val_loader)
    else:
        dm.setup('test')
        test_loader = dm.test_dataloader()
        trainer.test(model, test_loader, args.ckpt)
