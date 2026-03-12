from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping

import wandb
from pytorch_lightning.loggers import WandbLogger
from data import TorqueForceDataModule
from model import LitLSTM

if __name__ == "__main__":
    wandb.init(project="cutter-machine", name="BTTR", entity="ntcuong2103-vietnamese-german-university")

    dm = TorqueForceDataModule(
        signals_dir="data_root/signals",
        vector_dir="data_root/vectors",
        global_mean_std_file="data_root/global_mean_std.csv",
        batch_size=2,
        num_workers=2,
        window_size=500,
        step_size=125,
    )
    
    model = LitLSTM(input_size=1, control_param_size=6, hidden_size=64, num_layers=2, output_size=3, learning_rate=1e-3)

    trainer = Trainer(
        callbacks = [
            LearningRateMonitor(logging_interval='epoch'),
            ModelCheckpoint(filename='{epoch}-{step}-{val_loss:.4f}', save_top_k=5, monitor='val_loss', mode='min'),
            EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=True),
        ], 
        check_val_every_n_epoch=1,
        fast_dev_run=False,
        max_epochs=200,
        log_every_n_steps=1,
        # accelerator='gpu',
        # devices=1,
        logger=WandbLogger(),
    )

    trainer.fit(model, dm, ckpt_path=None)
    wandb.finish()