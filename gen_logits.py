import os
import argparse
import torch
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from sklearn import preprocessing

from models.passt import get_model
from models.mel import AugmentMelSTFT
from datasets.dcase23_split import get_training_set
from helpers.init import worker_init_fn


class PLModule(pl.LightningModule):
    def __init__(self, config):
        super(PLModule, self).__init__()
        self.save_hyperparameters(config)  # ✅ 允许从 checkpoint 加载 config

        self.config = config
        self.mel = AugmentMelSTFT(
            n_mels=config.n_mels,
            sr=config.resample_rate,
            win_length=config.window_size,
            hopsize=config.hop_size,
            n_fft=config.n_fft,
            freqm=config.freqm,
            timem=config.timem,
            fmin=config.fmin,
            fmax=config.fmax,
            fmin_aug_range=config.fmin_aug_range,
            fmax_aug_range=config.fmax_aug_range
        )

        self.model = get_model(
            arch=config.arch,
            n_classes=config.n_classes,
            input_fdim=config.input_fdim,
            s_patchout_t=config.s_patchout_t,
            s_patchout_f=config.s_patchout_f
        )

    def mel_forward(self, x):
        old_shape = x.size()
        x = x.reshape(-1, old_shape[2])
        x = self.mel(x)
        x = x.reshape(old_shape[0], old_shape[1], x.shape[1], x.shape[2])
        return x

    def forward(self, x):
        return self.model(x)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, files, *rest = batch
        x = self.mel_forward(x)
        logits = self.forward(x)
        if isinstance(logits, tuple):
            logits = logits[0]
        return files, logits


def generate_logits_lightning(config, ckpt_path, output_csv):
    # === Load dataset ===
    dataset = get_training_set(
        config.subset, config.cache_path, config.resample_rate,
        roll=config.roll, dir_prob=config.dir_prob
    )
    print("[Info] Dataset size:", len(dataset))

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        worker_init_fn=worker_init_fn
    )

    # === Load LightningModule with weights ===
    pl_module = PLModule.load_from_checkpoint(ckpt_path, config=config)
    trainer = pl.Trainer(logger=False, enable_checkpointing=False, accelerator="gpu" if torch.cuda.is_available() else "cpu")

    # === Predict ===
    predictions = trainer.predict(pl_module, dataloaders=dataloader)
    print("[Info] Prediction done, processing results...")

    # === Collect results ===
    all_files = [f for batch in predictions for f in batch[0]]
    logits_tensor = torch.cat([batch[1] for batch in predictions], dim=0)
    all_predictions = F.softmax(logits_tensor, dim=1)

    # === Load label names ===
    df_meta = pd.read_csv(config.meta_csv, sep="\t")
    le = preprocessing.LabelEncoder()
    le.fit_transform(df_meta['scene_label'].values)
    class_names = le.classes_

    # === Save to DataFrame ===
    scene_labels = [class_names[i] for i in torch.argmax(all_predictions, dim=1)]
    df = {
        "filename": all_files,
        "scene_label": scene_labels
    }
    for i, name in enumerate(class_names):
        df[name] = logits_tensor[:, i]

    df = pd.DataFrame(df)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, sep='\t', index=False)
    print(f"[Done] Logits + predictions saved to: {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default="/teacher_ckpt/via2gkii/checkpoints/epoch=149-step=20550.ckpt")
    parser.add_argument('--output_csv', type=str, default="/predictions/via2gkii/logits_cp_resnet.csv")
    parser.add_argument('--meta_csv', type=str, default="/dataset/meta.csv")
    parser.add_argument('--cache_path', type=str, default=os.path.join("datasets", "cpath"))
    parser.add_argument('--subset', type=int, default=25)
    '''
    parser.add_argument('--roll', type=int, default=None)
    parser.add_argument('--dir_prob', type=float, default=0.0)
    '''
    parser.add_argument('--roll', type=int, default=10_000)  # roll waveform over time
    parser.add_argument('--dir_prob', type=float, default=0.6)  # prob. to apply device impulse response augmentation
    parser.add_argument('--batch_size', type=int, default=80)
    parser.add_argument('--num_workers', type=int, default=12)

    # 模型参数
    parser.add_argument('--arch', type=str, default='passt_s_swa_p16_128_ap476')
    parser.add_argument('--n_classes', type=int, default=10)
    parser.add_argument('--input_fdim', type=int, default=128)
    parser.add_argument('--s_patchout_t', type=int, default=0)
    parser.add_argument('--s_patchout_f', type=int, default=6)

    # mel 参数
    parser.add_argument('--resample_rate', type=int, default=32000)
    parser.add_argument('--window_size', type=int, default=800)
    parser.add_argument('--hop_size', type=int, default=320)
    parser.add_argument('--n_fft', type=int, default=1024)
    parser.add_argument('--n_mels', type=int, default=128)
    parser.add_argument('--fmin', type=int, default=0)
    parser.add_argument('--fmax', type=int, default=None)
    parser.add_argument('--fmin_aug_range', type=int, default=1)
    parser.add_argument('--fmax_aug_range', type=int, default=1000)
    parser.add_argument('--freqm', type=int, default=48)
    parser.add_argument('--timem', type=int, default=20)

    args = parser.parse_args()
    generate_logits_lightning(args, args.ckpt_path, args.output_csv)
