import torch
import os
import glob
from collections import OrderedDict
import argparse
from run_training_split_twoT_train_acc_kd_submission3_64k import PLModule, train  # 你原来的 training 脚本中的模型类
from datasets.dcase23_split_ownT_csv2 import get_test_set
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from argparse import Namespace

def average_checkpoints(ckpt_paths):
    avg_state_dict = None
    for path in ckpt_paths:
        state_dict = torch.load(path, map_location="cpu")["state_dict"]
        if avg_state_dict is None:
            # clone 并转换为 float（防止整数类型不能平均）
            avg_state_dict = OrderedDict({
                k: v.clone().float() if torch.is_floating_point(v) else v.clone().to(torch.float32)
                for k, v in state_dict.items()
            })
        else:
            for k in avg_state_dict:
                if not torch.is_floating_point(avg_state_dict[k]):
                    avg_state_dict[k] = avg_state_dict[k].to(torch.float32)
                avg_state_dict[k] += state_dict[k].to(torch.float32)

    # 除以数量，做平均
    for k in avg_state_dict:
        avg_state_dict[k] /= len(ckpt_paths)

    return avg_state_dict


def main(args):
    # 1️⃣ 加载 ckpt
    ckpt_paths = sorted(glob.glob(os.path.join(args.ckpt_dir, "*.ckpt")))
    assert len(ckpt_paths) > 0, "No checkpoints found!"

    print(f"Found {len(ckpt_paths)} checkpoints for averaging:")
    for p in ckpt_paths:
        print(" -", p)

    # 2️⃣ 平均并保存
    avg_state = average_checkpoints(ckpt_paths)
    os.makedirs(args.output_dir, exist_ok=True)
    avg_ckpt_path = os.path.join(args.output_dir, "averaged_model.ckpt")
    torch.save({"state_dict": avg_state}, avg_ckpt_path)
    print(f"Saved averaged checkpoint to {avg_ckpt_path}")

    # 3️⃣ 初始化模型并加载平均参数
    #config = torch.load(ckpt_paths[0], map_location="cpu").get("hyper_parameters", None)
    #assert config is not None, "hyper_parameters not found in ckpt"

    model = PLModule(config)
    model.load_state_dict(avg_state)
    dataset=get_test_set(config.cache_path, config.resample_rate)
    print("[Info] Test set size:",len(dataset))
    # 4️⃣ 载入 test dataloader
    test_dl = DataLoader(dataset=get_test_set(config.cache_path, config.resample_rate),
                         batch_size=config.batch_size,
                         num_workers=config.num_workers)

    # 5️⃣ 启用 wandb 日志
    wandb_logger = WandbLogger(
        project=config.project_name,
        name=config.experiment_name + "_averaged_test",
        config=config
    )

    # 6️⃣ 用 trainer 测试
    trainer = Trainer(logger=wandb_logger, devices=1, accelerator="auto")
    trainer.test(model, dataloaders=test_dl)

    # wandb 会自动记录通过 self.log_dict() 产生的 test/class-wise/device-wise acc
    # wandb 会自动记录通过 self.log_dict() 产生的 test/class-wise/device-wise acc
    import pandas as pd

    # 提取 logged metrics（wandb logger 同步记录的）
    metrics = trainer.callback_metrics

    # 分类精度字典（去掉 macro/group/损失等字段）
    class_acc = {
        k: v.item() for k, v in metrics.items()
        if k.startswith("test/acc.") and not any(g in k for g in ["real", "seen", "unseen", "cnt", "macro_avg"]) and "loss" not in k
    }

    # 分为设备和标签两类并添加类型标签
    records = []
    for k, v in class_acc.items():
        name = k.replace("test/acc.", "")
        typ = "device" if name in ['a','b','c','s1','s2','s3','s4','s5','s6'] else "label"
        records.append({"name": name, "type": typ, "accuracy": v})

    # 构建并保存 DataFrame
    df_all = pd.DataFrame(records)
    df_all.to_csv(os.path.join(args.output_dir, "accuracy_summary.csv"), index=False)

    print("[ModelSoup] ✅ Saved class-wise and device-wise acc to CSV.")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, default="/model_soup", help="Path to folder with multiple ckpts")
    parser.add_argument("--output_dir", type=str, default="averaged_ckpt", help="Where to save averaged ckpt")
    config = Namespace(
        project_name="Submission",
        experiment_name="CPJKU_cp-avg_model_soup_test_sub3",
        num_workers=12,

        # dataset
        cache_path="datasets/cpath",

        # model
        n_classes=10,
        in_channels=1,
        base_channels=32,
        channels_multiplier=1.8,
        expansion_rate=2.1,
        subset=25,

        # training
        n_epochs=150,
        batch_size=256,
        mixstyle_p=0.0,
        mixstyle_alpha=0.0,
        weight_decay=0.0001,
        roll=10_000,
        dir_prob=0.6,

        # knowledge distillation
        temperature=2.0,
        kd_lambda=0.02,
        feature_weight=0.1,

        # learning rate schedule
        lr=0.0009,
        warm_up_len=14,
        ramp_down_start=50,
        ramp_down_len=84,
        last_lr_value=0.005,

        # preprocessing
        resample_rate=32000,
        window_size=3072,
        hop_size=500,
        n_fft=4096,
        n_mels=256,
        freqm=48,
        timem=0,
        fmin=0,
        fmax=None,
        fmin_aug_range=1,
        fmax_aug_range=1000
    )
    args = parser.parse_args()
    main(args)

