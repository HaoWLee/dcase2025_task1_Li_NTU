import os
import torch
from pytorch_lightning.callbacks import Callback

class ModelSoupCheckpoint(Callback):
    def __init__(self, monitor="val/acc", save_root="model_soup_ckpts", acc_threshold=0.59, run_name="unnamed_run"):
        super().__init__()
        self.monitor = monitor
        self.acc_threshold = acc_threshold
        self.save_dir = os.path.join(save_root, run_name)
        os.makedirs(self.save_dir, exist_ok=True)

    def on_validation_end(self, trainer, pl_module):
        current = trainer.callback_metrics.get(self.monitor)
        epoch = trainer.current_epoch

        if current is None:
            print(f"[ModelSoup] WARNING: '{self.monitor}' not found in callback_metrics.")
            return

        current = current.item()
        print(f"[ModelSoup] Epoch {epoch}, val/acc = {current:.4f}")

        if current >= self.acc_threshold:
            ckpt_path = os.path.join(self.save_dir, f"epoch={epoch}-valacc={current:.4f}.ckpt")
            try:
                torch.save({"state_dict": pl_module.state_dict()}, ckpt_path)
                print(f"[ModelSoup] ✅ Saved checkpoint: {ckpt_path}")
            except Exception as e:
                print(f"[ModelSoup] ❌ Failed to save checkpoint: {e}")
        else:
            print(f"[ModelSoup] Skipped saving (val/acc {current:.4f} < threshold {self.acc_threshold})")

