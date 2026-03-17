import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
import torch
from torch.utils.data import DataLoader
import argparse
import torch.nn.functional as F
import torch.nn as nn
import os
import wandb
from datasets.dcase25_split_ownT_csv2 import get_training_set, get_test_set
from helpers.init import worker_init_fn
from models.cp_mobile_clean import get_model
from models.mel import AugmentMelSTFT
from helpers.lr_schedule import exp_warmup_linear_down
from helpers.utils import mixstyle
from helpers import nessi
from models.cp_resnet import get_model as get_teacher_model
from model_soup_callback_acc import ModelSoupCheckpoint
class TeacherWrapper(nn.Module):
    def __init__(self, model, teacher_features):
        super(TeacherWrapper, self).__init__()
        self.model = model
        self.teacher_features = teacher_features
        # 强制在 Wrapper 里注册 hook，保证正确
        def get_activation(name):
            def hook(model, input, output):
                teacher_features[name] = output
            return hook

        self.model.stage1.register_forward_hook(get_activation('stage1'))
        self.model.stage2.register_forward_hook(get_activation('stage2'))
        self.model.stage3.register_forward_hook(get_activation('stage3'))
        #print("[DEBUG] teacher_model.model.stage1:", self.model.stage1)

        #print("[DEBUG] TeacherWrapper: hooks registered!")

    def forward(self, x):
        #print("[DEBUG] TeacherWrapper forward called")
    
        # 手动触发 stage1/2/3（保证 forward_hook 生效！）
        x = self.model.in_c(x)
        #print(f"[DEBUG] stage1 hook keys before stage1: {self.teacher_features.keys()}")
        x = self.model.stage1(x)   # 这里会 trigger hook for 'stage1'
        #print(f"[DEBUG] stage1 hook keys before stage1: {self.teacher_features.keys()}")
        x = self.model.stage2(x)   # trigger hook for 'stage2'
        x = self.model.stage3(x)   # trigger hook for 'stage3'
    
        #print("[DEBUG] After stage1/2/3")
    
        x = self.model.feed_forward(x)
    
        #print("[DEBUG] After feed_forward")
    
        logit = x.squeeze(2).squeeze(2)
        return logit



class PLModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config  # results from argparse and contains all configurations for our experiment
        # model to preprocess waveforms into log mel spectrograms
        self.mel = AugmentMelSTFT(n_mels=config.n_mels,
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


        print("=== Student Feature Shapes ===")
        # CP-Mobile - our model to be trained on the log mel spectrograms
        self.model = get_model(n_classes=config.n_classes,
                               in_channels=config.in_channels,
                               base_channels=config.base_channels,
                               channels_multiplier=config.channels_multiplier,
                               expansion_rate=config.expansion_rate
                               )
                               # 假设你的模型是 mel + model（即 AugmentMelSTFT + student model）
        
        self.distill_stages = ['stage1', 'stage2', 'stage3']
        #self.proj_stage1 = nn.Conv2d(in_channels=32, out_channels=92, kernel_size=1)
        #self.proj_stage2 = nn.Conv2d(in_channels=64, out_channels=92, kernel_size=1)
        #self.proj_stage3 = nn.Conv2d(in_channels=92,out_channels=92,kernel_size=1)
        # student proj
        self.proj_stage1 = nn.Conv2d(in_channels=32, out_channels=92, kernel_size=1)
        self.proj_stage2 = nn.Conv2d(in_channels=56, out_channels=92, kernel_size=1)  # s2 输出是 72
        self.proj_stage3 = nn.Conv2d(in_channels=104, out_channels=92, kernel_size=1)  # s3 输出是 168

        # teacher proj
        self.proj_teacher_stage1 = nn.Conv2d(in_channels=32, out_channels=92, kernel_size=1)
        self.proj_teacher_stage2 = nn.Conv2d(in_channels=64, out_channels=92, kernel_size=1)
        self.proj_teacher_stage3 = nn.Conv2d(in_channels=92, out_channels=92, kernel_size=1)

        self.teacher_model = get_teacher_model(base_channels=32)  # 改成你 teacher ckpt 对应的参数
        
        teacher_ckpt = torch.load("/teacher_ckpt/via2gkii/checkpoints/epoch=149-step=20550.ckpt", map_location="cpu")
        self.teacher_model.load_state_dict(teacher_ckpt["state_dict"], strict=False)
        self.teacher_model.eval()  # 不训练 teacher
        
        self.teacher_features = {}
        self.student_features = {}
        # 外面定义一个 get_activation 给 student 用
        def get_activation_student(name, features_dict):
            def hook(model, input, output):
                features_dict[name] = output
            return hook


        self.teacher_model = TeacherWrapper(self.teacher_model, self.teacher_features)
        print("[DEBUG] teacher_model:", self.teacher_model)
        print("[DEBUG] teacher_model.forward:", self.teacher_model.forward)
        self.model.stages.s1.register_forward_hook(get_activation_student('stage1', self.student_features))
        self.model.stages.s2.register_forward_hook(get_activation_student('stage2', self.student_features))
        self.model.stages.s3.register_forward_hook(get_activation_student('stage3', self.student_features))
        self.kl_div_loss = nn.KLDivLoss(log_target=True, reduction="none")  # KL Divergence loss for soft targets

        self.device_ids = ['a', 'b', 'c', 's1', 's2', 's3', 's4', 's5', 's6']
        self.label_ids = ['airport', 'bus', 'metro', 'metro_station', 'park', 'public_square', 'shopping_mall',
                          'street_pedestrian', 'street_traffic', 'tram']
        # categorization of devices into 'real', 'seen' and 'unseen'
        self.device_groups = {'a': "real", 'b': "real", 'c': "real",
                              's1': "seen", 's2': "seen", 's3': "seen",
                              's4': "unseen", 's5': "unseen", 's6': "unseen"}
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.kd_stage = "pretrain" 
    def mel_forward(self, x):
        """
        @param x: a batch of raw signals (waveform)
        return: a batch of log mel spectrograms
        """
        old_shape = x.size()
        x = x.reshape(-1, old_shape[2])  # for calculating log mel spectrograms we remove the channel dimension
        x = self.mel(x)
        x = x.reshape(old_shape[0], old_shape[1], x.shape[1], x.shape[2])  # batch x channels x mels x time-frames
        return x

    def forward(self, x):
        """
        :param x: batch of raw audio signals (waveforms)
        :return: final model predictions
        """
        x = self.mel_forward(x)
        x = self.model(x)
        return x

    def configure_optimizers(self):
        """
        This is the way pytorch lightening requires optimizers and learning rate schedulers to be defined.
        The specified items are used automatically in the optimization loop (no need to call optimizer.step() yourself).
        :return: dict containing optimizer and learning rate scheduler
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        schedule_lambda = \
            exp_warmup_linear_down(self.config.warm_up_len, self.config.ramp_down_len, self.config.ramp_down_start,
                                   self.config.last_lr_value)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule_lambda)
        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler
        }
    def on_train_epoch_start(self):
        if self.current_epoch < self.config.pretrain_epochs:
            self.kd_stage = "pretrain"
        else:
            self.kd_stage = "finetune"
    def training_step(self, train_batch, batch_idx):
        """
        :param train_batch: contains one batch from train dataloader
        :param batch_idx
        :return: a dict containing at least loss that is used to update model parameters, can also contain
                    other items that can be processed in 'training_epoch_end' to log other metrics than loss
        """
        
        #print("[DEBUG] teacher_model.model.stage1:", self.model.stage1)

        #for stage in self.distill_stages:
            #print(f"[DEBUG] getattr proj_{stage}: {getattr(self, f'proj_{stage}')}")
        x, file, labels, devices, cities, teacher_logits = train_batch
        x = self.mel_forward(x)  # we convert the raw audio signals into log mel spectrograms

        if self.config.mixstyle_p > 0:
            # frequency mixstyle
            x = mixstyle(x, self.config.mixstyle_p, self.config.mixstyle_alpha)

        # --- Forward teacher ---
        with torch.no_grad():
            teacher_logits_out = self.teacher_model(x)
        #print(f"[DEBUG] distill_stages: {self.distill_stages}")
        #print(f"[DEBUG] teacher_features keys: {self.teacher_features.keys()}")
        #print(f"[DEBUG] student_features keys: {self.student_features.keys()}")
        # --- Forward student ---
        y_hat = self.model(x)

        # --- Losses ---

        # 1️⃣ CE loss
        label_loss = F.cross_entropy(y_hat, labels, reduction="none").mean()

        # 2️⃣ logits distillation loss
        with torch.cuda.amp.autocast():
            y_hat_soft = F.log_softmax(y_hat / self.config.temperature, dim=-1)
        

        kd_loss = self.kl_div_loss(y_hat_soft, teacher_logits).mean() * (self.config.temperature ** 2)
        '''
        total_feature_loss = 0.0

        for stage in self.distill_stages:
            
            student_feat = self.student_features[stage]  # [B, C, H, W]
            teacher_feat = self.teacher_features[stage]  # [B, C', H', W']
            target_h, target_w = teacher_feat.shape[2], teacher_feat.shape[3]

            student_feat_pooled = F.adaptive_avg_pool2d(student_feat, output_size=(target_h, target_w))

            student_feat_proj = getattr(self, f'proj_{stage}')(student_feat_pooled)
            teacher_feat_proj = getattr(self, f'proj_teacher_{stage}')(teacher_feat)  # 不 pool，直接 proj

            student_feat_soft = F.log_softmax(student_feat_proj / self.config.temperature, dim=1)
            teacher_feat_soft = F.softmax(teacher_feat_proj / self.config.temperature, dim=1)

            feature_loss_stage = self.kl_div_loss(student_feat_soft, teacher_feat_soft).mean() *             (self.config.temperature ** 2)

            total_feature_loss += feature_loss_stage

       
         
            # Adaptive Pool teacher
            target_h, target_w = student_feat.shape[2], student_feat.shape[3]
            #teacher_feat_pooled = F.adaptive_avg_pool2d(teacher_feat, output_size=(target_h, target_w))
            student_feat_pooled = F.adaptive_avg_pool2d(student_feat, output_size=(target_h, target_w))
            print(f"[DEBUG][{stage}] teacher_feat shape: {teacher_feat.shape}, after pool: {student_feat_pooled.shape}")
            print(f"[DEBUG][{stage}] student_feat shape: {student_feat.shape}")

            # proj_layer: 你要改成 per-stage 的 proj_layer（见下面补充）
            student_feat_proj = getattr(self, f'proj_{stage}')(student_feat_pooled)  # 自动找 proj_stage1/2/3
            teacher_feat_proj = getattr(self, f'proj_teacher_{stage}')(teacher_feat)
            # *** 关键加打印！！！ ***
            print(f"[DEBUG][{stage}] student_feat_proj shape: {student_feat_proj.shape}")
            print(f"[DEBUG][{stage}] teacher_feat_proj shape: {teacher_feat_proj.shape}")
            # --- 用 KD loss 替代 MSE ---
            #student_feat_soft = F.log_softmax(student_feat_proj / self.config.temperature, dim=1)
            #teacher_feat_soft = F.softmax(teacher_feat_pooled / self.config.temperature, dim=1)

            #feature_loss_stage = self.kl_div_loss(student_feat_soft, teacher_feat_soft).mean() * (
            #            self.config.temperature ** 2)

            #total_feature_loss += feature_loss_stage
        '''
        total_feature_loss = 0.0

        for stage in self.distill_stages:
            student_feat = self.student_features[stage]  # [B, C, H, W]
            teacher_feat = self.teacher_features[stage]  # [B, C', H, W]
            
            # 先投影到同一维度
            student_feat_proj = getattr(self, f'proj_{stage}')(student_feat)  # [B, C_proj, H, W]
            teacher_feat_proj = getattr(self, f'proj_teacher_{stage}')(teacher_feat)  # [B, C_proj, H, W]
            '''
            # 论文 Gtf 做法：每 (t,f) bin 求 Gram 矩阵
            B, C, H, W = student_feat_proj.shape
        
            # reshape [B, C, H, W] → [H, W, B, C]
            student_feat_reshaped = student_feat_proj.permute(2, 3, 0, 1)
            teacher_feat_reshaped = teacher_feat_proj.permute(2, 3, 0, 1)

            # 计算 Self-similarity Gram matrix
            # G_student, G_teacher: [H, W, B, B]
            G_student = torch.einsum('hwbi,hwbj->hwij', student_feat_reshaped, student_feat_reshaped) / C
            G_teacher = torch.einsum('hwbi,hwbj->hwij', teacher_feat_reshaped, teacher_feat_reshaped) / C
            '''
            # 明确指定 target_H, target_W（固定或者动态）
            target_H = min(student_feat_proj.shape[2], teacher_feat_proj.shape[2])
            target_W = min(student_feat_proj.shape[3], teacher_feat_proj.shape[3])

            # 做 pooling
            student_feat_proj = F.adaptive_avg_pool2d(student_feat_proj, output_size=(target_H, target_W))
            teacher_feat_proj = F.adaptive_avg_pool2d(teacher_feat_proj, output_size=(target_H, target_W))

            # 现在 pooling 完，H, W 一定一致
            # 再 permute
            student_feat_reshaped = student_feat_proj.permute(2, 3, 0, 1)  # [H, W, B, C]
            teacher_feat_reshaped = teacher_feat_proj.permute(2, 3, 0, 1)  # [H, W, B, C]

            # Gtf
            G_student = torch.einsum('hwbi,hwbj->hwij', student_feat_reshaped, student_feat_reshaped) / student_feat_proj.shape[1]
            G_teacher = torch.einsum('hwbi,hwbj->hwij', teacher_feat_reshaped, teacher_feat_reshaped) / teacher_feat_proj.shape[1]

            # Gtf loss = MSE
            feature_loss_stage = F.mse_loss(G_student, G_teacher)

            total_feature_loss += feature_loss_stage
    

        # --- Total loss ---
        '''
        total_loss = (
                self.config.kd_lambda * label_loss +
                self.config.feature_weight * total_feature_loss +
                (1.0 - self.config.kd_lambda - self.config.feature_weight) * kd_loss
        )
        '''
        if self.kd_stage == "pretrain":
            #gamma = 1.0
            total_loss = (
                self.config.kd_weight_pretrain * kd_loss +
                self.config.feature_weight_pretrain * total_feature_loss +
                self.config.label_weight_pretrain * label_loss  # 稳定训练但不主导
            ) # 只用 feature_loss 作为 L_KD，模拟 Gtf loss
        else:  # finetune
            total_loss = (
                self.config.label_weight * label_loss +
                self.config.kd_weight * kd_loss +
                self.config.feature_weight * total_feature_loss
            )
        _, preds = torch.max(y_hat, dim=1)
        n_correct_pred_per_sample = (preds == labels)
        n_correct_pred = n_correct_pred_per_sample.sum()
        '''
        results = {
            "loss": total_loss,
            "label_loss": label_loss * self.config.kd_lambda,
            "kd_loss": kd_loss * (1.0 - self.config.kd_lambda - self.config.feature_weight),
            "feature_loss": total_feature_loss * self.config.feature_weight,
            "n_correct_pred": n_correct_pred,
            "n_pred": torch.as_tensor(len(labels), device=self.device)
        }
        '''
        results = {
            "loss": total_loss,

            # 原始 loss 值
            "label_loss_raw": label_loss,
            "kd_loss_raw": kd_loss,
            "feature_loss_raw": total_feature_loss,

            # 加权后 loss 值
            "label_loss_scaled": label_loss * self.config.label_weight,
            "kd_loss_scaled": kd_loss * self.config.kd_weight,
            "feature_loss_scaled": total_feature_loss * self.config.feature_weight,

            # loss 贡献占比
            "label_loss_percent": (self.config.label_weight * label_loss / total_loss),
            "kd_loss_percent": (self.config.kd_weight * kd_loss / total_loss),
            "feature_loss_percent": (self.config.feature_weight * total_feature_loss / total_loss),

            "n_correct_pred": n_correct_pred,
            "n_pred": torch.as_tensor(len(labels), device=self.device)
        }
        self.training_step_outputs.append(results)

        return results
    '''
    def on_train_epoch_end(self):
        outputs = self.training_step_outputs
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_label_loss = torch.stack([x['label_loss'] for x in outputs]).mean()
        avg_kd_loss = torch.stack([x['kd_loss'] for x in outputs]).mean()
        avg_feature_loss = torch.stack([x['feature_loss'] for x in outputs]).mean()

        train_acc = sum([x['n_correct_pred'] for x in outputs]) / sum([x['n_pred'] for x in outputs])

        logs = {
            "train/loss": avg_loss,
            "train/label_loss": avg_label_loss,
            "train/kd_loss": avg_kd_loss,
            "train/feature_loss": avg_feature_loss,
            "train/acc": train_acc
        }

        self.log_dict(logs, sync_dist=True)
        self.training_step_outputs.clear()
        pass
    '''
    def on_train_epoch_end(self):
        outputs = self.training_step_outputs

        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_label_loss = torch.stack([x['label_loss_raw'] for x in outputs]).mean()
        avg_kd_loss = torch.stack([x['kd_loss_raw'] for x in outputs]).mean()
        avg_feature_loss = torch.stack([x['feature_loss_raw'] for x in outputs]).mean()

        train_acc = sum([x['n_correct_pred'] for x in outputs]) / sum([x['n_pred'] for x in outputs])

        logs = {
            "train/loss": avg_loss,
            "train/label_loss": avg_label_loss,
            "train/kd_loss": avg_kd_loss,
            "train/feature_loss": avg_feature_loss,
            "train/acc": train_acc
        }
        
        # 可选：记录 loss 的加权占比（百分比）
        avg_label_percent = torch.stack([x['label_loss_percent'] for x in outputs]).mean()
        avg_kd_percent = torch.stack([x['kd_loss_percent'] for x in outputs]).mean()
        avg_feature_percent = torch.stack([x['feature_loss_percent'] for x in outputs]).mean()

        logs.update({
            "train/label_loss_percent": avg_label_percent,
            "train/kd_loss_percent": avg_kd_percent,
            "train/feature_loss_percent": avg_feature_percent
        })

        self.log_dict(logs, sync_dist=True)
        self.training_step_outputs.clear()
        pass
    # def training_epoch_end(self, outputs):
    #     """
    #     :param outputs: contains the items you log in 'training_step'
    #     :return: a dict containing the metrics you want to log to Weights and Biases
    #     """
    #     avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
    #     avg_label_loss = torch.stack([x['label_loss'] for x in outputs]).mean()
    #     avg_kd_loss = torch.stack([x['kd_loss'] for x in outputs]).mean()
    #     self.log_dict({'loss': avg_loss, 'label_loss': avg_label_loss, 'kd_loss': avg_kd_loss})


    def validation_step(self, val_batch, batch_idx):
        x, files, labels, devices, cities = val_batch
        x = self.mel_forward(x)
        y_hat = self.model(x)
        samples_loss = F.cross_entropy(y_hat, labels, reduction="none")
        loss = samples_loss.mean()

        # for computing accuracy
        _, preds = torch.max(y_hat, dim=1)
        n_correct_pred_per_sample = (preds == labels)
        n_correct_pred = n_correct_pred_per_sample.sum()

        dev_names = [d.rsplit("-", 1)[1][:-4] for d in files]
        # results = {'val_loss': loss, "n_correct_pred": n_correct_pred, "n_pred": len(labels)}
        results = {'val_loss': loss, "n_correct_pred": n_correct_pred, "n_pred": torch.as_tensor(len(labels), device=self.device)}
        self.validation_step_outputs.append(results)
        # # log metric per device and scene
        # for d in self.device_ids:
        #     results["devloss." + d] = torch.as_tensor(0., device=self.device)
        #     results["devcnt." + d] = torch.as_tensor(0., device=self.device)
        #     results["devn_correct." + d] = torch.as_tensor(0., device=self.device)
        # for i, d in enumerate(dev_names):
        #     results["devloss." + d] = results["devloss." + d] + samples_loss[i]
        #     results["devn_correct." + d] = results["devn_correct." + d] + n_correct_pred_per_sample[i]
        #     results["devcnt." + d] = results["devcnt." + d] + 1
        #
        # for l in self.label_ids:
        #     results["lblloss." + l] = torch.as_tensor(0., device=self.device)
        #     results["lblcnt." + l] = torch.as_tensor(0., device=self.device)
        #     results["lbln_correct." + l] = torch.as_tensor(0., device=self.device)
        # for i, l in enumerate(labels):
        #     results["lblloss." + self.label_ids[l]] = results["lblloss." + self.label_ids[l]] + samples_loss[i]
        #     results["lbln_correct." + self.label_ids[l]] = \
        #         results["lbln_correct." + self.label_ids[l]] + n_correct_pred_per_sample[i]
        #     results["lblcnt." + self.label_ids[l]] = results["lblcnt." + self.label_ids[l]] + 1
        # return results

    def on_validation_epoch_end(self):
        """
        Called at the end of the validation epoch. Aggregates per-device/per-label stats and logs them.
        """
        # Flatten the outputs into a dict of lists
        outputs = {k: [] for k in self.validation_step_outputs[0]}
        for step_output in self.validation_step_outputs:
            for k, v in step_output.items():
                outputs[k].append(v)

        # Stack each list of tensors
        for k in outputs:
            outputs[k] = torch.stack(outputs[k])

        # Compute overall metrics
        avg_loss = outputs["val_loss"].mean()
        acc = outputs["n_correct_pred"].sum() / outputs["n_pred"].sum()

        logs = {"acc": acc, "loss": avg_loss}
        #
        # # Per-device stats
        # for d in self.device_ids:
        #     dev_loss = outputs[f"devloss.{d}"].sum()
        #     dev_cnt = outputs[f"devcnt.{d}"].sum()
        #     dev_correct = outputs[f"devn_correct.{d}"].sum()
        #     logs[f"loss.{d}"] = dev_loss / dev_cnt
        #     logs[f"acc.{d}"] = dev_correct / dev_cnt
        #     logs[f"cnt.{d}"] = dev_cnt
        #
        #     # Group stats
        #     group_name = self.device_groups[d]
        #     logs[f"acc.{group_name}"] = logs.get(f"acc.{group_name}", 0.) + dev_correct
        #     logs[f"count.{group_name}"] = logs.get(f"count.{group_name}", 0.) + dev_cnt
        #     logs[f"lloss.{group_name}"] = logs.get(f"lloss.{group_name}", 0.) + dev_loss
        #
        # # Reduce group stats
        # for grp in set(self.device_groups.values()):
        #     logs[f"acc.{grp}"] = logs[f"acc.{grp}"] / logs[f"count.{grp}"]
        #     logs[f"lloss.{grp}"] = logs[f"lloss.{grp}"] / logs[f"count.{grp}"]
        #
        # # Per-label stats
        # for lbl in self.label_ids:
        #     lbl_loss = outputs[f"lblloss.{lbl}"].sum()
        #     lbl_cnt = outputs[f"lblcnt.{lbl}"].sum()
        #     lbl_correct = outputs[f"lbln_correct.{lbl}"].sum()
        #
        #     logs[f"loss.{lbl}"] = lbl_loss / lbl_cnt
        #     logs[f"acc.{lbl}"] = lbl_correct / lbl_cnt
        #     logs[f"cnt.{lbl}"] = lbl_cnt.float()
        #
        # # Compute macro-average accuracy over all labels
        # logs["macro_avg_acc"] = torch.mean(torch.stack([logs[f"acc.{l}"] for l in self.label_ids]))

        # Log everything with 'val/' prefix
        self.log_dict({f"val/{k}": v for k, v in logs.items()})
        self.validation_step_outputs.clear()

    # def test_step(self, test_batch, batch_idx):
    #     x, files, labels, devices, _ = test_batch
    #
    #     # For memory constraints, switch model to half-precision
    #     # self.model.half()
    #     x = self.mel_forward(x)
    #     # x = x.half()
    #
    #     y_hat = self.model(x)
    #     samples_loss = F.cross_entropy(y_hat, labels, reduction="none")
    #     # loss = samples_loss.mean()
    #     _, preds = torch.max(y_hat, dim=1)
    #     n_correct_per_sample = (preds == labels)
    #     n_correct = n_correct_per_sample.sum()
    #     dev_names = [d.rsplit("-", 1)[1][:-4] for d in files]
    #     results = {
    #         "loss": samples_loss.mean(),
    #         "n_correct": n_correct,
    #         "n_pred": torch.as_tensor(len(labels), device=self.device)
    #     }
    #
    #     # Per-device stats
    #     for d in self.device_ids:
    #         results[f"devloss.{d}"] = torch.as_tensor(0., device=self.device)
    #         results[f"devcnt.{d}"] = torch.as_tensor(0., device=self.device)
    #         results[f"devn_correct.{d}"] = torch.as_tensor(0., device=self.device)
    #
    #     for i, d in enumerate(devices):
    #         results["devloss." + d] = results["devloss." + d] + samples_loss[i]
    #         results["devn_correct." + d] = results["devn_correct." + d] + n_correct_per_sample[i]
    #         results["devcnt." + d] = results["devcnt." + d] + 1
    #         # results[f"devloss.{d}"] += samples_loss[i]
    #         # results[f"devn_correct.{d}"] += n_correct_per_sample[i]
    #         # results[f"devcnt.{d}"] += 1
    #
    #     # Per-label stats
    #     for lbl in self.label_ids:
    #         results[f"lblloss.{lbl}"] = torch.as_tensor(0., device=self.device)
    #         results[f"lblcnt.{lbl}"] = torch.as_tensor(0., device=self.device)
    #         results[f"lbln_correct.{lbl}"] = torch.as_tensor(0., device=self.device)
    #
    #     # for i, lbl_index in enumerate(labels):
    #     #     lbl_name = self.label_ids[lbl_index]
    #     #     results[f"lblloss.{lbl_name}"] += samples_loss[i]
    #     #     results[f"lbln_correct.{lbl_name}"] += n_correct_per_sample[i]
    #     #     results[f"lblcnt.{lbl_name}"] += 1
    #     for i, l in enumerate(labels):
    #         results["lblloss." + self.label_ids[l]] = results["lblloss." + self.label_ids[l]] + samples_loss[i]
    #         results["lbln_correct." + self.label_ids[l]] = \
    #             results["lbln_correct." + self.label_ids[l]] + n_correct_per_sample[i]
    #         results["lblcnt." + self.label_ids[l]] = results["lblcnt." + self.label_ids[l]] + 1
    #     # self.test_step_outputs.append({k: v.cpu() for k, v in results.items()})
    #     self.test_step_outputs.append(results)
    # def on_test_epoch_end(self):
    #         # Flatten the outputs
    #     outputs = {k: [] for k in self.test_step_outputs[0]}
    #     for step_output in self.test_step_outputs:
    #         for k, v in step_output.items():
    #             outputs[k].append(v)
    #
    #     # Stack each list of tensors
    #     for k in outputs:
    #         outputs[k] = torch.stack(outputs[k])
    #
    #     avg_loss = outputs["loss"].mean()
    #     acc = outputs["n_correct"].sum() / outputs["n_pred"].sum()
    #     logs = {"acc": acc, "loss": avg_loss}
    #
    #     # Device-level stats
    #     for d in self.device_ids:
    #         dev_loss = outputs[f"devloss.{d}"].sum()
    #         dev_cnt = outputs[f"devcnt.{d}"].sum()
    #         dev_correct = outputs[f"devn_correct.{d}"].sum()
    #         logs[f"loss.{d}"] = dev_loss / dev_cnt
    #         logs[f"acc.{d}"] = dev_correct / dev_cnt
    #         logs[f"cnt.{d}"] = dev_cnt
    #
    #         # Device groups
    #         grp = self.device_groups[d]
    #         logs[f"acc.{grp}"] = logs.get(f"acc.{grp}", 0.) + dev_correct
    #         logs[f"count.{grp}"] = logs.get(f"count.{grp}", 0.) + dev_cnt
    #         logs[f"lloss.{grp}"] = logs.get(f"lloss.{grp}", 0.) + dev_loss
    #
    #     # Group-level stats
    #     for grp in set(self.device_groups.values()):
    #         logs[f"acc.{grp}"] = logs[f"acc.{grp}"] / logs[f"count.{grp}"]
    #         logs[f"lloss.{grp}"] = logs[f"lloss.{grp}"] / logs[f"count.{grp}"]
    #
    #     # Label-level stats
    #     for lbl in self.label_ids:
    #         lbl_loss = outputs[f"lblloss.{lbl}"].sum()
    #         lbl_cnt = outputs[f"lblcnt.{lbl}"].sum()
    #         lbl_correct = outputs[f"lbln_correct.{lbl}"].sum()
    #         logs[f"loss.{lbl}"] = lbl_loss / lbl_cnt
    #         logs[f"acc.{lbl}"] = lbl_correct / lbl_cnt
    #         logs[f"cnt.{lbl}"] = lbl_cnt
    #
    #     # Macro-average accuracy over all labels
    #     logs["macro_avg_acc"] = torch.mean(torch.stack([logs[f"acc.{l}"] for l in self.label_ids]))
    #
    #     self.log_dict({f"test/{k}": v for k, v in logs.items()})
    #     self.test_step_outputs.clear()
    def test_step(self, batch, batch_idx):
        print("111")
        x, files, labels, devices, cities = batch
        x = self.mel_forward(x)
        self.model.half()
        x = x.half()
        y_hat = self.model(x)
        samples_loss = F.cross_entropy(y_hat, labels, reduction="none")
        loss = samples_loss.mean()

        _, preds = torch.max(y_hat, dim=1)
        n_correct_pred_per_sample = (preds == labels)
        n_correct_pred = n_correct_pred_per_sample.sum()

        dev_names = [d.rsplit("-", 1)[1][:-4] for d in files]

        results = {
            'test_loss': loss,
            'n_correct_pred': n_correct_pred,
            'n_pred': torch.tensor(len(labels), device=self.device)
        }

        for d in self.device_ids:
            results[f"devloss.{d}"] = torch.tensor(0., device=self.device)
            results[f"devcnt.{d}"] = torch.tensor(0., device=self.device)
            results[f"devn_correct.{d}"] = torch.tensor(0., device=self.device)

        for i, d in enumerate(dev_names):
            d = str(d)
            results[f"devloss.{d}"] += samples_loss[i]
            results[f"devn_correct.{d}"] += n_correct_pred_per_sample[i]
            results[f"devcnt.{d}"] += 1

        for l in self.label_ids:
            results[f"lblloss.{l}"] = torch.tensor(0., device=self.device)
            results[f"lblcnt.{l}"] = torch.tensor(0., device=self.device)
            results[f"lbln_correct.{l}"] = torch.tensor(0., device=self.device)

        for i, l in enumerate(labels):
            lbl_name = self.label_ids[l.item()]
            results[f"lblloss.{lbl_name}"] += samples_loss[i]
            results[f"lbln_correct.{lbl_name}"] += n_correct_pred_per_sample[i]
            results[f"lblcnt.{lbl_name}"] += 1

        # 关键：收集每个 batch 的输出
        self.test_step_outputs.append(results)

    def on_test_epoch_end(self):
        outputs = self.test_step_outputs

        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        test_acc = sum([x['n_correct_pred'] for x in outputs]) / sum([x['n_pred'] for x in outputs])

        logs = {
            'test/acc': test_acc,
            'test/loss': avg_loss,
        }

        for d in self.device_ids:
            dev_loss = torch.stack([x[f"devloss.{d}"] for x in outputs]).sum()
            dev_cnt = torch.stack([x[f"devcnt.{d}"] for x in outputs]).sum()
            dev_correct = torch.stack([x[f"devn_correct.{d}"] for x in outputs]).sum()

            logs[f"test/loss.{d}"] = dev_loss / dev_cnt
            logs[f"test/acc.{d}"] = dev_correct / dev_cnt
            logs[f"test/cnt.{d}"] = dev_cnt

            group = self.device_groups[d]
            logs[f"acc.{group}"] = logs.get(f"acc.{group}", 0.) + dev_correct
            logs[f"count.{group}"] = logs.get(f"count.{group}", 0.) + dev_cnt
            logs[f"lloss.{group}"] = logs.get(f"lloss.{group}", 0.) + dev_loss

        for grp in set(self.device_groups.values()):
            logs[f"test/acc.{grp}"] = logs[f"acc.{grp}"] / logs[f"count.{grp}"]
            logs[f"test/loss.{grp}"] = logs[f"lloss.{grp}"] / logs[f"count.{grp}"]

        for lbl in self.label_ids:
            lbl_loss = torch.stack([x[f"lblloss.{lbl}"] for x in outputs]).sum()
            lbl_cnt = torch.stack([x[f"lblcnt.{lbl}"] for x in outputs]).sum()
            lbl_correct = torch.stack([x[f"lbln_correct.{lbl}"] for x in outputs]).sum()

            logs[f"test/loss.{lbl}"] = lbl_loss / lbl_cnt
            logs[f"test/acc.{lbl}"] = lbl_correct / lbl_cnt
            logs[f"test/cnt.{lbl}"] = lbl_cnt

        logs["test/macro_avg_acc"] = torch.mean(torch.stack([logs[f"test/acc.{l}"] for l in self.label_ids]))

        self.log_dict(logs, sync_dist=True)

        self.test_step_outputs.clear()
def train(config):
    # logging is done using wandb
    wandb_logger = WandbLogger(
        project=config.project_name,
        notes="DCASE25 Task 1.",
        tags=["DCASE25"],
        config=config,  # this logs all hyperparameters for us
        name=config.experiment_name
    )

    # train dataloader
    assert config.subset in {100, 50, 25, 10, 5}, "Invalid subset value."
    # roll_samples = config.orig_sample_rate * config.roll_sec
    train_dl = DataLoader(dataset=get_training_set(config.subset,config.cache_path, config.resample_rate, config.roll,
                                                   config.dir_prob, config.temperature),
                          worker_init_fn=worker_init_fn,
                          num_workers=config.num_workers,
                          batch_size=config.batch_size,
                          shuffle=True)

    # test loader
    test_dl = DataLoader(dataset=get_test_set(config.cache_path, config.resample_rate),
                         worker_init_fn=worker_init_fn,
                         num_workers=config.num_workers,
                         batch_size=config.batch_size)

    # create pytorch lightening module
    pl_module = PLModule(config)

    # get model complexity from nessi and log results to wandb
    # ATTENTION: this is before layer fusion, therefore the MACs and Params slightly deviate from what is
    # reported in the challenge submission
    sample = next(iter(train_dl))[0][0].unsqueeze(0)
    shape = pl_module.mel_forward(sample).size()
    macs, params = nessi.get_model_size(pl_module.model, input_size=shape)
    wandb_logger.experiment.config['MACs'] = macs
    wandb_logger.experiment.config['Parameters'] = params

    # create monitor to keep track of learning rate - we want to check the behaviour of our learning rate schedule
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    # create the pytorch lightening trainer by specifying the number of epochs to train, the logger,
    # on which kind of device(s) to train and possible callbacks
    run_name = config.experiment_name
    model_soup_callback = ModelSoupCheckpoint(
        monitor="val/acc",
        save_root="model_soup_ckpts",
        acc_threshold=0.58,
        run_name=config.experiment_name
    )
    trainer = pl.Trainer(
        max_epochs=config.n_epochs,
        logger=wandb_logger,
        accelerator='auto',
        devices=1,
        callbacks=[lr_monitor, model_soup_callback]
    )  
    # start training and validation for the specified number of epochs
    trainer.fit(pl_module, train_dl, test_dl)
    # ==== 最终测试 ====
    trainer.test(ckpt_path='last', dataloaders=test_dl)

    # ==== 关闭 wandb ====
    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')

    # general
    parser.add_argument('--project_name', type=str, default="DCASE23_Task1")
    parser.add_argument('--experiment_name', type=str, default="cp-mobile_student_training_sub25_Tresnet_stageall_output4T_2step_modelsoup_submission1_64k")
    parser.add_argument('--num_workers', type=int, default=12)  # number of workers for dataloaders

    # dataset
    # location to store resampled waveform
    parser.add_argument('--cache_path', type=str, default=os.path.join("datasets", "cpath"))

    # model
    parser.add_argument('--n_classes', type=int, default=10)  # classification model with 'n_classes' output neurons
    parser.add_argument('--in_channels', type=int, default=1)
    # adapt the complexity of the neural network (3 main dimensions to scale CP-Mobile)
    parser.add_argument('--base_channels', type=int, default=32)
    parser.add_argument('--channels_multiplier', type=int, default=1.8)
    parser.add_argument('--expansion_rate', type=int, default=2.1)
    parser.add_argument('--subset', type=int, default=25)
    # training
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--mixstyle_p', type=float, default=0.0)  # frequency mixstyle
    parser.add_argument('--mixstyle_alpha', type=float, default=0.0)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    '''
    parser.add_argument('--roll', type=int, default=None)  # roll waveform over time
    parser.add_argument('--dir_prob', type=float, default=0.0)  # prob. to apply device impulse response augmentation
    '''
    parser.add_argument('--roll', type=int, default=10_000)  # roll waveform over time
    parser.add_argument('--dir_prob', type=float, default=0.6)  # prob. to apply device impulse response augmentation
    ## knowledge distillation
    parser.add_argument('--temperature', type=float, default=2.0)
    '''
    parser.add_argument('--kd_lambda', type=float, default=0.02)
    parser.add_argument('--feature_weight', type=float, default=0.1)
    parser.add_argument('--feature_weight_pretrain', type=float, default=0.1)
    '''
    parser.add_argument('--label_weight', type=float, default=1.0)
    parser.add_argument('--kd_weight', type=float, default=0.1)
    parser.add_argument('--feature_weight', type=float, default=0.05)
    parser.add_argument('--feature_weight_pretrain', type=float, default=0.1)
    parser.add_argument('--kd_weight_pretrain', type=float, default=1.0)
    parser.add_argument('--label_weight_pretrain', type=float, default=0.01)
    # learning rate + schedule
    # phases:
    #  1. exponentially increasing warmup phase (for 'warm_up_len' epochs)
    #  2. constant lr phase using value specified in 'lr' (for 'ramp_down_start' - 'warm_up_len' epochs)
    #  3. linearly decreasing to value 'las_lr_value' * 'lr' (for 'ramp_down_len' epochs)
    #  4. finetuning phase using a learning rate of 'last_lr_value' * 'lr' (for the rest of epochs up to 'n_epochs')
    parser.add_argument('--lr', type=float, default=0.0009)
    parser.add_argument('--warm_up_len', type=int, default=14)
    parser.add_argument('--ramp_down_start', type=int, default=50)
    parser.add_argument('--ramp_down_len', type=int, default=84)
    parser.add_argument('--last_lr_value', type=float, default=0.005)  # relative to 'lr'

    # preprocessing
    parser.add_argument('--resample_rate', type=int, default=32000)
    parser.add_argument('--window_size', type=int, default=3072)  # in samples (corresponds to 96 ms)
    parser.add_argument('--hop_size', type=int, default=500)  # in samples (corresponds to ~16 ms)
    parser.add_argument('--n_fft', type=int, default=4096)  # length (points) of fft, e.g. 4096 point FFT
    parser.add_argument('--n_mels', type=int, default=256)  # number of mel bins
    parser.add_argument('--freqm', type=int, default=48)  # mask up to 'freqm' spectrogram bins
    parser.add_argument('--timem', type=int, default=0)  # mask up to 'timem' spectrogram frames
    parser.add_argument('--fmin', type=int, default=0)  # mel bins are created for freqs. between 'fmin' and 'fmax'
    parser.add_argument('--fmax', type=int, default=None)
    parser.add_argument('--fmin_aug_range', type=int, default=1)  # data augmentation: vary 'fmin' and 'fmax'
    parser.add_argument('--fmax_aug_range', type=int, default=1000)
    
    parser.add_argument('--pretrain_epochs', type=int, default=170)

    args = parser.parse_args()
    train(args)
