# DCASE2025 - Task 1 - JOINT FEATURE AND OUTPUT DISTILLATION FOR LOW-COMPLEXITY ACOUSTIC SCENE CLASSIFICATION

Contact: **Haowen Li** (haowen.li@ntu.edu.sg), *Nanyang Technological University*



## Low-Complexity Acoustic Scene Classification with Device Information

For a detailed description of the challenge and this task visit the [DCASE website](https://dcase.community/challenge2025/).

Acoustic Scene Classification (ASC) automatically categorizes audio recordings into environmental sound scenes like *metro station*, *urban park*, or *public square*. Similar to last year, this task prioritizes **limited computational resources** (memory ≤ 128 kB and MACs ≤ 30 million) and **diverse recording conditions**. Additionally, **data efficiency** remains a crucial factor: participants must train on a small set of labeled audio data (corresponding to last year's 25% subset).



## Baseline System
https://github.com/CPJKU/dcase2025_task1_baseline


## Overview

This repository implements a **multi-teacher knowledge distillation framework** for low-complexity acoustic scene classification, developed for the **DCASE 2025 Challenge Task 1**.

The method transfers knowledge from multiple high-capacity teacher models (PaSST and CP-ResNet) to a lightweight **CP-Mobile** student model. Teacher predictions are ensembled to provide soft targets for output-level distillation, while intermediate feature representations from CP-ResNet are used for feature-level supervision. The final student model is trained using a combination of **soft target distillation, feature alignment, and cross-entropy loss**, followed by **Model Soup checkpoint averaging**.

This framework enables a compact model (≈61K parameters) to achieve strong performance under strict computational constraints.

## Training Pipeline Overview

The training pipeline follows a **multi-teacher knowledge distillation framework** designed for low-complexity acoustic scene classification. The process consists of several stages: teacher model training, teacher ensemble generation, student distillation training, and checkpoint averaging.

---

### Step 1: PaSST Teacher Training (`train_teacher_passt.py`)

Two PaSST teacher models are trained independently using different spectrogram configurations to capture complementary time–frequency characteristics.

- Model variants emphasize either **frequency resolution** or **temporal resolution**.
- Training uses the **25% subset** of the TAU Urban Acoustic Scenes 2022 Mobile dataset.
- Data augmentation includes **Time Roll**, **DIR convolution**, and **Freq-MixStyle**.
- After convergence, the **top-performing checkpoints are averaged using Model Soup** to improve robustness.

---

### Step 2: CP-ResNet Teacher Training (`train_teacher_cp_resnet.py`)

Two CP-ResNet teacher models are trained with different spectrogram configurations.

- Same **25% subset training protocol** as the PaSST teachers.
- **Freq-MixStyle augmentation** is applied to improve cross-device generalization.
- **Model Soup** is applied to the best checkpoints to stabilize teacher predictions.

After this stage, we obtain **four teacher models**:

- 2 × PaSST
- 2 × CP-ResNet

These teachers provide complementary supervision signals for knowledge distillation.

---

### Step 3: Teacher Logits Ensemble (`gen_logits.py`, `ensemble_logits_from_csv.py`)

To generate soft targets for knowledge distillation:

1. Each teacher model produces logits on the training data.
2. The **softmax outputs of all four teachers are averaged**.
3. The resulting ensemble predictions serve as **soft targets** for student training.

This ensemble improves stability and reduces bias from individual teacher models.

---

### Step 4: Student Model Training (`train_student_cp_mobile.py`)

A lightweight **CP-Mobile** model is trained using a **joint knowledge distillation framework**.

The training objective combines three loss components:

- **Soft Target Distillation Loss**
  - KL divergence between student predictions and ensembled teacher outputs

- **Feature Distillation Loss**
  - Aligns intermediate representations between the student and a **CP-ResNet teacher**

- **Cross-Entropy Loss**
  - Standard supervised loss using ground-truth labels

The final training objective is
L = α Lsoft + β Lfeat + γ Lce

where α, β, and γ control the balance between the distillation and supervised losses.

---

### Step 5: Model Soup (`average_ckpt.py`)

After training convergence, we apply **Model Soup** to the student model checkpoints.

- The **top-K checkpoints** with the best validation performance are selected.
- Their weights are **averaged to obtain the final model**.
- This approach reduces overfitting and improves generalization.

---

### Final Model

The final system uses a **CP-Mobile student model** trained with:

- Multi-teacher supervision (2×PaSST + 2×CP-ResNet)
- Output-level knowledge distillation
- Feature-level distillation from CP-ResNet
- Cross-entropy supervision
- Model Soup checkpoint averaging

This framework enables a compact model (≈61K parameters, 17M MACs) to achieve strong performance under strict low-complexity constraints.

## Getting Started

1. Clone this repository.
2. Create and activate a [conda](https://docs.anaconda.com/free/miniconda/index.html) environment:

```
conda create -n dcase2025 python=3.10
conda activate dcase2025
```

3. Install [PyTorch](https://pytorch.org/get-started/previous-versions/) version that suits your system. For example:

```
# for example:
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# or for the most recent versions:
pip3 install torch torchvision torchaudio
```

4. Install requirements:

```
pip3 install -r requirements.txt
```

5. Download and extract the [TAU Urban Acoustic Scenes 2022 Mobile, Development dataset](https://zenodo.org/records/6337421).

You should end up with a directory that contains, among other files, the following:
* A directory *audio* containing 230,350 audio files in *wav* format
* A file *meta.csv* that contains 230,350 rows with columns *filename*, *scene label*, *identifier* and *source label*


## Training Process

The training pipeline follows a **multi-teacher knowledge distillation framework** consisting of teacher training, teacher ensemble generation, and student distillation.

---

### Step 1: Train Teacher Models

Train the teacher models independently using the 25% subset of the dataset.

Two architectures are used as teachers:

- **PaSST**
- **CP-ResNet**

Each architecture is trained with different spectrogram configurations, resulting in **four teacher models in total**.


```
python train_teacher_passt.py
python train_teacher_cp_resnet.py
```

After training, the **best checkpoints are averaged using Model Soup** to produce more stable teacher models.

---

### Step 2: Generate Teacher Logits

The outputs from all teacher models are used to generate **soft targets**.

Teacher predictions are computed and then **averaged across the four teachers** to form the final logits used for distillation.

```
python gen_logits.py
python ensemble_logits_from_csv.py
```


---

### Step 3: Train the Student Model

The student model (**CP-Mobile**) is trained using a **joint distillation framework** combining:

- **Output-level distillation**
- **Feature-level distillation**
- **Cross-entropy supervision**
```
python train_student_cp_mobile.py
```

---

### Step 4: Model Soup

After training convergence, the best-performing checkpoints are averaged to produce the final model.
```
python average_ckpt.py
```

This improves model stability and reduces overfitting.
### Step 5: Evaluation

Run evaluation on the development split using the trained student model or the averaged checkpoint.
```
python run_eval.py
```

This step reports the final class-wise, device-wise, and macro-average accuracy.
## Results in Development Testset


### Class-wise results
| **Model**              | **Airport** | **Bus** | **Metro** | **Metro Station** | **Park** | **Public Square** | **Shopping Mall** | **Street Pedestrian** | **Street Traffic** | **Tram** | **Macro Avg. Accuracy** |
|------------------------|------------:|--------:|----------:|------------------:|---------:|------------------:|------------------:|----------------------:|-------------------:|---------:|:-----------------------:|
| General Model          |       45.30 |   71.68 |     55.56 |             52.86 |    80.84 |             49.83 |             72.76 |                 31.78 |             74.85  |    57.53 |      59.30              |

### Device-wise results
| **Split**              | **A** | **B** | **C** | **S1** | **S2** | **S3** | **S4** | **S5** | **S6** | **Macro Avg. Accuracy** |
|------------------------|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----------------------:|
| General Model          | 68.00 | 58.66 | 63.59 | 57.18 | 56.30 | 62.12 | 58.48 | 57.24 | 52.18 |        59.30            |


## Inference Code

https://github.com/HaoWLee/dcase2025_task1_inference





## Citation

If you use this code or find it helpful for your research, please cite our paper:

```bibtex
@article{li2025joint,
  title={Joint Feature and Output Distillation for Low-Complexity Acoustic Scene Classification},
  author={Li, Haowen and Yang, Ziyi and Wang, Mou and Tan, Ee-Leng and Yeow, Junwei and Peksi, Santi and Gan, Woon-Seng},
  journal={arXiv preprint arXiv:2507.19557},
  year={2025}
}