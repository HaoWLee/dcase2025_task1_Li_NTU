import pandas as pd
import os
import pathlib
from sklearn import preprocessing
from scipy.signal import convolve
from torch.utils.data import Dataset as TorchDataset
import torch
import numpy as np
import librosa
import torch.nn.functional as F
from torch.hub import download_url_to_file


dataset_root = "/home/lihaowen/dcase2025_task1_baseline-main/dataset"
dataset_dir = os.path.join(dataset_root, "tau25")
# dataset_dir = None
assert dataset_dir is not None, "Specify 'TAU Urban Acoustic Scenes 2022 Mobile dataset' location in variable " \
                                "'dataset_dir'. The dataset can be downloaded from this URL:" \
                                " https://zenodo.org/record/6337421"
'''
#teacher_logits_url = "https://github.com/fschmid56/cpjku_dcase23/releases/download/ensemble_logits/ensemble_logits.pt"
'''
dataset_config = {
    "dataset_name": "tau22",
    "meta_csv": os.path.join(dataset_root, "meta.csv"),
    "train_files_csv": os.path.join("/home/lihaowen/dcase2025_task1_baseline-main/", "evaluation_setup", "split25.csv"),
    #"test_files_csv": os.path.join(dataset_root, "evaluation_setup", "fold1_evaluate.csv"),
    "split_path": "evaluation_setup",
    "split_url": "https://github.com/CPJKU/dcase2024_task1_baseline/releases/download/files/",
    "test_split_csv": "test.csv",
    "eval_dir": os.path.join(dataset_dir, "..", "TAU-urban-acoustic-scenes-2024-mobile-evaluation"),
    "eval_fold_csv": os.path.join(dataset_dir, "..", "TAU-urban-acoustic-scenes-2024-mobile-evaluation",
                                  "evaluation_setup", "fold1_test.csv"),
    "dirs_path": os.path.join("datasets", "dirs"),
    #"logits_file": os.path.join("predictions/ensemble/", "ensemble_logits_2T.csv")
    #"logits_file": os.path.join("predictions/hgl99gad/", "logits_passt_25_augRollDir.csv")
    #"logits_file": os.path.join("predictions/via2gkii/", "logits_cpresnet_25_augRollDir.csv")
    "logits_file": os.path.join("ensemble/", "ensemble_logits_4T.csv")
    #"logits_file": os.path.join("ensemble/", "ensemble_logits_6T.csv")
    #"logits_file": os.path.join("predictions/c5yggw9j/", "logits_passt_25_augRollDir_para2_FMS.csv")
    #"logits_file": os.path.join("predictions/2n8ogfyv/", "logits_passt_25_augRollDir_4761.csv")
    #"logits_file": os.path.join("ensemble/", "ensemble_logits_5T.csv")
    #"logits_file": os.path.join("TFSpec_logits/", "logits_TFSep_25_augRollDir4.csv")
    #"logits_file": os.path.join("ensemble/", "ensemble_logits_5T_TFSpec.csv")
    # "logits_file": os.path.join("ensemble/", "ensemble_logits_6T_33.csv")
    #"logits_file": os.path.join("ensemble/", "ensemble_logits_2T_1C1P.csv")
}


class BasicDCASE22Dataset(TorchDataset):
    """
    Basic DCASE22 Dataset: loads data and caches resampled waveforms
    """

    def __init__(self, meta_csv, sr=32000, cache_path=None):
        """
        @param meta_csv: meta csv file for the dataset
        @param sr: specify sampling rate
        @param cache_path: specify cache path to store resampled waveforms
        return: waveform, file, label, device and city
        """
        df = pd.read_csv(meta_csv, sep="\t")
        le = preprocessing.LabelEncoder()
        self.labels = torch.from_numpy(le.fit_transform(df[['scene_label']].values.reshape(-1)))
        self.devices = le.fit_transform(df[['source_label']].values.reshape(-1))
        self.cities = le.fit_transform(df['identifier'].apply(lambda loc: loc.split("-")[0]).values.reshape(-1))
        # self.files = df[['filename']].values.reshape(-1)
        self.files = df['filename'].apply(lambda x: x.replace("audio/", "")).tolist()
        self.sr = sr
        if cache_path is not None:
            self.cache_path = os.path.join(cache_path, dataset_config["dataset_name"] + f"_r{self.sr}", "files_cache")
            os.makedirs(self.cache_path, exist_ok=True)
        else:
            self.cache_path = None

    def __getitem__(self, index):
        if self.cache_path:
            cpath = os.path.join(self.cache_path, str(index) + ".pt")
            try:
                sig = torch.load(cpath)
            except FileNotFoundError:
                sig, _ = librosa.load(os.path.join(dataset_dir, self.files[index]), sr=self.sr, mono=True)
                sig = torch.from_numpy(sig[np.newaxis])
                torch.save(sig, cpath)
        else:
            sig, _ = librosa.load(os.path.join(dataset_dir, self.files[index]), sr=self.sr, mono=True)
            sig = torch.from_numpy(sig[np.newaxis])
        return sig, self.files[index], self.labels[index], self.devices[index], self.cities[index]

    def __len__(self):
        return len(self.files)


class SimpleSelectionDataset(TorchDataset):
    """A dataset that selects a subsample from a dataset based on a set of sample ids.
        Supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __init__(self, dataset, available_indices):
        """
        @param dataset: dataset to load data from
        @param available_indices: available indices of samples for 'training', 'testing'
        return: waveform, file, label, device, city
        """
        self.available_indices = available_indices
        self.dataset = dataset

    def __getitem__(self, index):
        x, file, label, device, city = self.dataset[self.available_indices[index]]
        return x, file, label, device, city

    def __len__(self):
        return len(self.available_indices)


class DIRAugmentDataset(TorchDataset):
    """
   Augments Waveforms with a Device Impulse Response (DIR)
    """

    def __init__(self, ds, dirs, prob):
        self.ds = ds
        self.dirs = dirs
        self.prob = prob

    def __getitem__(self, index):
        x, file, label, device, city, logits = self.ds[index]

        fsplit = file.rsplit("-", 1)
        device = fsplit[1][:-4]

        if device == 'a' and torch.rand(1) < self.prob:
            # choose a DIR at random
            dir_idx = int(np.random.randint(0, len(self.dirs)))
            dir = self.dirs[dir_idx]

            x = convolve(x, dir, 'full')[:, :x.shape[1]]
            x = torch.from_numpy(x)
        return x, file, label, device, city, logits

    def __len__(self):
        return len(self.ds)


def load_dirs(dirs_path, resample_rate):
    all_paths = [path for path in pathlib.Path(os.path.expanduser(dirs_path)).rglob('*.wav')]
    all_paths = sorted(all_paths)
    all_paths_name = [str(p).rsplit("/", 1)[-1] for p in all_paths]

    print("Augment waveforms with the following device impulse responses:")
    for i in range(len(all_paths_name)):
        print(i, ": ", all_paths_name[i])

    def process_func(dir_file):
        sig, _ = librosa.load(dir_file, sr=resample_rate, mono=True)
        sig = torch.from_numpy(sig[np.newaxis])
        return sig

    return [process_func(p) for p in all_paths]


class RollDataset(TorchDataset):
    """A dataset implementing time rolling.
    """

    def __init__(self, dataset: TorchDataset, shift_range: int, axis=1):
        self.dataset = dataset
        self.shift_range = shift_range
        self.axis = axis

    def __getitem__(self, index):
        x, file, label, device, city, logits = self.dataset[index]
        sf = int(np.random.random_integers(-self.shift_range, self.shift_range))
        return x.roll(sf, self.axis), file, label, device, city, logits

    def __len__(self):
        return len(self.dataset)


class AddLogitsDataset(TorchDataset):
    """A dataset that loads and adds teacher logits to audio samples."""
    def __init__(self, dataset, map_indices, logits_file, temperature=2):
        self.dataset = dataset
        self.map_indices = map_indices

        if logits_file.endswith(".pt"):
            # 兼容旧逻辑
            logits = torch.load(logits_file).float()
            self.logits = F.log_softmax(logits / temperature, dim=-1)
        elif logits_file.endswith(".csv"):
            import pandas as pd
            df = pd.read_csv(logits_file, sep='\t')
            df = df.set_index("filename")
            self.filename_to_logits = {
                fname: F.log_softmax(
                torch.tensor(row.drop("scene_label").values.astype(np.float32))/temperature,
                dim=-1)
                for fname, row in df.iterrows()
                    #values = row.drop("scene_label").values.astype(np.float32)
                    #assert len(values)==10,f"(fname} has {len(values)} logits, expected 10"
            }
        else:
            raise ValueError(f"Unsupported logits file type: {logits_file}")

    def __getitem__(self, index):
        x, file, label, device, city = self.dataset[index]
        if hasattr(self, "logits"):
            logit = self.logits[self.map_indices[index]]
        else:
            key = os.path.basename(file)
            logit = self.filename_to_logits.get(key)
            if logit is None:
                raise KeyError(f"No logits found for file: {key}")
        return x, file, label, device, city, logit

    def __len__(self):
        return len(self.dataset)


# commands to create the datasets for training and testing
# def get_training_set(cache_path=None, resample_rate=32000, roll=False, dir_prob=0, temperature=2):
#     ds = get_base_training_set(dataset_config['meta_csv'], dataset_config['train_files_csv'], cache_path,
#                                resample_rate, temperature)
#     if dir_prob > 0:
#         ds = DIRAugmentDataset(ds, load_dirs(dataset_config['dirs_path'], resample_rate), dir_prob)
#     if roll:
#         ds = RollDataset(ds, shift_range=roll)
#     return ds

def get_training_set(split=100,cache_path=None, resample_rate=32000, roll=False, dir_prob=0, temperature=2):
    assert str(split) in ("5", "10", "25", "50", "100"), "Parameters 'split' must be in [5, 10, 25, 50, 100]"
    os.makedirs(dataset_config['split_path'], exist_ok=True)
    subset_fname = f"split{split}.csv"
    subset_split_file = os.path.join(dataset_config['split_path'], subset_fname)
    if not os.path.isfile(subset_split_file):
        # download split{x}.csv (file containing all audio snippets for respective development-train split)
        subset_csv_url = dataset_config['split_url'] + subset_fname
        print(f"Downloading file: {subset_fname}")
        download_url_to_file(subset_csv_url, subset_split_file)
    ds = get_base_training_set(dataset_config['meta_csv'], subset_split_file, cache_path,
                               resample_rate, temperature)
    if dir_prob > 0:
        ds = DIRAugmentDataset(ds, load_dirs(dataset_config['dirs_path'], resample_rate), dir_prob)
    if roll:
        ds = RollDataset(ds, shift_range=roll)
    return ds
def get_base_training_set(meta_csv, train_files_csv, cache_path, resample_rate, temperature):
    train_files = pd.read_csv(train_files_csv, sep='\t')['filename'].values.reshape(-1)
    meta = pd.read_csv(meta_csv, sep="\t")
    '''
    train_indices = list(meta[meta['filename'].isin(train_files)].index)
    ds = SimpleSelectionDataset(BasicDCASE22Dataset(meta_csv, sr=resample_rate, cache_path=cache_path), train_indices)
    ds = AddLogitsDataset(ds, train_indices, dataset_config['logits_file'], temperature)
    '''
    meta = pd.read_csv(meta_csv, sep="\t")
    train_files_df = pd.read_csv(train_files_csv, sep='\t')
    train_files = train_files_df['filename'].values.reshape(-1)

    # 获取meta中每个train file的索引
    file_to_index = {fn: i for i, fn in enumerate(meta['filename'])}
    train_indices = [file_to_index[fn] for fn in train_files]

    # 创建 dataset
    base_dataset = BasicDCASE22Dataset(meta_csv, sr=resample_rate, cache_path=cache_path)
    ds = SimpleSelectionDataset(base_dataset, train_indices)

    # logits 是按 train_files 的顺序排列的，因此 map_indices 应该是 range(len(train_files))
    map_indices = list(range(len(train_files)))

    ds = AddLogitsDataset(ds, map_indices, dataset_config['logits_file'], temperature)
    return ds


def get_test_set(cache_path=None, resample_rate=32000):
    os.makedirs(dataset_config['split_path'], exist_ok=True)
    test_split_csv = os.path.join(dataset_config['split_path'], dataset_config['test_split_csv'])
    if not os.path.isfile(test_split_csv):
        # download test.csv (file containing all audio snippets for development-test split)
        test_csv_url = dataset_config['split_url'] + dataset_config['test_split_csv']
        print(f"Downloading file: {dataset_config['test_split_csv']}")
        download_url_to_file(test_csv_url, test_split_csv)
    ds = get_base_test_set(dataset_config['meta_csv'], test_split_csv, cache_path,
                           resample_rate)
    return ds


def get_base_test_set(meta_csv, test_files_csv, cache_path, resample_rate):
    test_files = pd.read_csv(test_files_csv, sep='\t')['filename'].values.reshape(-1)
    meta = pd.read_csv(meta_csv, sep="\t")
    test_indices = list(meta[meta['filename'].isin(test_files)].index)
    ds = SimpleSelectionDataset(BasicDCASE22Dataset(meta_csv, sr=resample_rate, cache_path=cache_path), test_indices)
    return ds


