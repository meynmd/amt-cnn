import os, random
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate


class AMT_Dataset(Dataset):
    def __init__(self, data_root, label_dict_file, phase="train", seed=0, pitch_range=(21, 109)):
        random.seed(seed)
        self.root = data_root
        self.length = 0
        self.xys = []
        self.x_counts = {}
        self.y2x = {}
        self.phase = phase
        self.midi_range=pitch_range

        with open(data_root + "/" + label_dict_file, "r") as fp:
            name_num = [line.strip().split(",") for line in fp.readlines()]
            self.name2idx = {n : int(i) for n, i in name_num}
            self.idx2name = {int(i) : n for n, i in name_num}
        for d in os.listdir(data_root):
            filenames = glob.glob(data_root + "/" + d + "/" + phase + "/*.npy")
            if filenames:
                class_list = [(f, d) for f in filenames]
                self.length += len(class_list)
                self.xys += class_list
                self.x_counts[self.name2idx[d.replace("-", " ")]] = len(class_list)
                self.y2x[d.replace("-", " ")] = class_list

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x_path, y = self.xys[idx]
        x = np.load(x_path)
        # x = (x > 0.).astype(float)
        low, high = self.midi_range
        x = x[low: high, :]

        if self.phase == 'train':
            # transpose by random interval
            t = random.randint(0, 11)
            x = np.roll(x, t, axis=0)
            x[:t, :] = 0.

        x = torch.from_numpy(x).type(torch.float)
        y = y.replace("-", " ")
        y = self.name2idx[y]
        return x, y

    def get_from_class(self, class_label):
        for x_path, y in self.y2x[class_label]:
            x = torch.from_numpy(np.load(x_path))
            low, high = self.midi_range
            x = x[low: high, :]
            x = x.float()
            y = y.replace("-", " ")
            yield (x, self.name2idx[y]), (x_path, y)

    def idx2onehot(self, idx, size):
        vec = torch.FloatTensor(size).fill_(0.)
        vec[idx] = 1.
        return vec

    def get_x_count(self, y):
        return self.x_counts[y]

    def get_y_count(self):
        return len(self.x_counts)

    def get_all_labels(self):
        return self.x_counts.keys()

