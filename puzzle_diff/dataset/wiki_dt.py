import json
import random
from pathlib import Path

import pandas as pd
import sklearn
import torch
from sklearn.model_selection import KFold
from torch.utils.data import Dataset
from tqdm import tqdm

SEED = 42


class Wiki_dt(Dataset):
    def __init__(self, split="train", n_splits=5, split_idx=1):
        super().__init__()
        data_path = Path(f"datasets/wiki/wiki_movie_plots_deduped.csv")
        rows = load_data(data_path)

        indexes = torch.arange(len(rows))
        kfold = KFold(n_splits, random_state=SEED, shuffle=True)
        self.indexes = [x for x in kfold.split(indexes)][split_idx][
            0 if split == "train" else 1
        ]
        self.examples = rows

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        return self.examples[self.indexes[idx]]


### Reordering task
def load_data(in_file):
    """
    Loads the dataset file:
    in_file: json file
    Returns a list of tuples (input, output)
    """

    df = pd.read_csv(in_file)
    rows = [
        [l for l in x.split(".")[:20] if len(l) > 5 and len(x.split(".")) > 1]
        for x in df["Plot"]
    ]

    return rows


if __name__ == "__main__":
    dt = Wiki_dt(split="test")

    print(dt[0])
