import json
from pathlib import Path

from torch.utils.data import Dataset
from tqdm import tqdm


class Roc_dt(Dataset):
    def __init__(self, split="train"):
        super().__init__()
        data_path = Path(f"datasets/roc/{split}.jsonl")
        self.examples = load_data(data_path)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


### Reordering task
def load_data(in_file, task="in_shuf"):
    """
    Loads the dataset file:
    in_file: json file
    Returns a list of tuples (input, output)
    """
    all_lines = []
    with open(in_file, "r", encoding="utf-8") as f:
        for line in f:
            all_lines.append(json.loads(line))

    rows = []

    for x in all_lines:
        line = [x["shuf_sents"][int(i)] for i in x["orig_sents"]]
        rows.append(line)

    return rows


if __name__ == "__main__":
    dt = Roc_dt()
    print(dt[0])
