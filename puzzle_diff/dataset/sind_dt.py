import json
import pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from torch.utils.data import Dataset
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map  # or thread_map


def download_img(image_url):
    req = requests.get(image_url)
    id_jpg = image_url.split("/")[-1]
    img_data = req.content
    with open(f"/data/sind_large/{id_jpg}", "wb") as handler:
        handler.write(img_data)


class Sind_dt(Dataset):
    def __init__(self, split="train"):
        super().__init__()
        data_path = Path(f"datasets/sind/{split}.story-in-sequence.json")
        self.examples = load_data(data_path)
        download_images(data_path)

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
    annotations = all_lines[0]["annotations"]
    rows = []

    for i in range(0, len(annotations), 5):
        line = [annotations[i + d][0]["original_text"] for d in range(5)]
        rows.append(line)

    return rows


def download_images(in_file, data_path="/data/sind"):
    all_lines = []
    correct_annotations = {}
    with open(in_file, "r", encoding="utf-8") as f:
        for line in f:
            all_lines.append(json.loads(line))
    images = []
    for x in all_lines[0]["images"]:
        if "url_o" in x:
            images.append(x["url_o"])
        elif "url_m" in x:
            images.append(x["url_m"])

    process_map(download_img, images, chunksize=1)


if __name__ == "__main__":
    dt = Sind_dt()
    print(dt[0])
