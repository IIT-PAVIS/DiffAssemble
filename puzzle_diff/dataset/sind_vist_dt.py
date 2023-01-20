import glob
import imghdr
import json
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib
import requests
from matplotlib import pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map  # or thread_map


def download_img(image_url):

    id_jpg = image_url.split("/")[-1].split("_")[0] + ".jpg"
    img_path = Path(f"/data/vist_large/{id_jpg}")
    if img_path.exists():
        return
    req = requests.get(image_url)
    img_data = req.content
    if req.status_code != 200:
        print(image_url)
        return
    with open(f"/data/vist/images/{id_jpg}", "wb") as handler:
        handler.write(img_data)


class Sind_Vist_dt(Dataset):
    def __init__(self, download=False, split="train"):
        super().__init__()
        data_path = Path(f"datasets/sind/{split}.story-in-sequence.json")
        images_path = Path(f"datasets/vist/images")

        if download:
            download_images(data_path)

        self.examples = load_data(data_path, images_path)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        images_paths, sentences = self.examples[idx]
        images = [
            Image.open(img_path).convert("RGB").resize((64, 64))
            for img_path in images_paths
        ]
        return images, sentences, images_paths


### Reordering task
def load_data(in_file, images_path: Path):
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
    count = 0

    for i in range(0, len(annotations), 5):
        line = [annotations[i + d][0]["original_text"] for d in range(5)]
        img_for_line = []

        for d in range(5):
            img_path = Path(
                str(images_path / f"{annotations[i+d][0]['photo_flickr_id']}.jpg")
            )
            if not img_path.exists():
                break

            img_for_line.append(img_path)

        if len(img_for_line) != len(line):
            continue
        rows.append((img_for_line, line))

    return rows


### Reordering task
def resize_data(in_file, images_path: Path):
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
    count = 0

    def resize_img(img_path):
        if not img_path.exists():
            return
        if Path(f"/data/dst/{img_path.name}").exists():
            return

        try:
            img = Image.open(img_path).resize((128, 128)).convert("RGB")
            img.save(f"/data/dst/{img_path.name}")
        except:
            return

    images = []
    for i in tqdm(range(0, len(annotations), 5)):
        line = [annotations[i + d][0]["original_text"] for d in range(5)]
        img_for_line = []

        for d in range(5):
            img_path = Path(
                str(images_path / f"{annotations[i+d][0]['photo_flickr_id']}.jpg")
            )
            images.append(img_path)

    thread_map(resize_img, images)

    return rows


def download_images(in_file):
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

    thread_map(download_img, images)


if __name__ == "__main__":

    dt = Sind_Vist_dt(download=True, split="test")
    x = dt[100]
    for img in x[0]:
        plt.figure()
        plt.imshow(img)
        plt.show()

    print(x)
