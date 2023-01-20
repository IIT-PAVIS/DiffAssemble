import math
import random
from typing import List, Tuple

# import albumentations
# import cv2
import einops
import numpy as np
import torch
import torch_geometric as pyg
import torch_geometric.data as pyg_data
import torch_geometric.loader
import torchvision.transforms as transforms
from PIL import Image, ImageFile
from PIL.Image import Resampling
from torch import Tensor
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F


class Vist_dataset(pyg_data.Dataset):
    def __init__(
        self,
        dataset=None,
        dataset_get_fn=lambda x: x,
    ) -> None:
        super().__init__()

        assert dataset is not None and dataset_get_fn is not None
        self.dataset = dataset
        self.dataset_get_fn = dataset_get_fn

        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    def len(self) -> int:
        if self.dataset is not None:
            return len(self.dataset)
        else:
            raise Exception("Dataset not provided")

    def get(self, idx):
        images, phrases, img_path = self.dataset_get_fn(self.dataset[idx])

        frames = torch.cat([self.transforms(img)[None, :] for img in images])
        x = torch.linspace(-1, 1, len(phrases))

        adj_mat = torch.ones(len(phrases), len(phrases))
        edge_index, edge_attr = pyg.utils.dense_to_sparse(adj_mat)

        data = pyg_data.Data(
            x=x[:, None],
            frames=frames,
            phrases_text=phrases,
            edge_index=edge_index,
            img_path=img_path,
            ind_name=torch.tensor([idx]).long(),
            num_phrases=torch.tensor([len(phrases)]),
        )
        return data


if __name__ == "__main__":
    from sind_vist_dt import Sind_Vist_dt

    train_dt = Sind_Vist_dt(split="train")
    dt = Vist_dataset(train_dt, dataset_get_fn=lambda x: x)

    dl = torch_geometric.loader.DataLoader(dt, batch_size=100)
    dl_iter = iter(dl)

    for i in range(5):
        k = next(dl_iter)
    pass
