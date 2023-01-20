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
from PIL import Image
from PIL.Image import Resampling
from torch import Tensor
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F


class Text_dataset(pyg_data.Dataset):
    def __init__(
        self,
        dataset=None,
        dataset_get_fn=lambda x: x,
    ) -> None:
        super().__init__()

        assert dataset is not None and dataset_get_fn is not None
        self.dataset = dataset
        self.dataset_get_fn = dataset_get_fn

    def len(self) -> int:
        if self.dataset is not None:
            return len(self.dataset)
        else:
            raise Exception("Dataset not provided")

    def get(self, idx):
        if self.dataset is not None:
            phrases = self.dataset_get_fn(self.dataset[idx])

        # rdim = torch.randint(len(self.patch_per_dim), size=(1,)).item()
        # patch_per_dim = self.patch_per_dim[rdim]

        # height = patch_per_dim[0] * self.patch_size
        # width = patch_per_dim[1] * self.patch_size
        # img = img.resize((width, height), resample=Resampling.BICUBIC)
        # img = self.transforms(img)

        # xy, patches = divide_images_into_patches(img, patch_per_dim, self.patch_size)

        # xy = einops.rearrange(xy, "x y c -> (x y) c")
        # patches = einops.rearrange(patches, "x y c k1 k2 -> (x y) c k1 k2")
        x = torch.linspace(-1, 1, len(phrases))

        adj_mat = torch.ones(len(phrases), len(phrases))
        edge_index, edge_attr = pyg.utils.dense_to_sparse(adj_mat)

        data = pyg_data.Data(
            x=x[:, None],
            phrases_text=phrases,
            edge_index=edge_index,
            ind_name=torch.tensor([idx]).long(),
            num_phrases=torch.tensor([len(phrases)]),
        )
        return data
