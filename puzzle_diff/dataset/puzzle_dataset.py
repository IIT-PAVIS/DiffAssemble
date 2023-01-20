import math
import random
from typing import List, Tuple

import einops
import networkx as nx
import numpy as np
import torch
import torch_geometric as pyg
import torch_geometric.data as pyg_data
import torch_geometric.loader
import torchvision.transforms as transforms
from PIL import Image
from PIL.Image import Resampling
from scipy.sparse.linalg import eigsh
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F

# import albumentations
# import cv2







def generate_random_expander(num_nodes, degree, rng=None, max_num_iters=5, exp_index=0):
    """Generates a random d-regular expander graph with n nodes.
    Returns the list of edges. This list is symmetric; i.e., if
    (x, y) is an edge so is (y,x).
    Args:
      num_nodes: Number of nodes in the desired graph.
      degree: Desired degree.
      rng: random number generator
      max_num_iters: maximum number of iterations
    Returns:
      senders: tail of each edge.
      receivers: head of each edge.
    """
    if isinstance(degree, str):
        degree = round((int(degree[:-1]) * (num_nodes - 1)) / 100)
    num_nodes = num_nodes

    if rng is None:
        rng = np.random.default_rng()
    eig_val = -1
    eig_val_lower_bound = (
        max(0, degree - 2 * math.sqrt(degree - 1) - 0.1) if degree > 0 else 0
    )  # allow the use of zero degree

    max_eig_val_so_far = -1
    max_senders = []
    max_receivers = []
    cur_iter = 1

    # (bave): This is a hack.  This should hopefully fix the bug
    if num_nodes <= degree:
        degree = num_nodes - 1

    # (ali): if there are too few nodes, random graph generation will fail. in this case, we will
    # add the whole graph.
    if num_nodes <= 10:
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    max_senders.append(i)
                    max_receivers.append(j)
    else:
        while eig_val < eig_val_lower_bound and cur_iter <= max_num_iters:
            senders, receivers = generate_random_regular_graph(num_nodes, degree, rng)

            eig_val = get_eigenvalue(senders, receivers, num_nodes=num_nodes)
            if len(eig_val) == 0:
                print(
                    "num_nodes = %d, degree = %d, cur_iter = %d, mmax_iters = %d, senders = %d, receivers = %d"
                    % (
                        num_nodes,
                        degree,
                        cur_iter,
                        max_num_iters,
                        len(senders),
                        len(receivers),
                    )
                )
                eig_val = 0
            else:
                eig_val = eig_val[0]
            if eig_val > max_eig_val_so_far:
                max_eig_val_so_far = eig_val
                max_senders = senders
                max_receivers = receivers

            cur_iter += 1
    max_senders = torch.tensor(max_senders, dtype=torch.long).view(-1, 1)
    max_receivers = torch.tensor(max_receivers, dtype=torch.long).view(-1, 1)
    expander_edges = torch.cat([max_senders, max_receivers], dim=1)
    return expander_edges


def get_eigenvalue(senders, receivers, num_nodes):
    edge_index = torch.tensor(np.stack([senders, receivers]))
    edge_index, edge_weight = get_laplacian(
        edge_index, None, normalization=None, num_nodes=num_nodes
    )
    L = to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes)
    return eigsh(L, k=2, which="SM", return_eigenvectors=False)


def generate_random_regular_graph(num_nodes, degree, rng=None):
    """Generates a random d-regular connected graph with n nodes.
    Returns the list of edges. This list is symmetric; i.e., if
    (x, y) is an edge so is (y,x).
    Args:
      num_nodes: Number of nodes in the desired graph.
      degree: Desired degree.
      rng: random number generator
    Returns:
      senders: tail of each edge.
      receivers: head of each edge.
    """
    if (num_nodes * degree) % 2 != 0:
        raise TypeError("nodes * degree must be even")
    if rng is None:
        rng = np.random.default_rng()
    if degree == 0:
        return np.array([]), np.array([])
    nodes = rng.permutation(np.arange(num_nodes))
    num_reps = degree // 2
    num_nodes = len(nodes)

    ns = np.hstack([np.roll(nodes, i + 1) for i in range(num_reps)])
    edge_index = np.vstack((np.tile(nodes, num_reps), ns))

    if degree % 2 == 0:
        senders, receivers = np.concatenate(
            [edge_index[0], edge_index[1]]
        ), np.concatenate([edge_index[1], edge_index[0]])
        return senders, receivers
    else:
        edge_index = np.hstack(
            (edge_index, np.vstack((nodes[: num_nodes // 2], nodes[num_nodes // 2 :])))
        )
        senders, receivers = np.concatenate(
            [edge_index[0], edge_index[1]]
        ), np.concatenate([edge_index[1], edge_index[0]])
        return senders, receivers


class RandomCropAndResizedToOriginal(transforms.RandomResizedCrop):
    def forward(self, img):
        size = img.size
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return F.resized_crop(img, i, j, h, w, size, self.interpolation)


def _get_augmentation(augmentation_type: str = "none"):
    switch = {
        "weak": [transforms.RandomHorizontalFlip(p=0.5)],
        "hard": [
            transforms.RandomHorizontalFlip(p=0.5),
            RandomCropAndResizedToOriginal(
                size=(1, 1), scale=(0.8, 1), interpolation=InterpolationMode.BICUBIC
            ),
        ],
    }
    return switch.get(augmentation_type, [])


@torch.jit.script
def divide_images_into_patches(
    img, patch_per_dim: List[int], patch_size: int
) -> List[Tensor]:
    # img2 = einops.rearrange(img, "c h w -> h w c")

    # divide images in non-overlapping patches based on patch size
    # output dim -> a
    img2 = img.permute(1, 2, 0)
    patches = img2.unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
    y = torch.linspace(-1, 1, patch_per_dim[0])
    x = torch.linspace(-1, 1, patch_per_dim[1])
    xy = torch.stack(torch.meshgrid(x, y, indexing="xy"), -1)
    # print(patch_per_dim)

    return xy, patches


# generation of a unique graph for each number of nodes
def create_graph(patch_per_dim, degree, unique_graph):
    # Create an empty dictionary
    patch_edge_index_dict = {}
    for patch_dim in patch_per_dim:
        if degree == -1:
            num_patches = patch_dim[0] * patch_dim[1]
            adj_mat = torch.ones(num_patches, num_patches)
            edge_index, _ = adj_mat.nonzero().t().contiguous()
        else:
            num_patches = patch_dim[0] * patch_dim[1]
            edge_index = (
                generate_random_expander(
                    num_nodes=num_patches, degree=degree, rng=unique_graph
                )
                .t()
                .contiguous()
            )
        patch_edge_index_dict[patch_dim] = edge_index
    return patch_edge_index_dict


class Puzzle_Dataset(pyg_data.Dataset):
    def __init__(
        self,
        dataset=None,
        dataset_get_fn=None,
        patch_per_dim=[(7, 6)],
        patch_size=32,
        augment="",
        degree=-1,
        unique_graph=None,
        random=False,
    ) -> None:
        super().__init__()

        assert dataset is not None and dataset_get_fn is not None
        self.dataset = dataset
        self.dataset_get_fn = dataset_get_fn
        self.patch_per_dim = patch_per_dim
        self.unique_graph = unique_graph
        self.augment = augment
        self.random = random

        self.transforms = transforms.Compose(
            [
                *_get_augmentation(augment),
                transforms.ToTensor(),
            ]
        )
        self.patch_size = patch_size
        self.degree = degree

        if self.unique_graph is not None:
            self.edge_index = create_graph(
                self.patch_per_dim, self.degree, self.unique_graph
            )

    def len(self) -> int:
        if self.dataset is not None:
            return len(self.dataset)
        else:
            raise Exception("Dataset not provided")

    def get(self, idx):
        if self.dataset is not None:
            img = self.dataset_get_fn(self.dataset[idx])

        rdim = torch.randint(len(self.patch_per_dim), size=(1,)).item()
        patch_per_dim = self.patch_per_dim[rdim]

        height = patch_per_dim[0] * self.patch_size
        width = patch_per_dim[1] * self.patch_size
        img = img.resize((width, height))#, resample=Resampling.BICUBIC)
        img = self.transforms(img)

        xy, patches = divide_images_into_patches(img, patch_per_dim, self.patch_size)

        xy = einops.rearrange(xy, "x y c -> (x y) c")

        indexes = torch.arange(patch_per_dim[0] * patch_per_dim[1]).reshape(
            xy.shape[:-1]
        )
        patches = einops.rearrange(patches, "x y c k1 k2 -> (x y) c k1 k2")
        if self.random:
            patches = patches[torch.randperm(len(patches))]
        if self.degree == -1:
            adj_mat = torch.ones(
                patch_per_dim[0] * patch_per_dim[1], patch_per_dim[0] * patch_per_dim[1]
            )

            edge_index, _ = pyg.utils.dense_to_sparse(adj_mat)
        else:
            if not self.unique_graph:
                edge_index = generate_random_expander(
                    patch_per_dim[0] * patch_per_dim[1], self.degree
                ).T
        data = pyg_data.Data(
            x=xy,
            indexes=indexes,
            patches=patches,
            edge_index=self.edge_index[patch_per_dim]
            if self.unique_graph
            else edge_index,
            ind_name=torch.tensor([idx]).long(),
            patches_dim=torch.tensor([patch_per_dim]),
        )
        return data


class Puzzle_Dataset_Pad(Puzzle_Dataset):
    def __init__(
        self,
        dataset=None,
        dataset_get_fn=None,
        patch_per_dim=[(7, 6)],
        patch_size=32,
        padding=0,
        augment=False,
        degree=-1,
        unique_graph=None,
    ) -> None:
        super().__init__(
            dataset=dataset,
            dataset_get_fn=dataset_get_fn,
            patch_per_dim=patch_per_dim,
            patch_size=patch_size,
            augment=augment,
            degree=degree,
            unique_graph=unique_graph,
        )
        self.padding = padding

    def zero_margin(self, tensor):
        # Set the border elements of each batch image to zero
        tensor[:, :, : self.padding, :] = 0  # Top rows
        tensor[:, :, -self.padding :, :] = 0  # Bottom rows
        tensor[:, :, :, : self.padding] = 0  # Leftmost columns
        tensor[:, :, :, -self.padding :] = 0  # Rightmost columns
        return tensor

    def get(self, idx):
        if self.dataset is not None:
            img = self.dataset_get_fn(self.dataset[idx])

        rdim = torch.randint(len(self.patch_per_dim), size=(1,)).item()
        patch_per_dim = self.patch_per_dim[rdim]

        height = patch_per_dim[0] * self.patch_size
        width = patch_per_dim[1] * self.patch_size

        img = img.resize((width, height))#, resample=Resampling.BICUBIC)

        img = self.trans

        forms(img)
        xy, patches = divide_images_into_patches(img, patch_per_dim, self.patch_size)

        xy = einops.rearrange(xy, "x y c -> (x y) c")
        patches = einops.rearrange(patches, "x y c k1 k2 -> (x y) c k1 k2")
        if self.padding > 0:
            patches = self.zero_margin(patches)
        indexes = torch.arange(patch_per_dim[0] * patch_per_dim[1]).reshape(
            xy.shape[:-1]
        )
        if self.degree == -1:
            adj_mat = torch.ones(
                patch_per_dim[0] * patch_per_dim[1], patch_per_dim[0] * patch_per_dim[1]
            )

            edge_index, _ = pyg.utils.dense_to_sparse(adj_mat)
        else:
            if not self.unique_graph:
                edge_index = generate_random_expander(
                    patch_per_dim[0] * patch_per_dim[1], self.degree
                ).T
        data = pyg_data.Data(
            x=xy,
            indexes=indexes,
            patches=patches,
            edge_index=self.edge_index[patch_per_dim]
            if self.unique_graph
            else edge_index,
            ind_name=torch.tensor([idx]).long(),
            patches_dim=torch.tensor([patch_per_dim]),
        )
        return data


class Puzzle_Dataset_ROT_MP(Puzzle_Dataset):
    def __init__(
        self,
        dataset=None,
        dataset_get_fn=None,
        patch_per_dim=[(7, 6)],
        patch_size=32,
        augment=False,
        concat_rot=True,
        missing_perc=10,
    ) -> None:
        super().__init__(
            dataset=dataset,
            dataset_get_fn=dataset_get_fn,
            patch_per_dim=patch_per_dim,
            patch_size=patch_size,
            augment=augment,
        )
        self.concat_rot = concat_rot
        self.missing_pieces_perc = missing_perc

    def get(self, idx):
        if self.dataset is not None:
            img = self.dataset_get_fn(self.dataset[idx])

        rdim = torch.randint(len(self.patch_per_dim), size=(1,)).item()
        patch_per_dim = self.patch_per_dim[rdim]

        height = patch_per_dim[0] * self.patch_size
        width = patch_per_dim[1] * self.patch_size

        img = img.resize((width, height))#, resample=Resampling.BICUBIC)

        img = self.transforms(img)
        xy, patches = divide_images_into_patches(img, patch_per_dim, self.patch_size)

        xy = einops.rearrange(xy, "x y c -> (x y) c")
        patches = einops.rearrange(patches, "x y c k1 k2 -> (x y) c k1 k2")

        patches_num = patches.shape[0]

        patches_numpy = (
            (patches * 255).long().numpy().transpose(0, 2, 3, 1).astype(np.uint8)
        )
        patches_im = [Image.fromarray(patches_numpy[x]) for x in range(patches_num)]
        random_rot = torch.randint(low=0, high=4, size=(patches_num,))
        random_rot_one_hot = torch.nn.functional.one_hot(random_rot, 4)

        # rotation classes : 0 -> no rotation
        #                   1 -> 90 degrees
        #                   2 -> 180 degrees
        #                   3 -> 270 degrees

        indexes = torch.arange(patch_per_dim[0] * patch_per_dim[1]).reshape(
            xy.shape[:-1]
        )

        rots = torch.tensor(
            [
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],
            ]
        )

        rots_tensor = random_rot_one_hot @ rots
        rotated_patch = [
            x.rotate(rot * 90) for (x, rot) in zip(patches_im, random_rot)
        ]  # in PIL

        rotated_patch_tensor = [
            torch.tensor(np.array(patch)).permute(2, 0, 1).float() / 255
            for patch in rotated_patch
        ]

        patches = torch.stack(rotated_patch_tensor)
        if self.concat_rot:
            xy = torch.cat([xy, rots_tensor], 1)

        num_pieces = xy.shape[0]
        pieces_to_remove = math.ceil(num_pieces * self.missing_pieces_perc / 100)

        perm = list(range(num_pieces))

        random.shuffle(perm)
        perm = perm[: num_pieces - pieces_to_remove]
        xy = xy[perm]
        patches = patches[perm]

        adj_mat = torch.ones(xy.shape[0], xy.shape[0])
        edge_index, edge_attr = pyg.utils.dense_to_sparse(adj_mat)

        data = pyg_data.Data(
            x=xy,
            indexes=indexes,
            rot=rots_tensor,
            rot_index=random_rot,
            patches=patches,
            edge_index=edge_index,
            ind_name=torch.tensor([idx]).long(),
            patches_dim=torch.tensor([patch_per_dim]),
        )
        return data


class Puzzle_Dataset_MP(Puzzle_Dataset):
    def __init__(
        self,
        dataset=None,
        dataset_get_fn=None,
        patch_per_dim=[(7, 6)],
        patch_size=32,
        missing_perc=10,
        augment=False,
    ) -> None:
        super().__init__(
            dataset=dataset,
            dataset_get_fn=dataset_get_fn,
            patch_per_dim=patch_per_dim,
            patch_size=patch_size,
            augment=augment,
        )
        self.missing_pieces_perc = missing_perc

    def get(self, idx):
        if self.dataset is not None:
            img = self.dataset_get_fn(self.dataset[idx])

        rdim = torch.randint(len(self.patch_per_dim), size=(1,)).item()
        patch_per_dim = self.patch_per_dim[rdim]

        height = patch_per_dim[0] * self.patch_size
        width = patch_per_dim[1] * self.patch_size

        img = img.resize((width, height))#, resample=Resampling.BICUBIC)

        img = self.transforms(img)
        xy, patches = divide_images_into_patches(img, patch_per_dim, self.patch_size)

        xy = einops.rearrange(xy, "x y c -> (x y) c")
        patches = einops.rearrange(patches, "x y c k1 k2 -> (x y) c k1 k2")

        num_pieces = xy.shape[0]
        pieces_to_remove = math.ceil(num_pieces * self.missing_pieces_perc / 100)

        perm = list(range(num_pieces))

        random.shuffle(perm)
        perm = perm[: num_pieces - pieces_to_remove]
        xy = xy[perm]
        patches = patches[perm]

        adj_mat = torch.ones(xy.shape[0], xy.shape[0])
        edge_index, edge_attr = pyg.utils.dense_to_sparse(adj_mat)
        data = pyg_data.Data(
            x=xy,
            patches=patches,
            edge_index=edge_index,
            ind_name=torch.tensor([idx]).long(),
            patches_dim=torch.tensor([patch_per_dim]),
        )
        return data


class Puzzle_Dataset_ROT(Puzzle_Dataset):
    def __init__(
        self,
        dataset=None,
        dataset_get_fn=None,
        patch_per_dim=[(7, 6)],
        patch_size=32,
        augment=False,
        concat_rot=True,
        degree=-1,
        unique_graph=None,
        all_equivariant=False,
        random_dropout=False,
    ) -> None:
        super().__init__(
            dataset=dataset,
            dataset_get_fn=dataset_get_fn,
            patch_per_dim=patch_per_dim,
            patch_size=patch_size,
            augment=augment,
            degree=degree,
            unique_graph=unique_graph,
        )
        self.concat_rot = concat_rot
        self.degree = degree
        self.all_equivariant = all_equivariant
        self.unique_graph = unique_graph
        self.random_dropout = random_dropout
        if self.unique_graph is not None:
            self.edge_index = create_graph(
                self.patch_per_dim, self.degree, self.unique_graph
            )

    def get(self, idx):
        if self.dataset is not None:
            img = self.dataset_get_fn(self.dataset[idx])

        rdim = torch.randint(len(self.patch_per_dim), size=(1,)).item()
        patch_per_dim = self.patch_per_dim[rdim]

        height = patch_per_dim[0] * self.patch_size
        width = patch_per_dim[1] * self.patch_size

        img = img.resize((width, height), resample=Resampling.LANCZOS)#, resample=Resampling.BICUBIC)

        img = self.transforms(img)
        xy, patches = divide_images_into_patches(img, patch_per_dim, self.patch_size)

        xy = einops.rearrange(xy, "x y c -> (x y) c")
        patches = einops.rearrange(patches, "x y c k1 k2 -> (x y) c k1 k2")

        patches_num = patches.shape[0]

        patches_numpy = (
            (patches * 255).long().numpy().transpose(0, 2, 3, 1).astype(np.uint8)
        )
        patches_im = [Image.fromarray(patches_numpy[x]) for x in range(patches_num)]
        random_rot = torch.randint(low=0, high=4, size=(patches_num,))
        random_rot_one_hot = torch.nn.functional.one_hot(random_rot, 4)

        # if self.degree == '100%':

        if self.degree == -1 or self.degree == "100%":
            adj_mat = torch.ones(
                patch_per_dim[0] * patch_per_dim[1], patch_per_dim[0] * patch_per_dim[1]
            )

            edge_index, _ = pyg.utils.dense_to_sparse(adj_mat)
        elif self.random_dropout:
            adj_mat = torch.ones(
                patch_per_dim[0] * patch_per_dim[1], patch_per_dim[0] * patch_per_dim[1]
            )

            edge_index, _ = pyg.utils.dense_to_sparse(adj_mat)
            degree = round(
                (int(self.degree[:-1]) * (int(patch_per_dim[0] * patch_per_dim[1]) - 1))
                / 100
            )
            n_connections = int(patch_per_dim[0] * patch_per_dim[1] * degree)
            edge_index = edge_index[:, torch.randperm(edge_index.shape[1])][
                :, :n_connections
            ]

        else:
            if not self.unique_graph:
                edge_index = generate_random_expander(
                    patch_per_dim[0] * patch_per_dim[1], self.degree
                ).T

        # rotation classes : 0 -> no rotation
        #                   1 -> 90 degrees
        #                   2 -> 180 degrees
        #                   3 -> 270 degrees

        indexes = torch.arange(patch_per_dim[0] * patch_per_dim[1]).reshape(
            xy.shape[:-1]
        )

        rots = torch.tensor(
            [
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],
            ]
        )

        rots_tensor = random_rot_one_hot @ rots

        # ruoto l'immagine casualmente

        rotated_patch = [
            x.rotate(rot * 90) for (x, rot) in zip(patches_im, random_rot)
        ]  # in PIL

        if self.all_equivariant:
            rotated_patch_1 = [
                [x.rotate(rot * 90) for rot in range(4)] for x in rotated_patch
            ]  # type: ignore

            rotated_patch_tensor = [
                [
                    torch.tensor(np.array(patch)).permute(2, 0, 1).float() / 255
                    for patch in test
                ]
                for test in rotated_patch_1
            ]
        else:
            rotated_patch_tensor = [
                torch.tensor(np.array(patch)).permute(2, 0, 1).float() / 255
                for patch in rotated_patch
            ]

        patches = (
            torch.stack([torch.stack(i) for i in rotated_patch_tensor])
            if self.all_equivariant
            else torch.stack(rotated_patch_tensor)
        )
        if self.concat_rot:
            xy = torch.cat([xy, rots_tensor], 1)

        data = pyg_data.Data(
            x=xy,
            indexes=indexes,
            rot=rots_tensor,
            rot_index=random_rot,
            patches=patches,
            edge_index=self.edge_index[patch_per_dim]
            if self.unique_graph
            else edge_index,
            ind_name=torch.tensor([idx]).long(),
            patches_dim=torch.tensor([patch_per_dim]),
        )
        return data


if __name__ == "__main__":
    from celeba_dt import CelebA_HQ

    train_dt = CelebA_HQ(train=True)
    dt = Puzzle_Dataset_ROT(
        train_dt, dataset_get_fn=lambda x: x[0], patch_per_dim=[(4, 4)]
    )

    dl = torch_geometric.loader.DataLoader(dt, batch_size=100)
    dl_iter = iter(dl)

    for i in range(5):
        k = next(dl_iter)
    pass
