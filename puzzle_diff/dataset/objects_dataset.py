import math
import os
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


    

class Objects_Dataset(pyg_data.Dataset):
    def __init__(self, dataset=None, dataset_get_fn=None, missing=0, degree=0) -> None:
        super().__init__()

        assert dataset is not None and dataset_get_fn is not None
        self.dataset = dataset
        self.dataset_get_fn = dataset_get_fn
        self.missing = missing
        self.degree = degree

    def len(self) -> int:
        if self.dataset is not None:
            return len(self.dataset)
        else:
            raise Exception("Dataset not provided")

    def get(self, idx):
        parts = self.dataset_get_fn(self.dataset[idx])
        valids = torch.tensor(parts["part_valids"].astype(bool))
        num_valids = valids.sum()
        pcds = torch.tensor(parts["part_pcs"]).permute(0, 2, 1)[valids].permute(0, 2, 1)
        quat = parts["part_quat"][valids]
        trans = parts["part_trans"][valids]
        target = torch.tensor(
            np.hstack((quat, trans))
        )  # torch.tensor(quat) torch.tensor(trans)
        if self.missing > 0:
            num_pieces = target.shape[0]
            pieces_to_remove = math.ceil(num_pieces * self.missing / 100)
            perm = list(range(num_pieces))
            random.shuffle(perm)
            perm = perm[: num_pieces - pieces_to_remove]
            target = target[perm]
            pcds = pcds[perm]
            adj_mat = torch.ones(target.shape[0], target.shape[0])
            edge_index, edge_attr = pyg.utils.dense_to_sparse(adj_mat)
        else:
            adj_mat = torch.ones(num_valids, num_valids)
            edge_index, edge_attr = pyg.utils.dense_to_sparse(adj_mat)

        if self.degree > 0:
            adj_mat = torch.ones(
                target.shape[0], target.shape[0]
            )

            edge_index, _ = pyg.utils.dense_to_sparse(adj_mat)
            degree = round(
                (int(self.degree) * (target.shape[0] - 1))
                / 100
            )
            n_connections = int(target.shape[0] * degree)
            edge_index = edge_index[:, torch.randperm(edge_index.shape[1])][
                :, :n_connections
            ]
        # adj_mat = torch.ones(num_valids, num_valids)
        # edge_index, edge_attr = pyg.utils.dense_to_sparse(adj_mat)

        category = self.dataset.data_list[idx].split("/")[1]

        data = pyg_data.Data(
            x=target,
            pcds=pcds,
            valids=valids,
            edge_index=edge_index,
            data_id=parts["data_id"],
            category=category,
        )
        return data


@torch.no_grad()
def visualize(cfg):
    # Initialize model
    model = build_model(cfg)
    ckp = torch.load(cfg.exp.weight_file, map_location="cpu")
    model.load_state_dict(ckp["state_dict"])
    model = model.cuda().eval()

    # Initialize dataloaders
    _, val_loader = build_dataloader(cfg)
    val_dst = val_loader.dataset

    # save some predictions for visualization
    vis_lst, loss_lst = [], []
    for batch in tqdm(val_loader):
        batch = {k: v.float().cuda() for k, v in batch.items()}
        out_dict = model(batch)  # trans/rot: [B, P, 3/4/(3, 3)]
        # compute loss to measure the quality of the predictions
        batch["part_rot"] = Rotation3D(batch["part_quat"], rot_type="quat").convert(
            model.rot_type
        )
        loss_dict, _ = model._calc_loss(out_dict, batch)  # loss is [B]
        # the criterion to cherry-pick examples
        loss = loss_dict["rot_pt_l2_loss"] + loss_dict["trans_mae"]
        # convert all the rotations to quaternion for simplicity
        out_dict = {
            "data_id": batch["data_id"].long(),
            "pred_trans": out_dict["trans"],
            "pred_quat": out_dict["rot"].to_quat(),
            "gt_trans": batch["part_trans"],
            "gt_quat": batch["part_rot"].to_quat(),
            "part_valids": batch["part_valids"].long(),
        }
        out_dict = {k: v.cpu().numpy() for k, v in out_dict.items()}
        out_dict_lst = [
            {k: v[i] for k, v in out_dict.items()} for i in range(loss.shape[0])
        ]
        vis_lst += out_dict_lst
        loss_lst.append(loss.cpu().numpy())
    loss_lst = np.concatenate(loss_lst, axis=0)
    top_idx = np.argsort(loss_lst)[: args.vis]

    # apply the predicted transforms to the original meshes and save them
    save_dir = os.path.join(os.path.dirname(cfg.exp.weight_file), "vis", args.category)
    for rank, idx in enumerate(top_idx):
        out_dict = vis_lst[idx]
        data_id = out_dict["data_id"]
        mesh_dir = os.path.join(val_dst.data_dir, val_dst.data_list[data_id])
        mesh_files = os.listdir(mesh_dir)
        mesh_files.sort()
        assert len(mesh_files) == out_dict["part_valids"].sum()
        subfolder_name = (
            f"rank{rank}-{len(mesh_files)}pcs-" f"{mesh_dir.split('/')[-1]}"
        )
        cur_save_dir = os.path.join(save_dir, mesh_dir.split("/")[-2], subfolder_name)
        os.makedirs(cur_save_dir, exist_ok=True)
        for i, mesh_file in enumerate(mesh_files):
            mesh = trimesh.load(os.path.join(mesh_dir, mesh_file))
            mesh.export(os.path.join(cur_save_dir, mesh_file))
            # R^T (mesh - T) --> init_mesh
            gt_trans, gt_quat = out_dict["gt_trans"][i], out_dict["gt_quat"][i]
            gt_rmat = quaternion_to_rmat(gt_quat)
            init_trans = -(gt_rmat.T @ gt_trans)
            init_rmat = gt_rmat.T
            init_pmat = trans_rmat_to_pmat(init_trans, init_rmat)
            init_mesh = mesh.apply_transform(init_pmat)
            init_mesh.export(os.path.join(cur_save_dir, f"input_{mesh_file}"))
            init_pc = trimesh.sample.sample_surface(init_mesh, val_dst.num_points)[0]
            save_pc(init_pc, os.path.join(cur_save_dir, f"input_{mesh_file[:-4]}.ply"))
            # predicted pose
            pred_trans, pred_quat = out_dict["pred_trans"][i], out_dict["pred_quat"][i]
            pred_pmat = trans_quat_to_pmat(pred_trans, pred_quat)
            pred_mesh = init_mesh.apply_transform(pred_pmat)
            pred_mesh.export(os.path.join(cur_save_dir, f"pred_{mesh_file}"))
            pred_pc = trimesh.sample.sample_surface(pred_mesh, val_dst.num_points)[0]
            save_pc(pred_pc, os.path.join(cur_save_dir, f"pred_{mesh_file[:-4]}.ply"))

    print(f"Saving {len(top_idx)} predictions for visualization...")


def visualize_mash(batch, trans, quat):
    data_id = batch["data_id"]
    mesh_dir = os.path.join(
        dataset.dataset.data_dir, dataset.dataset.data_list[data_id]
    )
    mesh_files = os.listdir(mesh_dir)
    mesh_files.sort()

    for i, mesh_file in enumerate(mesh_files):
        mesh = trimesh.load(os.path.join(mesh_dir, mesh_file))
        mesh.export(os.path.join(cur_save_dir, mesh_file))
        # R^T (mesh - T) --> init_mesh

        gt_rmat = quaternion_to_rmat(gt_quat)
        init_trans = -(gt_rmat.T @ gt_trans)
        init_rmat = gt_rmat.T
        init_pmat = trans_rmat_to_pmat(init_trans, init_rmat)
        init_mesh = mesh.apply_transform(init_pmat)
        init_mesh.export(os.path.join(cur_save_dir, f"input_{mesh_file}"))
        init_pc = trimesh.sample.sample_surface(init_mesh, val_dst.num_points)[0]
        save_pc(init_pc, os.path.join(cur_save_dir, f"input_{mesh_file[:-4]}.ply"))
        # predicted pose
        pred_trans, pred_quat = out_dict["pred_trans"][i], out_dict["pred_quat"][i]
        pred_pmat = trans_quat_to_pmat(pred_trans, pred_quat)
        pred_mesh = init_mesh.apply_transform(pred_pmat)
        pred_mesh.export(os.path.join(cur_save_dir, f"pred_{mesh_file}"))
        pred_pc = trimesh.sample.sample_surface(pred_mesh, val_dst.num_points)[0]
        save_pc(pred_pc, os.path.join(cur_save_dir, f"pred_{mesh_file[:-4]}.ply"))


if __name__ == "__main__":
    data_dict = dict(
        data_dir="/home/sfiorini/Documents/Positional_Puzzle/datasets/breaking-bad",
        data_fn="data_split/everyday.train.txt",
        data_keys=("part_ids",),
        category="BeerBottle",  # all
        num_points=1000,
        min_num_part=2,
        max_num_part=2,
        shuffle_parts=False,
        rot_range=-1,
        overfit=-1,
    )
    from breakingbad_dt import GeometryPartDataset

    dataset = GeometryPartDataset(**data_dict)

    dt = Objects_Dataset(dataset, lambda x: x)
    dl = pyg.loader.DataLoader(dt, batch_size=100)
    dl_iter = iter(dl)

    for i in range(5):
        k = next(dl_iter)
        breakpoint()
        print(k)
