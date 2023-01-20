import os

import numpy as np
import torch
import torch_geometric as pyg
import torch_geometric.data as pyg_data
from torch.utils.data import Dataset, DataLoader



class Objects_Dataset(pyg_data.Dataset):
    def __init__(
        self,
        dataset=None,
        dataset_get_fn=None,
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
        parts = self.dataset_get_fn(self.dataset[idx])
        valids = torch.tensor(parts["part_valids"].astype(bool))
        num_valids = valids.sum()
        pcds = torch.tensor(parts["part_pcs"]).permute(0, 2, 1)[valids].permute(0, 2, 1)
        quat = parts["part_quat"][valids]
        trans = parts["part_trans"][valids]
        target = torch.tensor(np.hstack((quat, trans)))
        adj_mat = torch.ones(num_valids, num_valids)
        edge_index, edge_attr = pyg.utils.dense_to_sparse(adj_mat)

        data = pyg_data.Data(
            x=target,
            pcds=pcds,
            valids=valids,
            edge_index=edge_index,
            data_id=parts["data_id"],
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


import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNet(nn.Module):
    """PointNet feature extractor.

    Input point clouds [B, N, 3].
    Output per-point feature [B, N, feat_dim] or global feature [B, feat_dim].
    """

    def __init__(self, feat_dim, global_feat=True):
        super().__init__()

        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, feat_dim, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(feat_dim)

        self.global_feat = global_feat

    def forward(self, x):
        """x: [B, N, 3]"""
        print(x.size())
        x = x.transpose(2, 1).contiguous()  # [B, 3, N]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.bn5(self.conv5(x))  # [B, feat_dim, N]
        if self.global_feat:
            feat = x.max(dim=-1)[0]  # [B, feat_dim]
        else:
            feat = x.transpose(2, 1).contiguous()  # [B, N, feat_dim]
        return feat





if __name__ == "__main__":
    data_dict = dict(
        data_dir="/home/sfiorini/Documents/Positional_Puzzle/datasets/breaking-bad",
        data_fn="data_split/everyday.val.txt",
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
    import torch_geometric

    dataset = GeometryPartDataset(**data_dict)
    breakpoint()
    dt = Objects_Dataset(dataset, lambda x: x)
    dl_train = torch_geometric.loader.DataLoader(
        dt, batch_size=2, num_workers=2, shuffle=False
    )

    #dl = pyg.loader.DataLoader(dt, batch_size=2)
    dl_iter = iter(dl_train)

    train_loader = DataLoader(
        dataset=dataset,
        batch_size=2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    dl_iter_2 = iter(train_loader)

    model = PointNet(4)
    model.requires_grad_(False)
    for i in range(1):
        k = next(dl_iter)
        #print(k.pcds.size())
        #breakpoint()
        print(k.pcds)
        print(model(k.pcds))

        k_2 = next(dl_iter_2)
        #breakpoint()
        B, P, N, _ = k_2['part_pcs'].shape  # [B, P, N, 3]
        valid_mask = (k_2['part_valids'] == 1)
        # shared-weight encoder
        valid_pcs = k_2['part_pcs'][valid_mask]  # [n, N, 3]
        print(k.pcds - valid_pcs)
        print(model(valid_pcs))
