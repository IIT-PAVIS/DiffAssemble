import argparse
import glob
import os
import sys

import torch_geometric

sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))

import argparse
import math
import random
import string
import warnings

import matplotlib
import pytorch_lightning as pl
from dataset import dataset_utils as du
from model import spatial_diffusion_3d_test_double_diffusion as sd3d

# from model import spatial_diffusion_3d_only_rotation as sd3d
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
from pytorch_lightning.loggers import WandbLogger

import wandb

# matplotlib.use("qtagg")


warnings.filterwarnings("ignore")


def get_random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = "".join(random.choice(letters) for i in range(length))
    return result_str  # print("Random string of length", length, "is:", result_str)


def main(
    batch_size,
    gpus,
    steps,
    num_workers,
    dataset,
    sampling,
    inference_ratio,
    offline,
    classifier_free_prob,
    classifier_free_w,
    noise_weight,
    data_augmentation,
    checkpoint_path,
    predict_xstart,
    loss_type,
    evaluate,
    visual_pretrained,
    freeze_backbone,
    n_layers,
    backbone,
    lr,
    category,
    max_epochs,
    wandb_id,
    use_vn_dgcnn_equiv_inv_mp,
    max_num_part,
    min_num_part,
    use_6dof_rot,
    architecture,
    missing
):
    ### Define dataset
    train_dt, _, test_dt = du.get_dataset_3d(
        dataset=dataset, category=category, max_num_part=max_num_part, min_num_part=min_num_part, missing=missing
    )

    dl_train = torch_geometric.loader.DataLoader(
        train_dt, batch_size=batch_size, num_workers=num_workers, shuffle=True
    )
    dl_test = torch_geometric.loader.DataLoader(
        test_dt, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )

    model = sd3d.GNN_Diffusion(
        steps=steps,
        sampling=sampling,
        inference_ratio=inference_ratio,
        classifier_free_w=classifier_free_w,
        classifier_free_prob=classifier_free_prob,
        noise_weight=noise_weight,
        model_mean_type=sd3d.ModelMeanType.EPSILON
        if not predict_xstart
        else sd3d.ModelMeanType.START_X,
        visual_pretrained=visual_pretrained,
        freeze_backbone=freeze_backbone,
        n_layers=n_layers,
        loss_type=loss_type,
        backbone=backbone,
        learning_rate=lr,
        max_epochs=max_epochs,
        use_vn_dgcnn_equiv_inv_mp=use_vn_dgcnn_equiv_inv_mp,
        max_num_part=max_num_part,
        use_6dof=use_6dof_rot,
        architecture=architecture
    )

    ### define training

    franklin = True if gpus > 1 else False

    experiment_name = f"3d-{dataset}-{steps}-{get_random_string(6)}-backbone:{backbone}-category:{category}-max_num_partes:{max_num_part}-architecture-{architecture}"

    tags = [f"{dataset}", "3d", f'{"franklin" if franklin else "pc"}', "train"]

    wandb_logger = WandbLogger(
        project="Puzzle-Diff",
        settings=wandb.Settings(code_dir="."),
        offline=offline,
        name=experiment_name,
        entity="puzzle_diff_academic",
        tags=tags,
        id=wandb_id if wandb_id else None,
        resume="must" if wandb_id else None,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="rmse_t_AVG", mode="min", save_top_k=2, save_last=True
    )
    model.initialize_torchmetrics(train_dt.dataset.used_categories)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=gpus,
        strategy="ddp" if gpus > 1 else None,
        max_epochs=max_epochs,
        check_val_every_n_epoch=5,
        logger=wandb_logger,
        num_sanity_val_steps=2,
        callbacks=[checkpoint_callback, ModelSummary(max_depth=2)],
    )
    if wandb_id:
        checkpoint_path = sorted(glob.glob(f"Puzzle-Diff/{wandb_id}/checkpoints/*"))[-1]
        print(checkpoint_path)
    if evaluate:
        model = sd3d.GNN_Diffusion.load_from_checkpoint(checkpoint_path)

        model.initialize_torchmetrics(test_dt.dataset.used_categories)
        model.noise_weight = noise_weight
        model.inference_ratio = inference_ratio
        model.test_dataset = dl_test.dataset
        model.save_eval_images = True
        trainer.test(model, dl_test)
    else:
        trainer.fit(model, dl_train, dl_test, ckpt_path=checkpoint_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    # Add the arguments to the parser
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--dataset", default="breaking-bad", choices=["breaking-bad"])
    ap.add_argument("--sampling", default="DDIM", choices=["DDPM", "DDIM"])
    ap.add_argument("--inference_ratio", type=int, default=10)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--offline", action="store_true", default=False)
    ap.add_argument("--classifier_free_w", type=float, default=0.2)
    ap.add_argument("--classifier_free_prob", type=float, default=0.0)
    ap.add_argument("--data_augmentation", type=str, default="none")
    ap.add_argument("--checkpoint_path", type=str, default="")
    ap.add_argument("--noise_weight", type=float, default=0.0)
    ap.add_argument("--predict_xstart", type=bool, default=True)
    ap.add_argument("--backbone", type=str, default="vn_dgcnn")
    ap.add_argument("--architecture", type=str, default="transformer")
    ap.add_argument("--freeze_backbone", type=bool, default=False)
    ap.add_argument("--visual_pretrained", type=bool, default=True)
    ap.add_argument("--loss_type", type=str, default="all")
    ap.add_argument("--category", type=str, default="")
    ap.add_argument("--evaluate", type=bool, default=False)
    ap.add_argument("--max_epochs", type=int, default=500)
    ap.add_argument("--use_equi_inv", action="store_true", default=False)
    ap.add_argument("--wandb_id", type=str)
    ap.add_argument("--max_num_part", type=int, default=20)
    ap.add_argument("--min_num_part", type=int, default=2)
    ap.add_argument("--use_6dof_rot", action="store_true", default=False)
    ap.add_argument("--missing", type=int, default=0)

    args = ap.parse_args()
    print(args)
    main(
        batch_size=args.batch_size,
        gpus=args.gpus,
        steps=args.steps,
        num_workers=args.num_workers,
        dataset=args.dataset,
        sampling=args.sampling,
        inference_ratio=args.inference_ratio,
        offline=args.offline,
        classifier_free_prob=args.classifier_free_prob,
        classifier_free_w=args.classifier_free_w,
        noise_weight=args.noise_weight,
        data_augmentation=args.data_augmentation,
        checkpoint_path=args.checkpoint_path,
        predict_xstart=args.predict_xstart,
        loss_type=args.loss_type,
        evaluate=args.evaluate,
        freeze_backbone=args.freeze_backbone,
        visual_pretrained=args.visual_pretrained,
        n_layers=args.n_layers,
        backbone=args.backbone,
        lr=args.lr,
        category=args.category,
        max_epochs=args.max_epochs,
        wandb_id=args.wandb_id,
        use_vn_dgcnn_equiv_inv_mp=args.use_equi_inv,
        max_num_part=args.max_num_part,
        min_num_part=args.min_num_part,
        use_6dof_rot=args.use_6dof_rot,
        architecture=args.architecture,
        missing=args.missing
    )
