import argparse
import os
import random
import string
import sys

import torch
import torch_geometric

sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))

import argparse
import logging
import math

import pytorch_lightning as pl
from dataset import dataset_utils as du
from model import spatial_diffusion as sd
from model import spatial_diffusion_discrete as sdd
from model import spatial_diffusion_discrete_rot as sdd_rot
from model.spatial_diffusion import GNN_Diffusion
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
from pytorch_lightning.loggers import WandbLogger

import wandb


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
    puzzle_sizes,
    sampling,
    inference_ratio,
    offline,
    noise_weight,
    checkpoint_path,
    discrete,
    rotation,
):
    ### Define dataset

    if rotation:
        _, test_dt, puzzle_sizes = du.get_dataset_ROT(
            dataset=dataset,
            puzzle_sizes=puzzle_sizes,
        )
    else:
        _, test_dt, puzzle_sizes = du.get_dataset(
            dataset=dataset, puzzle_sizes=puzzle_sizes
        )

    dl_test = torch_geometric.loader.DataLoader(
        test_dt, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )

    if discrete and rotation:
        model_cls = sdd_rot.GNN_Diffusion
    elif discrete:
        model_cls = sdd.GNN_Diffusion
    else:
        model_cls = sd.GNN_Diffusion

    model = model_cls.load_from_checkpoint(checkpoint_path)
    model.noise_weight = noise_weight
    model.inference_ratio = inference_ratio
    model.initialize_torchmetrics(puzzle_sizes)
    model.steps = steps
    ### define training

    franklin = True if gpus > 1 else False

    experiment_name = f"eval-{dataset}-{puzzle_sizes}-{steps}-{get_random_string(6)}"

    tags = [f"{dataset}", f'{"franklin" if franklin else "fisso"}', "train"]

    wandb_logger = WandbLogger(
        project="Puzzle-Diff",
        settings=wandb.Settings(code_dir="."),
        offline=True,
        name=experiment_name,
        tags=tags,
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=gpus,
        strategy="ddp" if gpus > 1 else None,
        logger=wandb_logger,
        callbacks=[ModelSummary(max_depth=2)],
    )
    logging.warning(f"Saving to {experiment_name}")
    trainer.predict(model, dl_test)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    # Add the arguments to the parser
    ap.add_argument("-batch_size", type=int, default=10)
    ap.add_argument("-gpus", type=int, default=1)
    ap.add_argument("-steps", type=int, default=300)
    ap.add_argument("-num_workers", type=int, default=8)
    ap.add_argument(
        "-dataset", default="wikiart", choices=["celeba", "wikiart", "cifar100"]
    )
    ap.add_argument("-sampling", default="DDIM", choices=["DDPM", "DDIM"])
    ap.add_argument("-inference_ratio", type=int, default=10)
    ap.add_argument(
        "-puzzle_sizes",
        nargs="+",
        default=[12],
        type=int,
        help="Input a list of values",
    )
    ap.add_argument("--offline", action="store_true", default=False)
    ap.add_argument("--noise_weight", type=float, default=1.0)
    ap.add_argument("--checkpoint_path", type=str, default="")
    ap.add_argument("--discrete", type=bool, default=False)
    ap.add_argument("--rotation", type=bool, default=False)

    args = ap.parse_args()
    print(args)
    main(
        batch_size=args.batch_size,
        gpus=args.gpus,
        steps=args.steps,
        num_workers=args.num_workers,
        dataset=args.dataset,
        puzzle_sizes=args.puzzle_sizes,
        sampling=args.sampling,
        inference_ratio=args.inference_ratio,
        offline=args.offline,
        noise_weight=args.noise_weight,
        checkpoint_path=args.checkpoint_path,
        discrete=args.discrete,
        rotation=args.rotation,
    )
