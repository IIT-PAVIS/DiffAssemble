import argparse
import os
import sys
import glob
import torch_geometric

sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))

import argparse
import math
import random
import string


import pytorch_lightning as pl
from dataset.dataset_utils import get_dataset, get_dataset_ROT

# from model import spatial_diffusion as sd
from model import spatial_diffusion_on_angle as sd_angle

import matplotlib
import pytorch_lightning as pl
from dataset import dataset_utils as du
from model import spatial_diffusion as sd
from model import spatial_diffusion_discrete as sdd
from model import spatial_diffusion_discrete_rot as sdd_rot

from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
from pytorch_lightning.loggers import WandbLogger

import wandb


def get_random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = "".join(random.choice(letters) for i in range(length))
    return result_str  # print("Random string of length", length, "is:", result_str)


class Percent(object):
    def __new__(self, percent_string):
        if percent_string.endswith("%"):
            return str(percent_string)
        else:
            return int(percent_string)


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
    classifier_free_prob,
    classifier_free_w,
    noise_weight,
    data_augmentation,
    checkpoint_path,
    rotation,
    only_rotation,
    predict_xstart,
    evaluate,
    angle_type,
    discrete,
    loss_type,
    cold_diffusion,
    visual_pretrained,
    freeze_backbone,
    backbone,
    n_layers,
    architecture,
    degree,
    virt_nodes,
    max_epochs,
    unique_graph,
    inf_fully,
    all_equivariant,
    wandb_id,
    padding,
    random_dropout,
    acc_grad,
    save_eval_images=False,
    missing=0
):
    ### Define dataset
    if rotation:
        train_dt, test_dt, puzzle_sizes = du.get_dataset_ROT(
            dataset=dataset,
            puzzle_sizes=puzzle_sizes,
            augment=data_augmentation,
            degree=degree,
            unique_graph=unique_graph,
            inf_fully=inf_fully,
            all_equivariant=all_equivariant,
            random_dropout=random_dropout,
            missing=missing
        )
    elif padding:
        train_dt, test_dt, puzzle_sizes = du.get_dataset_padding(
            dataset=dataset,
            puzzle_sizes=puzzle_sizes,
            augment=data_augmentation,
            degree=degree,
            inf_fully=inf_fully,
            padding=padding,
        )

    else:
        train_dt, test_dt, puzzle_sizes = du.get_dataset(
            dataset=dataset,
            puzzle_sizes=puzzle_sizes,
            augment=data_augmentation,
            degree=degree,
            unique_graph=unique_graph,
            inf_fully=inf_fully,
        )

    dl_train = torch_geometric.loader.DataLoader(  # type: ignore
        train_dt, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )
    dl_test = torch_geometric.loader.DataLoader(  # type: ignore
        test_dt, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )

    if discrete and rotation:
        model = sdd_rot.GNN_Diffusion(
            steps=steps,
            sampling=sampling,
            inference_ratio=inference_ratio,
            classifier_free_w=classifier_free_w,
            classifier_free_prob=classifier_free_prob,
            noise_weight=noise_weight,
            rotation=rotation,
            model_mean_type=sd.ModelMeanType.START_X,
            puzzle_sizes=puzzle_sizes,
            scheduler=sd.ModelScheduler.LINEAR,
            loss_type=loss_type,
            only_rotation=only_rotation,
            cold_diffusion=cold_diffusion,
            virt_nodes=virt_nodes,
        )
    elif discrete:
        model = sdd.GNN_Diffusion(
            steps=steps,
            sampling=sampling,
            inference_ratio=inference_ratio,
            classifier_free_w=classifier_free_w,
            classifier_free_prob=classifier_free_prob,
            noise_weight=noise_weight,
            rotation=rotation,
            model_mean_type=sd.ModelMeanType.START_X,
            puzzle_sizes=puzzle_sizes,
            scheduler=sd.ModelScheduler.LINEAR,
            loss_type=loss_type,
        )
    else:
        model = sd.GNN_Diffusion(
            steps=steps,
            sampling=sampling,
            inference_ratio=inference_ratio,
            classifier_free_w=classifier_free_w,
            classifier_free_prob=classifier_free_prob,
            noise_weight=noise_weight,
            rotation=rotation,
            model_mean_type=sd.ModelMeanType.EPSILON
            if not predict_xstart
            else sd.ModelMeanType.START_X,
            visual_pretrained=visual_pretrained,
            freeze_backbone=freeze_backbone,
            backbone=backbone,
            n_layers=n_layers,
            architecture=architecture,
            virt_nodes=virt_nodes,
            all_equivariant=all_equivariant,
        )

    model.initialize_torchmetrics(puzzle_sizes)

    ### define training

    franklin = True if gpus > 1 else False

    experiment_name = f"{dataset}-{puzzle_sizes}-{steps}-degree:{degree}-virtnode:{virt_nodes}-{get_random_string(6)}-{'discrete' if discrete else 'continuous'}-{'random_dropout' if random_dropout else 'ours_dropout'}-arch-{architecture}"

    if rotation:
        experiment_name = "ROT-" + experiment_name + f"backbone:{backbone}"

    if padding:
        experiment_name = "PADDING-" + experiment_name

    tags = [f"{dataset}", f'{"franklin" if franklin else "fisso"}', "train"]

    wandb_logger = WandbLogger(
        project="Puzzle-Diff",
        settings=wandb.Settings(code_dir="."),
        offline=offline,
        name=experiment_name,
        # entity="puzzle_diff",
        entity="puzzle_diff_academic",
        tags=tags,
        id=wandb_id if wandb_id else None,
        resume="must" if wandb_id else None,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="overall_acc", mode="max", save_top_k=2, save_last=True
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=gpus,
        accumulate_grad_batches=acc_grad if acc_grad > 0 else None,
        strategy="ddp" if gpus > 1 else None,
        check_val_every_n_epoch=5,
        logger=wandb_logger,
        num_sanity_val_steps=2,
        callbacks=[checkpoint_callback, ModelSummary(max_depth=2)],
        max_epochs=max_epochs,
    )
    if wandb_id:
        checkpoint_path = sorted(glob.glob(f"Puzzle-Diff/{wandb_id}/checkpoints/*"))[-1]
        print(checkpoint_path)
    if evaluate:
        model = sd.GNN_Diffusion.load_from_checkpoint(checkpoint_path)
        model.initialize_torchmetrics(puzzle_sizes)
        model.noise_weight = noise_weight
        model.inference_ratio = inference_ratio
        model.save_eval_images = save_eval_images

        trainer.test(model, dl_test)
    else:
        trainer.fit(model, dl_train, dl_test, ckpt_path=checkpoint_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    # Add the arguments to the parser
    ap.add_argument("-batch_size", type=int, default=6)
    ap.add_argument("-gpus", type=int, default=1)
    ap.add_argument("-steps", type=int, default=300)
    ap.add_argument("-num_workers", type=int, default=8)
    ap.add_argument("-max_epochs", type=int, default=1000)
    ap.add_argument(
        "-dataset",
        default="wikiart",
        choices=["celeba", "wikiart", "cifar100", "coco", "imagenet"],
    )
    ap.add_argument("-sampling", default="DDIM", choices=["DDPM", "DDIM"])
    ap.add_argument("-inference_ratio", type=int, default=10)

    # ap.add_argument("--degree", type=int, default=-1)
    ap.add_argument("--degree", type=Percent, default="100%")

    ap.add_argument("--virt_nodes", type=int, default=4)
    ap.add_argument("--unique_graph", type=bool, default=False)
    ap.add_argument("--inf_fully", type=bool, default=False)

    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument(
        "-puzzle_sizes", nargs="+", default=[6], type=int, help="Input a list of values"
    )

    ap.add_argument("--offline", action="store_true", default=False)
    ap.add_argument("--wandb_id", type=str)

    ap.add_argument("--classifier_free_w", type=float, default=0.2)
    ap.add_argument("--classifier_free_prob", type=float, default=0.0)
    ap.add_argument("--data_augmentation", type=str, default="none")
    ap.add_argument("--checkpoint_path", type=str, default="")
    ap.add_argument("--noise_weight", type=float, default=0.0)
    ap.add_argument("--predict_xstart", type=bool, default=False)
    ap.add_argument("--rotation", type=bool, default=False)
    ap.add_argument("--only_rotation", action="store_true", default=False)
    ap.add_argument("--angle_type", type=str, default="radian")
    ap.add_argument("--freeze_backbone", type=bool, default=False)
    ap.add_argument("--visual_pretrained", type=bool, default=True)
    ap.add_argument("--discrete", type=bool, default=False)
    ap.add_argument("--cold_diffusion", type=bool, default=False)
    ap.add_argument("--loss_type", type=str, default="cross_entropy")
    ap.add_argument("--backbone", type=str, default="efficientnet_b0")
    ap.add_argument("--architecture", type=str, default="transformer")
    ap.add_argument("--all_equivariant", type=bool, default=False)
    ap.add_argument("--evaluate", type=bool, default=False)
    ap.add_argument("--padding", type=int, default=0)
    ap.add_argument("--acc_grad", type=int, default=0)
    ap.add_argument("--missing", type=int, default=0)
    ap.add_argument("--random_dropout", type=bool, default=False)
    ap.add_argument("--save_eval_images", type=bool, default=False)

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
        classifier_free_prob=args.classifier_free_prob,
        classifier_free_w=args.classifier_free_w,
        noise_weight=args.noise_weight,
        data_augmentation=args.data_augmentation,
        checkpoint_path=args.checkpoint_path,
        rotation=args.rotation,
        only_rotation=args.only_rotation,
        angle_type=args.angle_type,
        predict_xstart=args.predict_xstart,
        discrete=args.discrete,
        loss_type=args.loss_type,
        cold_diffusion=args.cold_diffusion,
        evaluate=args.evaluate,
        freeze_backbone=args.freeze_backbone,
        visual_pretrained=args.visual_pretrained,
        backbone=args.backbone,
        n_layers=args.n_layers,
        architecture=args.architecture,
        degree=args.degree,
        virt_nodes=args.virt_nodes,
        max_epochs=args.max_epochs,
        unique_graph=args.unique_graph,
        inf_fully=args.inf_fully,
        all_equivariant=args.all_equivariant,
        wandb_id=args.wandb_id,
        padding=args.padding,
        random_dropout=args.random_dropout,
        acc_grad=args.acc_grad,
        save_eval_images=args.save_eval_images,
        missing=args.missing
    )
