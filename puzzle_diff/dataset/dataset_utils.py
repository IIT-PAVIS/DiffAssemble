import numpy as np
import torch
from torchvision.datasets import CIFAR100, ImageNet

from .breakingbad_dt import GeometryPartDataset
from .celeba_dt import CelebA_HQ
from .nips_dt import Nips_dt
from .objects_dataset import Objects_Dataset
from .puzzle_dataset import (
    Puzzle_Dataset,
    Puzzle_Dataset_MP,
    Puzzle_Dataset_Pad,
    Puzzle_Dataset_ROT,
    Puzzle_Dataset_ROT_MP,
    generate_random_expander,
)
from .roc_dt import Roc_dt
from .sind_dt import Sind_dt
from .sind_vist_dt import Sind_Vist_dt
from .text_dataset import Text_dataset
from .vist_dataset import Vist_dataset
from .wiki_dt import Wiki_dt
from .wikiart_dt import Wikiart_DT

ALLOWED_DT = ["celeba", "cifar100", "wikiart", "imagenet"]
ALLOWED_TEXT = ["nips", "sind", "roc", "wiki"]


def get_dataset(
    dataset: str,
    puzzle_sizes: list,
    augment=False,
    degree=-1,
    unique_graph=False,
    inf_fully=True,
) -> tuple:
    """
    Get dataset of images based on specified dataset name and puzzle sizes.

    Parameters:
    - dataset (str): The name of the dataset to be used (e.g., "celeba", "cifar100", "wikiart").
    - puzzle_sizes (list): A list of puzzle sizes to be used.

    - degree (int): Degree of the graph. If -1 use FC graph
    - unique_graph (boolean): Defines the strategy used to create the topology of the graph. If it is false, each sample is associated with a random graph topology.
    - inf_fully (boolean): degree for the test set (inference). If True is -1, this, we use FC graph.

    Returns:
    - Tuple of three elements:
        - puzzleDt_train (Puzzle_Dataset): The training dataset.
        - puzzleDt_test (Puzzle_Dataset): The testing dataset.
        - real_puzzle_sizes (list): List of tuples containing the actual puzzle sizes.
    """

    # Check if the dataset is supported
    assert (
        dataset in ALLOWED_DT
    ), f"dataset {dataset} not supported, need to be one of {ALLOWED_DT}"

    # Create a list of tuples containing the actual puzzle sizes
    real_puzzle_sizes = [(x, x) for x in puzzle_sizes]

    # Define a lambda function to get the first element of a tuple
    get_fn = lambda x: x[0]

    # Load the specified dataset
    if dataset == "celeba":
        train_dt = CelebA_HQ(train=True)
        test_dt = CelebA_HQ(train=False)
    elif dataset == "cifar100":
        train_dt = CIFAR100(root="./datasets", download=True, train=True)
        test_dt = CIFAR100(root="./datasets", download=True, train=False)
    elif dataset == "wikiart":
        train_dt = Wikiart_DT(train=True)
        test_dt = Wikiart_DT(train=False)
    elif dataset == "imagenet":
        train_dt = ImageNet("./datasets/imagenet2012", split="train")
        test_dt = ImageNet("./datasets/imagenet2012", split="val")
    else:
        raise Exception("Not supported")

    # set a seed in case we want a topology  depending only on the size of the graph
    rng = np.random.randint(1, 123456) if unique_graph else None

    # Create puzzle datasets using the loaded datasets
    puzzleDt_train = Puzzle_Dataset(
        dataset=train_dt,
        dataset_get_fn=get_fn,
        patch_per_dim=real_puzzle_sizes,
        augment=augment,
        degree=degree,
        unique_graph=rng,
    )
    puzzleDt_test = Puzzle_Dataset(
        dataset=test_dt,
        dataset_get_fn=get_fn,
        patch_per_dim=real_puzzle_sizes,
        augment=False,
        degree=-1 if inf_fully else degree,  # fully connected graph in inference
        unique_graph=rng,
        random=False,  # creating random puzzles
    )

    return (puzzleDt_train, puzzleDt_test, real_puzzle_sizes)


def get_dataset_missing_pieces(
    dataset: str, puzzle_sizes: list, missing_pieces_perc: int, augment: bool = False
) -> tuple:
    """
    Get dataset of images based on specified dataset name and puzzle sizes.

    Parameters:
    - dataset (str): The name of the dataset to be used (e.g., "celeba", "cifar100", "wikiart").
    - puzzle_sizes (list): A list of puzzle sizes to be used.

    Returns:
    - Tuple of three elements:
        - puzzleDt_train (Puzzle_Dataset): The training dataset.
        - puzzleDt_test (Puzzle_Dataset): The testing dataset.
        - real_puzzle_sizes (list): List of tuples containing the actual puzzle sizes.
    """

    # Check if the dataset is supported
    assert (
        dataset in ALLOWED_DT
    ), f"dataset {dataset} not supported, need to be one of {ALLOWED_DT}"

    # Create a list of tuples containing the actual puzzle sizes
    real_puzzle_sizes = [(x, x) for x in puzzle_sizes]

    # Define a lambda function to get the first element of a tuple
    get_fn = lambda x: x[0]

    # Load the specified dataset
    if dataset == "celeba":
        train_dt = CelebA_HQ(train=True)
        test_dt = CelebA_HQ(train=False)
    elif dataset == "cifar100":
        train_dt = CIFAR100(root="./datasets", download=True, train=True)
        test_dt = CIFAR100(root="./datasets", download=True, train=False)
    elif dataset == "wikiart":
        train_dt = Wikiart_DT(train=True)
        test_dt = Wikiart_DT(train=False)

    # Create puzzle datasets using the loaded datasets
    puzzleDt_train = Puzzle_Dataset_MP(
        dataset=train_dt,
        dataset_get_fn=get_fn,
        patch_per_dim=real_puzzle_sizes,
        missing_perc=missing_pieces_perc,
        augment=augment,
    )
    puzzleDt_test = Puzzle_Dataset_MP(
        dataset=test_dt,
        dataset_get_fn=get_fn,
        patch_per_dim=real_puzzle_sizes,
        missing_perc=missing_pieces_perc,
        augment=False,
    )

    return (puzzleDt_train, puzzleDt_test, real_puzzle_sizes)


def get_dataset_ROT(
    dataset: str,
    puzzle_sizes: list,
    augment=False,
    degree=-1,
    unique_graph=False,
    inf_fully=True,
    all_equivariant=True,
    random_dropout=False,
    missing=0,
) -> tuple:
    """
    Get dataset of images based on specified dataset name and puzzle sizes.

    Parameters:
    - dataset (str): The name of the dataset to be used (e.g., "celeba", "cifar100", "wikiart").
    - puzzle_sizes (list): A list of puzzle sizes to be used.
    - degree (int): Degree of the graph. If -1 use FC graph
    - unique_graph (boolean): Defines the strategy used to create the topology of the graph. If it is false, each sample is associated with a random graph topology.
    - inf_fully (boolean): degree for the test set (inference). If True is -1, this, we use FC graph.

    Returns:
    - Tuple of three elements:
        - puzzleDt_train (Puzzle_Dataset): The training dataset.
        - puzzleDt_test (Puzzle_Dataset): The testing dataset.
        - real_puzzle_sizes (list): List of tuples containing the actual puzzle sizes.
    """

    # Check if the dataset is supported
    assert (
        dataset in ALLOWED_DT
    ), f"dataset {dataset} not supported, need to be one of {ALLOWED_DT}"

    # Create a list of tuples containing the actual puzzle sizes
    real_puzzle_sizes = [(x, x) for x in puzzle_sizes]

    # Define a lambda function to get the first element of a tuple
    get_fn = lambda x: x[0]

    # Load the specified dataset
    if dataset == "celeba":
        train_dt = CelebA_HQ(train=True)
        test_dt = CelebA_HQ(train=False)
    elif dataset == "cifar100":
        train_dt = CIFAR100(root="./datasets", download=True, train=True)
        test_dt = CIFAR100(root="./datasets", download=True, train=False)
    elif dataset == "wikiart":
        train_dt = Wikiart_DT(train=True)
        test_dt = Wikiart_DT(train=False)
    elif dataset == "imagenet":
        train_dt = ImageNet("./datasets/imagenet2012", split="train")
        test_dt = ImageNet("./datasets/imagenet2012", split="val")

    else:
        raise Exception("Dataset not supported")
    # set a seed in case we want a topology depending only on the size of the graph
    rng = np.random.randint(1, 123456) if unique_graph else None

    # Create puzzle datasets using the loaded datasets
    puzzleDt_train = Puzzle_Dataset_ROT(
        dataset=train_dt,
        dataset_get_fn=get_fn,
        patch_per_dim=real_puzzle_sizes,
        augment=augment,
        degree=degree,
        unique_graph=rng,
        all_equivariant=all_equivariant,
        random_dropout=random_dropout,
    )

    if missing > 0:
        puzzleDt_train = Puzzle_Dataset_ROT_MP(
                dataset=train_dt,
                dataset_get_fn=get_fn,
                patch_per_dim=real_puzzle_sizes,
                augment=augment,
                missing_perc=missing
            )
        
        puzzleDt_test = Puzzle_Dataset_ROT_MP(
            dataset=test_dt,
            dataset_get_fn=get_fn,
            patch_per_dim=real_puzzle_sizes,
            augment=False,
            missing_perc=missing
        )
    else:
        puzzleDt_test = Puzzle_Dataset_ROT(
            dataset=test_dt,
            dataset_get_fn=get_fn,
            patch_per_dim=real_puzzle_sizes,
            augment=False,
        )

    return (puzzleDt_train, puzzleDt_test, real_puzzle_sizes)


def get_dataset_padding(
    dataset: str,
    puzzle_sizes: list,
    augment=False,
    degree=-1,
    padding=1,
    inf_fully=True,
) -> tuple:
    """
    Get dataset of images based on specified dataset name and puzzle sizes.

    Parameters:
    - dataset (str): The name of the dataset to be used (e.g., "celeba", "cifar100", "wikiart").
    - puzzle_sizes (list): A list of puzzle sizes to be used.

    - degree (int): Degree of the graph. If -1 use FC graph
    - unique_graph (boolean): Defines the strategy used to create the topology of the graph. If it is false, each sample is associated with a random graph topology.
    - inf_fully (boolean): degree for the test set (inference). If True is -1, this, we use FC graph.
    Returns:
    - Tuple of three elements:
        - puzzleDt_train (Puzzle_Dataset): The training dataset.
        - puzzleDt_test (Puzzle_Dataset): The testing dataset.
        - real_puzzle_sizes (list): List of tuples containing the actual puzzle sizes.
    """

    # Check if the dataset is supported
    assert (
        dataset in ALLOWED_DT
    ), f"dataset {dataset} not supported, need to be one of {ALLOWED_DT}"

    # Create a list of tuples containing the actual puzzle sizes
    real_puzzle_sizes = [(x, x) for x in puzzle_sizes]

    # Define a lambda function to get the first element of a tuple
    get_fn = lambda x: x[0]

    # Load the specified dataset
    if dataset == "celeba":
        train_dt = CelebA_HQ(train=True)
        test_dt = CelebA_HQ(train=False)
    elif dataset == "cifar100":
        train_dt = CIFAR100(root="./datasets", download=True, train=True)
        test_dt = CIFAR100(root="./datasets", download=True, train=False)
    elif dataset == "wikiart":
        train_dt = Wikiart_DT(train=True)
        test_dt = Wikiart_DT(train=False)

    # Create puzzle datasets using the loaded datasets
    puzzleDt_train = Puzzle_Dataset_Pad(
        dataset=train_dt,
        dataset_get_fn=get_fn,
        patch_per_dim=real_puzzle_sizes,
        augment=augment,
        degree=degree,
        unique_graph=None,
    )
    puzzleDt_test = Puzzle_Dataset_Pad(
        dataset=test_dt,
        dataset_get_fn=get_fn,
        patch_per_dim=real_puzzle_sizes,
        augment=False,
        degree=-1 if inf_fully else degree,  # fully connected graph in inference
        unique_graph=None,
        padding=padding,
    )

    return (puzzleDt_train, puzzleDt_test, real_puzzle_sizes)


def get_dataset_old(
    dataset: str, puzzle_sizes: list
) -> tuple[Puzzle_Dataset, Puzzle_Dataset, list]:
    assert (
        dataset in ALLOWED_DT
    ), f"dataset {dataset} not supported, need to be one of {ALLOWED_DT}"

    real_puzzle_sizes = [(x, x) for x in puzzle_sizes]

    get_fn = lambda x: x[0]

    if dataset == "celeba":
        train_dt = CelebA(
            root="./datasets",
            download=True,
            split="train",
        )

        test_dt = CelebA(
            root="./datasets",
            download=True,
            split="test",
        )

    if dataset == "cifar100":
        train_dt = CIFAR100(
            root="./datasets",
            download=True,
            train=True,
        )

        test_dt = CIFAR100(root="./datasets", download=True, train=False)

    if dataset == "wikiart":
        train_dt = Wikiart_DT(train=True)
        test_dt = Wikiart_DT(train=False)

    puzzleDt_train = Puzzle_Dataset(
        dataset=train_dt,
        dataset_get_fn=get_fn,
        patch_per_dim=real_puzzle_sizes,
    )

    puzzleDt_test = Puzzle_Dataset(
        dataset=test_dt,
        dataset_get_fn=get_fn,
        patch_per_dim=real_puzzle_sizes,
    )

    return (puzzleDt_train, puzzleDt_test, real_puzzle_sizes)


def get_dataset_text(dataset: str, cv_split):
    assert dataset in ALLOWED_TEXT

    if dataset == "nips":
        train_dt = Nips_dt(split="train")
        val_dt = Nips_dt(split="val")
        test_dt = Nips_dt(split="test")
    elif dataset == "sind":
        train_dt = Sind_dt(split="train")
        val_dt = Sind_dt(split="val")
        test_dt = Sind_dt(split="test")
    elif dataset == "roc":
        train_dt = Roc_dt(split="train")
        val_dt = Roc_dt(split="test")
        test_dt = Roc_dt(split="test")
    elif dataset == "wiki":
        train_dt = Wiki_dt(split="train", split_idx=cv_split)
        val_dt = Wiki_dt(split="test", split_idx=cv_split)
        test_dt = Wiki_dt(split="test", split_idx=cv_split)
    else:
        raise Exception(f"Dataset {dataset} is not provided.")

    train_dt = Text_dataset(train_dt)
    val_dt = Text_dataset(val_dt)
    test_dt = Text_dataset(test_dt)

    return train_dt, val_dt, test_dt


def get_dataset_vist(dataset: str):
    if dataset == "sind":
        train_dt = Sind_Vist_dt(split="train")
        test_dt = Sind_Vist_dt(split="test")
    else:
        raise Exception(f"Dataset {dataset} is not provided.")

    train_dt = Vist_dataset(train_dt)
    test_dt = Vist_dataset(test_dt)

    return train_dt, None, test_dt


def get_dataset_3d(
    dataset: str, category: str, max_num_part: int, min_num_part: int, missing: int
):
    train_dict = dict(
        data_dir="datasets/breaking-bad",  # "/media/hd1/gscarpellini/breaking-bad",
        data_fn="data_split/everyday.train.txt",
        data_keys=("part_ids",),
        category=category,  # "",  # all
        num_points=1000,
        min_num_part=min_num_part,
        max_num_part=max_num_part,
        shuffle_parts=False,
        rot_range=-1,
        overfit=-1,
    )
    test_dict = dict(
        data_dir="datasets/breaking-bad",  # "/media/hd1/gscarpellini/breaking-bad",
        data_fn="data_split/everyday.val.txt",
        data_keys=("part_ids",),
        category=category,
        num_points=1000,
        min_num_part=min_num_part,
        max_num_part=max_num_part,
        shuffle_parts=False,
        rot_range=-1,
        overfit=-1,
    )

    if dataset == "breaking-bad":
        train_dt = GeometryPartDataset(**train_dict)
        test_dt = GeometryPartDataset(**test_dict)
    else:
        raise Exception(f"Dataset {dataset} is not provided.")

    train_dt = Objects_Dataset(train_dt, lambda x: x)
    test_dt = Objects_Dataset(test_dt, lambda x: x, missing)

    return train_dt, None, test_dt
