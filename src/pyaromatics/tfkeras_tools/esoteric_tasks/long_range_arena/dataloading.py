import os
import torch
from pathlib import Path
from typing import Callable, Optional, TypeVar, Dict, Tuple, Union
from .dataloaders.lra import IMDB, ListOps, PathFinder, AAN
from .dataloaders.basic import CIFAR10, MNIST
from .dataloaders.copy import CopyTask

# DEFAULT_CACHE_DIR_ROOT = Path("./cache_dir/")
FILENAME = os.path.realpath(__file__)
CDIR = os.path.dirname(FILENAME)
DEFAULT_CACHE_DIR_ROOT = os.path.join(CDIR, '..', '..', '..', '..', 'data', "lra_cache_dir")
os.makedirs(DEFAULT_CACHE_DIR_ROOT, exist_ok=True)

DataLoader = TypeVar("DataLoader")
InputType = [str, Optional[int], Optional[int]]
ReturnType = Tuple[DataLoader, DataLoader, DataLoader, Dict, int, int, int, int]

# Custom loading functions must therefore have the template.
dataset_fn = Callable[[str, Optional[int], Optional[int]], ReturnType]


# Example interface for making a loader.
def custom_loader(cache_dir: str, batch_size: int = 50, seed: int = 42) -> ReturnType:
    ...


def make_data_loader(
        dset,
        dobj,
        seed: int,
        batch_size: int = 128,
        shuffle: bool = True,
        drop_last: bool = True,
        collate_fn: callable = None,
):
    """

    :param dset: 			(PT dset):		PyTorch dataset object.
    :param dobj (=None): 	(AG data): 		Dataset object, as returned by A.G.s dataloader.
    :param seed: 			(int):			Int for seeding shuffle.
    :param batch_size: 		(int):			Batch size for batches.
    :param shuffle:         (bool):			Shuffle the data loader?
    :param drop_last: 		(bool):			Drop ragged final batch (particularly for training).
    :return:
    """

    # Create a generator for seeding random number draws.
    if seed is not None:
        rng = torch.Generator()
        rng.manual_seed(seed)
    else:
        rng = None

    if dobj is not None:
        assert collate_fn is None
        collate_fn = dobj._collate_fn

    # Generate the dataloaders.
    return torch.utils.data.DataLoader(
        dataset=dset,
        collate_fn=collate_fn,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        generator=rng,
    )


def create_lra_imdb_classification_dataset(
        cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT, batch_size: int = 50, seed: int = 42
) -> ReturnType:
    print("[*] Generating LRA-text (IMDB) Classification Dataset")
    name = "imdb"
    dataset_obj = IMDB("imdb")
    dataset_obj.cache_dir = Path(cache_dir) / name
    dataset_obj.setup()

    trainloader = make_data_loader(
        dataset_obj.dataset_train, dataset_obj, seed=seed, batch_size=batch_size
    )
    testloader = make_data_loader(
        dataset_obj.dataset_test,
        dataset_obj,
        seed=seed,
        batch_size=batch_size,
        drop_last=False,
        shuffle=False,
    )
    valloader = None

    N_CLASSES = dataset_obj.d_output
    SEQ_LENGTH = dataset_obj.l_max
    IN_DIM = 135  # We should probably stop this from being hard-coded.
    TRAIN_SIZE = len(dataset_obj.dataset_train)

    aux_loaders = {}

    return (
        trainloader,
        valloader,
        testloader,
        aux_loaders,
        N_CLASSES,
        SEQ_LENGTH,
        IN_DIM,
        TRAIN_SIZE,
    )


def create_lra_listops_classification_dataset(
        cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT, batch_size: int = 50, seed: int = 42
) -> ReturnType:
    print("[*] Generating LRA-listops Classification Dataset")

    name = "listops"
    dir_name = "./raw_datasets/lra_release/lra_release/listops-1000"

    dataset_obj = ListOps(name, data_dir=dir_name)
    dataset_obj.cache_dir = Path(cache_dir) / name
    dataset_obj.setup()

    trn_loader = make_data_loader(
        dataset_obj.dataset_train, dataset_obj, seed=seed, batch_size=batch_size
    )
    val_loader = make_data_loader(
        dataset_obj.dataset_val,
        dataset_obj,
        seed=seed,
        batch_size=batch_size,
        drop_last=False,
        shuffle=False,
    )
    tst_loader = make_data_loader(
        dataset_obj.dataset_test,
        dataset_obj,
        seed=seed,
        batch_size=batch_size,
        drop_last=False,
        shuffle=False,
    )

    N_CLASSES = dataset_obj.d_output
    SEQ_LENGTH = dataset_obj.l_max
    IN_DIM = 20
    TRAIN_SIZE = len(dataset_obj.dataset_train)

    aux_loaders = {}

    return (
        trn_loader,
        val_loader,
        tst_loader,
        aux_loaders,
        N_CLASSES,
        SEQ_LENGTH,
        IN_DIM,
        TRAIN_SIZE,
    )


def create_lra_path32_classification_dataset(
        cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT, batch_size: int = 50, seed: int = 42
) -> ReturnType:
    print("[*] Generating LRA-Pathfinder32 Classification Dataset")
    name = "pathfinder"
    resolution = 32
    dir_name = f"./raw_datasets/lra_release/lra_release/pathfinder{resolution}"

    dataset_obj = PathFinder(name, data_dir=dir_name, resolution=resolution)
    dataset_obj.cache_dir = Path(cache_dir) / name
    dataset_obj.setup()

    trn_loader = make_data_loader(
        dataset_obj.dataset_train, dataset_obj, seed=seed, batch_size=batch_size
    )
    val_loader = make_data_loader(
        dataset_obj.dataset_val,
        dataset_obj,
        seed=seed,
        batch_size=batch_size,
        drop_last=False,
        shuffle=False,
    )
    tst_loader = make_data_loader(
        dataset_obj.dataset_test,
        dataset_obj,
        seed=seed,
        batch_size=batch_size,
        drop_last=False,
        shuffle=False,
    )

    N_CLASSES = dataset_obj.d_output
    SEQ_LENGTH = dataset_obj.dataset_train.tensors[0].shape[1]
    IN_DIM = dataset_obj.d_input
    TRAIN_SIZE = dataset_obj.dataset_train.tensors[0].shape[0]

    aux_loaders = {}

    return (
        trn_loader,
        val_loader,
        tst_loader,
        aux_loaders,
        N_CLASSES,
        SEQ_LENGTH,
        IN_DIM,
        TRAIN_SIZE,
    )


def create_lra_pathx_classification_dataset(
        cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT, batch_size: int = 50, seed: int = 42
) -> ReturnType:
    print("[*] Generating LRA-PathX Classification Dataset")

    name = "pathfinder"
    resolution = 128
    dir_name = f"./raw_datasets/lra_release/lra_release/pathfinder{resolution}"

    dataset_obj = PathFinder(name, data_dir=dir_name, resolution=resolution)
    dataset_obj.cache_dir = Path(cache_dir) / name
    dataset_obj.setup()

    trn_loader = make_data_loader(
        dataset_obj.dataset_train, dataset_obj, seed=seed, batch_size=batch_size
    )
    val_loader = make_data_loader(
        dataset_obj.dataset_val,
        dataset_obj,
        seed=seed,
        batch_size=batch_size,
        drop_last=False,
        shuffle=False,
    )
    tst_loader = make_data_loader(
        dataset_obj.dataset_test,
        dataset_obj,
        seed=seed,
        batch_size=batch_size,
        drop_last=False,
        shuffle=False,
    )

    N_CLASSES = dataset_obj.d_output
    SEQ_LENGTH = dataset_obj.dataset_train.tensors[0].shape[1]
    IN_DIM = dataset_obj.d_input
    TRAIN_SIZE = dataset_obj.dataset_train.tensors[0].shape[0]

    aux_loaders = {}

    return (
        trn_loader,
        val_loader,
        tst_loader,
        aux_loaders,
        N_CLASSES,
        SEQ_LENGTH,
        IN_DIM,
        TRAIN_SIZE,
    )


def create_lra_image_classification_dataset(
        cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT, seed: int = 42, batch_size: int = 128
) -> ReturnType:
    """
    Cifar is quick to download and is automatically cached.
    """

    print("[*] Generating LRA-listops Classification Dataset")
    name = "cifar"
    kwargs = {
        "grayscale": True,  # LRA uses a grayscale CIFAR image.
    }

    dataset_obj = CIFAR10(
        name, data_dir=cache_dir, **kwargs
    )  # TODO - double check what the dir here does.
    dataset_obj.setup()

    trn_loader = make_data_loader(
        dataset_obj.dataset_train, dataset_obj, seed=seed, batch_size=batch_size
    )
    val_loader = make_data_loader(
        dataset_obj.dataset_val,
        dataset_obj,
        seed=seed,
        batch_size=batch_size,
        drop_last=False,
        shuffle=False,
    )
    tst_loader = make_data_loader(
        dataset_obj.dataset_test,
        dataset_obj,
        seed=seed,
        batch_size=batch_size,
        drop_last=False,
        shuffle=False,
    )

    N_CLASSES = dataset_obj.d_output
    SEQ_LENGTH = 32 * 32
    IN_DIM = 1
    TRAIN_SIZE = len(dataset_obj.dataset_train)

    aux_loaders = {}

    return (
        trn_loader,
        val_loader,
        tst_loader,
        aux_loaders,
        N_CLASSES,
        SEQ_LENGTH,
        IN_DIM,
        TRAIN_SIZE,
    )


def create_lra_aan_classification_dataset(
        cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT,
        batch_size: int = 50,
        seed: int = 42,
) -> ReturnType:
    print("[*] Generating LRA-AAN Classification Dataset")

    name = "aan"
    dir_name = "./raw_datasets/lra_release/lra_release/tsv_data"
    kwargs = {
        "n_workers": 1,  # Multiple workers seems to break AAN.
    }

    dataset_obj = AAN(name, data_dir=dir_name, **kwargs)
    dataset_obj.cache_dir = Path(cache_dir) / name
    dataset_obj.setup()

    trn_loader = make_data_loader(
        dataset_obj.dataset_train, dataset_obj, seed=seed, batch_size=batch_size
    )
    val_loader = make_data_loader(
        dataset_obj.dataset_val,
        dataset_obj,
        seed=seed,
        batch_size=batch_size,
        drop_last=False,
        shuffle=False,
    )
    tst_loader = make_data_loader(
        dataset_obj.dataset_test,
        dataset_obj,
        seed=seed,
        batch_size=batch_size,
        drop_last=False,
        shuffle=False,
    )

    N_CLASSES = dataset_obj.d_output
    SEQ_LENGTH = dataset_obj.l_max
    IN_DIM = len(dataset_obj.vocab)
    TRAIN_SIZE = len(dataset_obj.dataset_train)

    aux_loaders = {}

    return (
        trn_loader,
        val_loader,
        tst_loader,
        aux_loaders,
        N_CLASSES,
        SEQ_LENGTH,
        IN_DIM,
        TRAIN_SIZE,
    )


def create_cifar_classification_dataset(
        cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT, seed: int = 42, batch_size: int = 128
) -> ReturnType:
    """
    Cifar is quick to download and is automatically cached.
    """

    print("[*] Generating CIFAR (color) Classification Dataset")
    name = "cifar"
    kwargs = {
        "grayscale": False,  # LRA uses a grayscale CIFAR image.
    }

    dataset_obj = CIFAR10(name, data_dir=cache_dir, **kwargs)
    dataset_obj.setup()

    trn_loader = make_data_loader(
        dataset_obj.dataset_train, dataset_obj, seed=seed, batch_size=batch_size
    )
    val_loader = make_data_loader(
        dataset_obj.dataset_val,
        dataset_obj,
        seed=seed,
        batch_size=batch_size,
        drop_last=False,
        shuffle=False,
    )
    tst_loader = make_data_loader(
        dataset_obj.dataset_test,
        dataset_obj,
        seed=seed,
        batch_size=batch_size,
        drop_last=False,
        shuffle=False,
    )

    N_CLASSES = dataset_obj.d_output
    SEQ_LENGTH = 32 * 32
    IN_DIM = 3
    TRAIN_SIZE = len(dataset_obj.dataset_train)

    aux_loaders = {}

    return (
        trn_loader,
        val_loader,
        tst_loader,
        aux_loaders,
        N_CLASSES,
        SEQ_LENGTH,
        IN_DIM,
        TRAIN_SIZE,
    )


def create_mnist_classification_dataset(
        cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT, seed: int = 42, batch_size: int = 128
) -> ReturnType:
    print("[*] Generating MNIST Classification Dataset")

    name = "mnist"

    kwargs = {"permute": False}

    dataset_obj = MNIST(name, data_dir=cache_dir, **kwargs)
    dataset_obj.setup()

    trn_loader = make_data_loader(
        dataset_obj.dataset_train, dataset_obj, seed=seed, batch_size=batch_size
    )
    val_loader = make_data_loader(
        dataset_obj.dataset_val,
        dataset_obj,
        seed=seed,
        batch_size=batch_size,
        drop_last=False,
        shuffle=False,
    )
    tst_loader = make_data_loader(
        dataset_obj.dataset_test,
        dataset_obj,
        seed=seed,
        batch_size=batch_size,
        drop_last=False,
        shuffle=False,
    )

    N_CLASSES = dataset_obj.d_output
    SEQ_LENGTH = 28 * 28
    IN_DIM = 1
    TRAIN_SIZE = len(dataset_obj.dataset_train)
    aux_loaders = {}
    return (
        trn_loader,
        val_loader,
        tst_loader,
        aux_loaders,
        N_CLASSES,
        SEQ_LENGTH,
        IN_DIM,
        TRAIN_SIZE,
    )


def create_pmnist_classification_dataset(
        cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT, seed: int = 42, batch_size: int = 128
) -> ReturnType:
    print("[*] Generating permuted-MNIST Classification Dataset")

    name = "mnist"
    kwargs = {"permute": True}

    dataset_obj = MNIST(name, data_dir=cache_dir, **kwargs)
    dataset_obj.setup()

    trn_loader = make_data_loader(
        dataset_obj.dataset_train, dataset_obj, seed=seed, batch_size=batch_size
    )
    val_loader = make_data_loader(
        dataset_obj.dataset_val,
        dataset_obj,
        seed=seed,
        batch_size=batch_size,
        drop_last=False,
        shuffle=False,
    )
    tst_loader = make_data_loader(
        dataset_obj.dataset_test,
        dataset_obj,
        seed=seed,
        batch_size=batch_size,
        drop_last=False,
        shuffle=False,
    )

    N_CLASSES = dataset_obj.d_output
    SEQ_LENGTH = 28 * 28
    IN_DIM = 1
    TRAIN_SIZE = len(dataset_obj.dataset_train)
    aux_loaders = {}
    return (
        trn_loader,
        val_loader,
        tst_loader,
        aux_loaders,
        N_CLASSES,
        SEQ_LENGTH,
        IN_DIM,
        TRAIN_SIZE,
    )


def create_copy_classification_dataset(
        cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT, seed: int = 42, batch_size: int = 128
) -> ReturnType:
    print("[*] Generating copy task")
    name = "copy"

    PATTERN_LENGTH = 20
    PADDING_LENGTH = 5
    TRAIN_SAMPLES = 20000
    IN_DIM = 8

    kwargs = {
        "train_size": TRAIN_SAMPLES,
        "val_size": 1000,
        "test_size": 1000,
        "in_dim": IN_DIM,
        "pattern_length": PATTERN_LENGTH,
        "padding_length": PADDING_LENGTH,
    }

    dataset_obj = CopyTask(name, **kwargs)
    dataset_obj.setup()

    trn_loader = make_data_loader(
        dataset_obj.dataset_train, dataset_obj, seed=seed, batch_size=batch_size
    )
    val_loader = make_data_loader(
        dataset_obj.dataset_val,
        dataset_obj,
        seed=seed,
        batch_size=batch_size,
        drop_last=False,
        shuffle=False,
    )
    tst_loader = make_data_loader(
        dataset_obj.dataset_test,
        dataset_obj,
        seed=seed,
        batch_size=batch_size,
        drop_last=False,
        shuffle=False,
    )

    OUT_DIM = dataset_obj.out_dim
    SEQ_LENGTH = dataset_obj.seq_length_in
    TRAIN_SIZE = kwargs["train_size"]
    aux_loaders = {}
    return (
        trn_loader,
        val_loader,
        tst_loader,
        aux_loaders,
        OUT_DIM,
        SEQ_LENGTH,
        IN_DIM,
        TRAIN_SIZE,
    )


Datasets = {
    # Other loaders.
    "mnist-classification": create_mnist_classification_dataset,
    "pmnist-classification": create_pmnist_classification_dataset,
    "cifar-classification": create_cifar_classification_dataset,
    # LRA.
    "imdb-classification": create_lra_imdb_classification_dataset,
    "listops-classification": create_lra_listops_classification_dataset,
    "aan-classification": create_lra_aan_classification_dataset,
    "lra-cifar-classification": create_lra_image_classification_dataset,
    "pathfinder-classification": create_lra_path32_classification_dataset,
    "pathx-classification": create_lra_pathx_classification_dataset,
    # Synthetic memory tasks.
    "copy-classification": create_copy_classification_dataset,
}

if __name__ == '__main__':
    create_dataset_fn = Datasets['listops-classification']
    (
        trainloader,
        valloader,
        testloader,
        aux_dataloaders,
        n_classes,
        seq_len,
        in_dim,
        train_size,
    ) = create_dataset_fn(args.dir_name, seed=None, batch_size=16)
