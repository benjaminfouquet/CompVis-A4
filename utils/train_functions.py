import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from tqdm.auto import trange
from types import FunctionType
from typing import Dict, Iterable, List, Tuple

from classes.asymmetricRecall import AsymmetricRecall
from classes.embDataset import EmbDataset


def training_step(
    model: nn.Module,
    loss_func: FunctionType,
    optimizer: optim.Optimizer,
    train_sets: Iterable[Dict],
    temperature: int,
    topk_list: List[int],
    device,
):
    """
    Training the model for one epoch.
    Generally it will take two dicts:
        - the first one is the extracted embds of the original dataset
        - the second one is augmented version of these embds
    """
    model.train()

    total_loss = 0

    for training_set in train_sets:
        left, right = training_set["left"], training_set["right"]

        out_left, out_right = model(left), model(right)

        loss = loss_func(out_left, out_right, temperature, device)

        total_loss += loss

    total_loss /= len(train_sets)

    # debate: include this in the loop above or not ?
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    metrics = {"Train/Loss": total_loss.item()}

    aR = AsymmetricRecall(
        out_left.clone().detach().cpu(), out_right.clone().detach().cpu()
    )

    for k in topk_list:
        metrics[f"Train/top {k}"] = aR.eval(at=k)
    return metrics

def validate(
    model: nn.Module,
    loss_func: FunctionType,
    valid_set: Dict,
    temperature: int,
    topk_list: List[int],
    device,
):
    """
    Validating the model.
    """
    model.eval()

    with torch.no_grad():
        left, right = valid_set["left"], valid_set["right"]
        out_left, out_right = model(left), model(right)

        test_loss = loss_func(out_left, out_right, temperature, device)

        metrics = {"Test/Loss": test_loss.item()}

        aR = AsymmetricRecall(out_left.detach().cpu(), out_right.detach().cpu())

        for k in topk_list:
            metrics[f"Test/top {k}"] = aR.eval(at=k)

        return metrics


def train(
    model: nn.Module,
    optimizer,
    loss_func: FunctionType,
    train_set: Iterable[Dict],
    valid_set: Dict,
    num_epochs: int,
    save_to: str,
    temperature: int,
    topk_list: Iterable[int],
    device,
) -> Dict:
    """
    This will train the model with the given optimizer for {num_epochs} epochs.
    And it will save the metrics defined in training_step and validate in the given path {save_to}.
    """
    for k in topk_list:
        assert k > 0

    writer = SummaryWriter(save_to)
    metrics_per_run = {"train": dict(), "test": dict()}

    for epoch in trange(1, num_epochs + 1):
        if epoch%10 == 0:
            print('Epoch ', epoch, '/', num_epochs, end='\r')
        info = training_step(
            model, loss_func, optimizer, train_set, temperature, topk_list, device
        )
        for metric, val in info.items():
            if not (metric in metrics_per_run["train"]):
                metrics_per_run["train"][metric] = []
            metrics_per_run["train"][metric].append(val)

            writer.add_scalar(metric, val, epoch)
            writer.flush()

        info = validate(model, loss_func, valid_set, temperature, topk_list, device)
        for metric, val in info.items():
            if not (metric in metrics_per_run["test"]):
                metrics_per_run["test"][metric] = []
            metrics_per_run["test"][metric].append(val)

            writer.add_scalar(metric, val, epoch)
            writer.flush()

    np.save(save_to / "data", metrics_per_run)

    return metrics_per_run


def split_train_test(dataset: EmbDataset, test_split: float, device) -> Tuple[Dict]:
    """
    Split the given dataset into train and test sets using test_split percentage.
    If augmented_dataset is True, it will assume that it is a concated dataset where the original
    datapoints are stocked first then followed by their augmented versions.
    """
    assert isinstance(dataset, EmbDataset)
    # Creating data indices for training and validation splits:
    augmented_dataset = dataset.augmented
    dataset_size = len(dataset) if not augmented_dataset else len(dataset) // 2
    indices = list(range(dataset_size))
    split = int(np.floor(test_split * dataset_size))

    np.random.shuffle(indices)

    train_indices_normal, test_indices = indices[split:], indices[:split]
    embds = dataset.load_all_to_device(device)
    left, right = embds["left"].to(device), embds["right"].to(device)

    splitted = {
        "train": [
            {
                "left": left[train_indices_normal],
                "right": right[train_indices_normal],
            }
        ],
        "test": {
            "left": left[test_indices],
            "right": right[test_indices],
        },
    }

    if augmented_dataset:
        train_indices_aug = [i + len(dataset) // 2 for i in train_indices_normal]
        splitted["train"].append(
            {
                "left": left[train_indices_aug],
                "right": right[train_indices_aug],
            }
        )

    return splitted