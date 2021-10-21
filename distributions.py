import os
from collections import defaultdict
from typing import List
import matplotlib.pyplot as plt
import hub
import numpy as np
from tqdm import tqdm

import torch
from torch.nn.functional import l1_loss as mae


target_frequency = 10

EPOCHS = 1
WORKERS = 8
DATASET_URI = "./distributions_evaluation_dataset"
num_classes = 100
num_samples_per_class = 1000
BATCH_SIZE = num_classes * target_frequency




def create_dataset():
    print("creating dataset", DATASET_URI)
    ds = hub.empty(DATASET_URI, overwrite=True)
    ds.create_tensor("images", dtype=np.float64)
    ds.create_tensor("labels", dtype=np.uint32)
    with ds:
        for label in range(num_classes):
            ds.images.extend(np.ones((num_samples_per_class, 32, 32, 3)))
            ds.labels.extend([label] * num_samples_per_class)
    print("dataset created")
    return ds



def get_best_case_batch():
    """Model performance is optimal when the average batch of batches is fully uniform."""
    values = []
    for c in range(num_classes):
        for _ in range(target_frequency):
            values.append(c)
    assert len(values) == BATCH_SIZE
    return np.array(values, dtype=int)


def get_normal_case_batch():
    """This is what class batches will look like in the real world."""
    return np.random.randint(0, num_classes, size=BATCH_SIZE, dtype=int)


def get_worst_case_batch(num_classes: int, batch_idx: int):
    """Model performance is the worst when a batch contains only 1 class."""
    use_class = batch_idx % num_classes
    values = [use_class] * BATCH_SIZE
    return np.array(values, dtype=int)


def plot_batches(batches: List[np.ndarray], titles: List[str], num_classes: int):
    assert len(batches) == len(titles)

    fig, axs = plt.subplots(len(batches))

    for i in range(len(batches)):
        axs[i].title.set_text(titles[i])
        # axs[i].hist(actual_labels, bins=bins)
        axs[i].hist(batches[i], bins=num_classes)

    plt.show()


def calculate_frequencies(tensor: torch.Tensor):
    """Calculate the frequencies of each class in the tensor."""

    freq = torch.zeros(num_classes, dtype=int)
    for x in tensor.flatten():
        freq[x] += 1
    return freq


freq_T = calculate_frequencies(get_best_case_batch()).float()


def quantify_batches(batches: List[np.ndarray], titles: List[str]):
    """The mean absolute error of the frequencies between each `batch` in `batches` is calculated with `target_batch` as the target.
    
    Minimum loss for any given batch is 0.
    """

    losses = {}

    for batch, title in zip(batches, titles):
        #assert len(batch) == BATCH_SIZE, f"{title} batch length was {len(batch)} but expected {BATCH_SIZE}"

        # get frequencies of classes in the batch
        X = torch.tensor(batch)
        # freq_X = torch.unique(X, return_counts=True)[1].float()  # TODO: better way to get frequencies
        freq_X = calculate_frequencies(X)

        loss = mae(freq_X.float(), freq_T).item()
        losses[title] = loss

    return losses


def get_hub_loss(buffer_size: int):
    shuffle = buffer_size > 0

    losses = []

    ds = hub.load(DATASET_URI)
    ptds = ds.pytorch(num_workers=WORKERS, batch_size=BATCH_SIZE, shuffle=shuffle, buffer_size=buffer_size)
    for epoch in range(EPOCHS):
        for _, T in tqdm(ptds, desc=f"hub torch buffer={buffer_size}, epoch={epoch+1}/{EPOCHS}"):
            loss = quantify_batches([T.int()], ["hub"])["hub"]
            losses.append(loss)

    return np.mean(losses)


def get_numpy_loss():
    x = get_normal_case_batch()
    return quantify_batches([x], ["numpy"])["numpy"]



if __name__ == '__main__':
    ds = create_dataset()

    buffer_sizes = [0, *np.linspace(1, len(ds), 10, dtype=int).tolist()]
    print("running on buffer sizes", buffer_sizes)

    hub_shuffled_losses = [get_hub_loss(buffer_size) for buffer_size in buffer_sizes]
    numpy_losses = [get_numpy_loss()] * len(buffer_sizes)

    plt.title("pytorch shuffling quality")

    plt.plot(buffer_sizes, hub_shuffled_losses, label="hub")
    plt.plot(buffer_sizes, numpy_losses, label="target (numpy random uniform)")

    plt.legend()
    plt.xlabel(f"shuffle buffer -- 0=unshuffled, {len(ds)}=len(ds)")
    plt.ylabel("mean absolute error to uniform distribution")

    plt.show()


