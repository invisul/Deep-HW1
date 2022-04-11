import math
import numpy as np
import torch
import torch.utils.data
from typing import Sized, Iterator
from torch.utils.data import Dataset, Sampler


class FirstLastSampler(Sampler):
    """
    A sampler that returns elements in a first-last order.
    """

    def __init__(self, data_source: Sized):
        """
        :param data_source: Source of data, can be anything that has a len(),
        since we only care about its number of elements.
        """
        super().__init__(data_source)
        self.data_source = data_source

    def __iter__(self) -> Iterator[int]:
        for iter_num in range(len(self.data_source)):
            if iter_num % 2 == 0:
                yield iter_num//2
            else:
                yield len(self.data_source)-1-iter_num//2

    def __len__(self):
        return len(self.data_source)


class SamplerByListOfIndices(Sampler):
    """
    A sampler that returns elements in a first-last order.
    """

    def __init__(self, data_source: Sized, indices: list):
        """
        :param data_source: Source of data, can be anything that has a len(),
        since we only care about its number of elements.
        """
        super().__init__(data_source)
        if len(indices) > len(data_source):
            raise ValueError("list of indices is larger than the number of elements in the data source")
        self.data_source = data_source
        self.indices = indices

    def __iter__(self) -> Iterator[int]:
        for index in self.indices:
            yield index

    def __len__(self):
        return len(self.indices)


def create_train_validation_loaders(
    dataset: Dataset, validation_ratio, batch_size=100, num_workers=2
):
    """
    Splits a dataset into a train and validation set, returning a
    DataLoader for each.
    :param dataset: The dataset to split.
    :param validation_ratio: Ratio (in range 0,1) of the validation set size to
        total dataset size.
    :param batch_size: Batch size the loaders will return from each set.
    :param num_workers: Number of workers to pass to dataloader init.
    :return: A tuple of train and validation DataLoader instances.
    """
    if not (0.0 < validation_ratio < 1.0):
        raise ValueError(validation_ratio)

    # 1) random shuffle for indices
    indices = list(np.random.permutation(len(dataset)))

    # 2) split by validation ratio
    index_of_split = round(validation_ratio * len(dataset))
    indices_of_validation_set = indices[:index_of_split]
    indices_of_training_set = indices[index_of_split:]

    # 3) create samplers
    # valid_set = dataset[indices_of_validation_set]
    dl_valid = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                                           sampler=SamplerByListOfIndices(dataset, indices_of_validation_set))
    # train_set = dataset[indices_of_training_set]
    dl_train = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                                           sampler=SamplerByListOfIndices(dataset, indices_of_training_set))

    return dl_train, dl_valid
