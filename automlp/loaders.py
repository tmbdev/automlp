from __future__ import print_function
import copy as pycopy
import logging
import sys
import time
from random import randint, uniform

import matplotlib.pyplot as plt
import numpy as np
import pylab
import torch
import torch.utils.data as torchdata
import torchvision
from IPython import display
from numpy import *
from pylab import randn
from scipy import ndimage as ndi
from torch import nn, optim

import h5py
import params


def from_numpy(a):
    """Robust conversion from anything to torch.Tensor

    :param a: anything
    :returns: torch.Tensor or Python number

    """
    if isinstance(a, (int, float)):
        return a
    if isinstance(a, torch.Tensor):
        return a
    if isinstance(a, np.ndarray):
        return torch.from_numpy(a)
    return a


def one_hot_tensor(classes, nclasses, value=1.0):
    """Construct a one-hot encoding for the given classes.

    :param classes: classes (torch.Tensor)
    :param nclasses: number of classes
    :param value: value of "one"
    :returns: one-hot encoding

    This works for arbitrary rank tensors. It assumes
    that dim 0 is the batch and inserts the one-hot encoding
    before dim 1.
    """
    classes = classes.to(torch.int64)
    targets = torch.zeros(len(classes), nclasses)
    targets = targets.scatter(
        1, classes.unsqueeze(1).to(
            targets.device), value)
    return targets


def getbatch_random(dataset, batch_size, convert=from_numpy):
    """Given a dataset, return a randomly chosen batch.

    :param dataset: a Dataset
    :param batch_size: desired batch size
    :param convert: conversion function
    """
    n = len(dataset)
    batch = [dataset[randint(0, n - 1)] for i in range(batch_size)]
    inputs = [convert(sample[0]) for sample in batch]
    targets = [convert(sample[1]) for sample in batch]
    return inputs, targets


def dataset_batches(dataset, batch_size, convert=from_numpy):
    """Iterate over batches from a Dataset.

    :param dataset: a Dataset
    :param batch_size: desired batch size
    :param convert: conversion function
    """
    inputs = []
    targets = []
    for sample in dataset:
        inputs.append(convert(sample[0]))
        targets.append(convert(sample[1]))
        if len(inputs) >= batch_size:
            yield inputs, targets
            inputs, targets = [], []
    if len(inputs) >= batch_size:
        yield inputs, targets


class ArrayDataset(object):
    """Convert arrays into a dataset."""
    def __init__(self, inputs, targets):
        """Convert arrays into a Dataset.

        :param inputs: input array
        :param targets: targets array

        Dim 0 is assumed to be the sample number, the remaining dimensions are as is.
        Works for both Numpy and torch.Tensor arrays.

        """
        
        assert len(inputs) == len(targets)
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return from_numpy(self.inputs[index]), from_numpy(self.targets[index])


def repeated_iterator(loader, epochs):
    """Iterate over an object repeatedly.

    :param loader: object to be iterated over (usually DataLoader)
    :param epochs: number of epochs to iterate
    :returns: iterator

    """

    for epoch in xrange(epochs):
        for sample in loader:
            yield sample

def unbatching_iterator(loader, epochs):
    """Iterate over individual samples, given a DataLoader.

    :param loader: DataLoader
    :param epochs: number of epochs to iterate
    :returns: iterator over individual samples (unbatched)

    """
    samples = []
    for epoch in xrange(epochs):
        for batch in loader:
            for i in xrange(len(batch[0])):
                sample = [x[i] for x in batch]
                yield sample

class RebatchingLoader(torchdata.DataLoader):
    def __init__(self, loader, batch_size=1, epochs=1000000000):
        """A wrapper around a DataLoader that changes batch sizes.

        :param loader: original loader
        :param batch_size: new batch size
        :param epochs: number of epochs to iterate for

        """
        assert not isinstance(loader, torchdata.Dataset)
        assert isinstance(loader, torchdata.DataLoader), type(loader)
        self.loader = loader
        self.epochs = epochs
        self.batch_size = batch_size
        self.source = None

    def set_batch_size(self, bs):
        """Dynamically change the batch size.

        :param bs: new batch size
        """
        self.batch_size = bs

    def __len__(self):
        """Total # samples.

        :returns: #samples (approximate)
        """
        return (self.epochs * len(self.loader) * self.loader.batch_size) // self.batch_size

    def __iter__(self):
        """Create an iterator over rebatched samples.

        :returns: iterator
        """
        samples = []
        for sample in unbatching_iterator(self.loader, self.epochs):
            samples.append(sample)
            if len(samples) >= self.batch_size:
                samples = map(list, zip(*samples))[:self.batch_size]
                yield [torch.stack(x) for x in samples]
                samples = samples[self.batch_size:]
        if len(samples)>0:
            # may lose some samples if we change batch size at end
            samples = map(list, zip(*samples))
            yield [torch.stack(x) for x in samples]


