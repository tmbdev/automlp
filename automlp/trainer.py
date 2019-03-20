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

from .loaders import from_numpy, one_hot_tensor

def deprecated(f):
    def g():
        raise Exception("{}: deprecated".format(f))
    return g


@deprecated
def as_tensor(a):
    """Convert a to torch.Tensor.
    """
    if isinstance(a, np.ndarray):
        a = torch.from_numpy(a)
    if a.dtype == torch.uint8:
        return a.type(torch.float) / 255.0
    elif a.dtype in [torch.float32, torch.float64]:
        return a.to(torch.float32)
    else:
        raise ValueError("unknown dtype: {}".format(a.dtype))


def log_model(model, key, **kw):
    """Log information to model.META["log"]
    """
    if "log" not in model.META:
        model.META["log"] = []
    kw["time"] = time.time()
    kw["ntrain"] = model.META.get("ntrain", -1)
    kw["key"] = key
    model.META["log"].append(kw)


def make_sgd_optimizer(model, options):
    """Create an SGD optimizer with the given options.
    """
    return optim.SGD(model.parameters(),
                     lr=options.get("lr", 0.01),
                     momentum=options.get("momentum", 0.9),
                     weight_decay=options.get("weight_decay", 0.0))


def generic_make_optimizer(model, options):
    """Create an optimizer based on the options.

    optimizer["__type__"] is one of sgd, adam, adagrad, or rprop; the rest
    of the dictionary contains the optimizer parameters (see the PyTorch
    documentation)

    :param model: the model for which the optimizer is to be created
    :param options: the options
    :returns:
    :rtype:

    """
    opt = options.get("__type__", "sgd")
    if opt == "sgd":
        return optim.SGD(model.parameters(),
                         lr=options.get("lr", 0.01),
                         momentum=options.get("momentum", 0.9),
                         weight_decay=options.get("weight_decay", 0.0))
    elif opt == "adam":
        beta1 = 1.0 - options.get("beta1", 0.1)
        beta2 = 1.0 - options.get("beta2", 0.001)
        return optim.Adam(model.parameters(),
                          lr=options.get("lr", 1e-3),
                          betas=(beta1, beta2),
                          eps=options.get("eps", 1e-8),
                          weight_decay=options.get("weight_decay", 0.0))
    elif opt == "adagrad":
        return optim.Adagrad(model.parameters(),
                             lr=options.get("lr", 1e-3),
                             lr_decay=options.get("lr_decay", 0),
                             weight_decay=options.get("weight_decay", 0.0))
    elif opt == "rprop":
        eta1 = options.get("eta1", 0.5)
        eta2 = options.get("eta2", 1.2)
        minstep = options.get("minstep", 1e-6)
        maxstep = options.get("maxstep", 50)
        return optim.Rprop(model.parameters(),
                           lr=options.get("lr", 1e-2),
                           etas=(eta1, eta2),
                           step_sizes=(minstep, maxstep))
    else:
        raise ValueError("unknown optimizer: {}".format(opt))


def guess_input_device(model):
    """Given a model, guess what device it takes its inputs on.
    """
    if hasattr(model, "get_input_device"):
        return model.get_input_device()
    for p in model.parameters():
        if p.device is not None:
            return p.device
    return None


def smart_target(classes, outputs, mode, value=1.0):
    """Given target classes and network outputs, construct an SGD training target.

    :param classes: classes, either as list of integers or one-hot encoding
    :param outputs: actual outputs from the network (for shape)
    :param mode: model (crossentropy, logistic_mse, linear_mse)
    :param value: value for one-hot encoding
    :returns: tensor with same shape as outputs
    """

    # for crossentropy, just return the class list

    if mode == "crossentropy":
        classes = from_numpy(classes).to(torch.int64)
        assert classes.dim() == outputs.dim() - 1
        return classes

    # if the targets have the output shape, use them directly

    classes = from_numpy(classes)
    if classes.dim() == outputs.dim():
        assert classes.size() == outputs.size()
        assert classes.dtype in [torch.float16, torch.float32, torch.float64]
        return classes

    # construct a one-hot encoding otherwise

    classes = classes.to(torch.int64)
    assert classes.dim() == outputs.dim() - 1
    targets = torch.zeros_like(outputs)
    targets = targets.scatter(
        1, classes.unsqueeze(1).to(
            targets.device), value)
    return targets


class Trainer(object):
    """A wrapper that helps with training a model."""

    def __init__(self, model, mode="crossentropy", device=None,
                 make_optimizer=generic_make_optimizer):
        """Wrap a trainer around a model.

        :param model: model to be trained
        :param mode: training mode (crossentropy, linear_mse, logistic_mse, or dict)
        :param device: input device for model (guesses if None)
        :param make_optimizer: function for instantiating the optimizer
        """
        self.model = model
        if not hasattr(model, "META"):
            model.META = {}
        if "ntrain" not in model.META:
            model.META["ntrain"] = 0
        if "loss" not in model.META:
            model.META["loss"] = 1e37

        self.make_optimizer = make_optimizer
        self.device = device or guess_input_device(model)
        self.set_mode(mode)


    def set_mode(self, mode):
        """Set the training mode.

        :param mode: training mode

        Training modes:
        - crossentropy - uses nn.CrossEntropyLoss; target must be classes
        - logistic_mse - passes output through nn.Sigmoid, then uses nn.MSELoss
        - linear_mse - uses nn.MSELoss directly on output
        - a dictionary: defines normalizer, make_target, and criterion functions

        The crossentropy and logistic_mse modes take exactly the same
        models for classification.

        When a dictionary is given:
        - mode["normalizer"] is a function applied to the output of the model (default: identity)
        - mode["target"] is a function that makes a target from the given input (default: smart_target)
        - mode["criterion"] is an nn.Criterion (default: nn.MSELoss)

        """
        self.make_target = lambda t, x: smart_target(t, x, mode)
        self.mode = mode
        if isinstance(mode, dict):
            self.normalizer = mode.get("normalizer", lambda x: x)
            self.make_target = mode.get("target", self.make_target)
            self.criterion = mode.get("criterion", nn.MSELoss()).to(self.device)
        elif mode.lower() == "crossentropy":
            self.normalizer = lambda x: x
            self.criterion = nn.CrossEntropyLoss().to(self.device)
        elif mode.lower() == "logistic_mse":
            self.normalizer = torch.sigmoid
            self.criterion = nn.MSELoss().to(self.device)
        elif mode.lower() == "linear_mse":
            self.normalizer = lambda x: x
            self.criterion = nn.MSELoss().to(self.device)
        elif isinstance(mode, str):
            raise ValueError(
                "mode must be one of: crossentropy, logistic_mse, linear_mse")
        else:
            self.normalizer = lambda x: x
            self.criterion = mode

    def forward_batch(self, batch):
        """Forward propagate a batch through the model.

        :param batch:
        :returns: outputs
        :rtype: torch.Tensor
        """
        inputs = from_numpy(batch).to(self.device)
        outputs = self.model.forward(inputs)
        outputs = self.normalizer(outputs)
        return outputs

    def train_batch(self, inputs, classes):
        """Train a batch.

        :param inputs: inputs to model
        :param classes: targets/classes (classes if rank is one less than output, otherwise target)
        :returns: total loss for batch
        :rtype: float

        """
        self.optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            outputs = self.forward_batch(inputs)
            target = self.make_target(classes, outputs).to(self.device)
            loss = self.criterion(outputs, target).to(self.device)
            loss.backward()
            self.optimizer.step()
        self.model.META["ntrain"] += len(outputs)
        return float(loss.cpu())

    def classify(self, inputs, batch_size=200):
        """Perform classification on a tensor of inputs

        :param inputs: tensor of inputs (first dim = batch dim)
        :param batch_size: batch size for classification
        :returns: clssifications
        :rtype: numpy array

        """
        results = []
        with torch.no_grad():
            for i in range(0, len(inputs), batch_size):
                outputs = self.forward_batch(inputs[i:i + batch_size])
                total += outputs.size(0)
                _, indexes = outputs.cpu().max(1)
                results.append(indexes.numpy())
        return np.concatenate(results)

    def set_optimizer(self, options):
        """Set the optimizer/parameters (including learning rate).

        :param options: dictionary of optimizer options

        The dictionary contains "__type__" for the optimizer type
        and otherwise the optimizer parameters as keyword arguments.
        See `generic_make_optimizer` for options.

        """
        log_model(self.model, "options", options=options)
        self.optimizer = self.make_optimizer(self.model, options)

    def train_dataloader(self, loader, ntrain=100000, options=None):
        """Given a dataloader, train the model on it for the given number of samples.

        :param loader: nn.DataLoader
        :param ntrain: number of samples to train with
        :param options: optimizer options (learning rate, etc.)
        :returns: average losses towards end of training

        NB: Use an automlp.RebatchingLoader to change the batchsize.

        """
        if options is not None:
            self.set_optimizer(options)
        losses = []
        n = 0
        while n < ntrain:
            for inputs, targets in loader:
                n += len(inputs)
                if n >= ntrain:
                    break
                loss = self.train_batch(inputs, targets)
                losses.append(loss)
        result = mean(losses[-100:])
        log_model(self.model, "train", loss=result)
        self.model.META["loss"] = result
        return result

    def evaluate_dataloader(self, loader, classification=False):
        """Given a dataloader, evaluate the model on the samples.

        :param loader: nn.DataLoader
        :param classification: evaluate using classification accuracy, not loss
        :returns: average loss
        """
        totals = []
        losses = []
        with torch.no_grad():
            for inputs, targets in loader:
                outputs = self.forward_batch(inputs)
                if classification:
                    _, pred = outputs.cpu().max(1)
                    if targets.size() == outputs.size():
                        targets = targets.cpu().max(1)
                    pred = pred.reshape(-1)
                    targets = targets.reshape(-1)
                    loss = float((pred.int() != targets.int()).int().sum())
                else:
                    target = self.make_target(targets, outputs).to(self.device)
                    loss = float(self.criterion(outputs, target))
                totals += [outputs.size(0)]
                losses += [loss]
        if classification:
            result = sum(losses) / float(sum(totals))
        else:
            result = mean(losses)
        log_model(self.model, "eval", type="loss", loss=result,
                  total=sum(totals), ntrain=self.model.META["ntrain"])
        self.model.META["loss"] = result
        return result

    def train_dataset(self, dataset, ntrain=None, options=None, batch_size=None):
        """Train on a dataset.

        :param dataset: nn.Dataset
        :param ntrain: number of samples to train for
        :param options: optimizer options
        :returns: training loss estimate (towards end)

        The batch_size is usually taken from options, but can be overridden.

        """

        assert batch_size is not None
        loader = torchdata.DataLoader(
            dataset, batch_size=batch_size, shuffle=True)
        return self.train_dataloader(loader, ntrain=ntrain, options=options)

    def evaluate_dataset(self, dataset, classification=False, batch_size=200):
        """Evaluate on a dataset.

        :param dataset: nn.Dataset
        :param classification: evaluate classification error instead of loss
        :param batch_size: batch size
        :returns: average loss

        """
        loader = torchdata.DataLoader(
            dataset, batch_size=batch_size, shuffle=False)
        return self.evaluate_dataloader(loader, classification=classification)


