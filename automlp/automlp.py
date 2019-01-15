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

from .trainer import Trainer
from .loaders import RebatchingLoader

training_figsize = (4, 4)


def memlog(gpu=None, msg=""):
    """Optional memory logging function (currently disabled).
    """
    return
    import torch.cuda
    import inspect
    usage = int(torch.cuda.memory_allocated(gpu) * 1e-6)
    prev = inspect.currentframe().f_back
    fname, lineno, fun, lines, index = inspect.getframeinfo(prev)


def make_batch(l):
    """Given a list of arrays/tensors, create a batch.

    :param l: list of arrays/tensors
    :returns: batch
    :rtype: torch.Tensor

    """
    if isinstance(l[0], int):
        return torch.LongTensor(l)
    if isinstance(l[0], float):
        return torch.FloatTensor(l)
    if isinstance(l[0], torch.Tensor):
        return torch.stack(l)
    raise ValueError("unknown batch type: {}".format(type(l[0])))


default_parameters = params.ParameterSet(
    params.LogParameter("lr", 1e-6, 1e2),
    params.QuantizedLogParameter("batch_size", 5, 500)
)


def plot_log(log, ax=None, value="loss", key="ntrain", selector="train", **kw):
    """Given a training log, plot the value against the key.
    """
    ax = ax or plt.gca()
    selected = [(r[key], r[value]) for r in log if r["key"] == selector]
    plt.plot(*zip(*selected), **kw)


def plot_options(
        log,
        ax=None,
        value="lr",
        key="ntrain",
        selector="options",
        **kw):
    """Plot training options against the key.
    """
    ax = ax or plt.gca()
    selected = [(r[key], r["options"][value])
                for r in log if r["key"] == selector]
    plt.plot(*zip(*selected), **kw)


def plot_models(models, **kw):
    """Given a list of models, plot all the training curves.
    """
    fig = kw.get("fig", plt.gcf())
    fig.clf()
    fig.add_subplot(1, 1, 1)
    ax = fig.get_axes()[0]
    ax2 = ax.twinx()
    display.clear_output(wait=True)
    ax.cla()
    ax2.cla()
    ax.set_title(kw.get("title", ""))
    ax.set_yscale(kw.get("yscale", "log"))
    ax2.set_yscale(kw.get("yscale2", "log"))
    if "ylim" in kw:
        ax.set_ylim(kw.get("ylim"))
    for model in models:
        plot_log(model.META["log"], ax, c="blue", alpha=0.5)
        plot_log(model.META["log"], ax2, selector="eval", c="red", alpha=0.5)
    display.clear_output(wait=True)
    display.display(fig)


def plot_automlp(amlp, **kw):
    """A plot function for the automlp hook.

    :param amlp: 

    """
    if amlp is None:
        display.clear_output(wait=True)
    else:
        plot_models(amlp.population, **kw)


class AutoMLP(object):
    """Automatic training of neural networks."""
    def __init__(self,
                 make_model,
                 training,
                 testing,
                 parameters=default_parameters,
                 initial_popsize=16,
                 popsize=4,
                 initial=100000,
                 epoch=50000,
                 mintrain=1e6,
                 maxtrain=3e6,
                 selection_noise=0.05,
                 eval_batch_size=128,
                 stop_no_improvement=2.0,
                 selection_key="training_loss",
                 progress=2,
                 after_training=plot_automlp,
                 mode="crossentropy",
                 verbose=False,
                 classification=False):
        """Initialize an AutoMLP instance.

        :param make_model: a function for creating a model
        :param training: the training data (DataLoader)
        :param testing: the test set (DataLoader)
        :param parameters: the parameter space (params.ParameterSet)
        :param initial_popsize: initial population size
        :param popsize: population size during training
        :param initial: initial number of training steps
        :param epoch: number of training samples between testing
        :param mintrain: minimum number of training steps before stopping
        :param maxtrain: maximum number of training steps before stopping
        :param selection_noise: how much noise to add before the selection step
        :param eval_batch_size: batch size used for evaluation
        :param stop_no_improvement: stop after this number of training steps without improvement
        :param selection_key: the key on which we select
        :param progress: a function
        :param after_training: a function that is called after each training epoch (for plotting)
        :param mode: training mode (passed to Trainer): crossentropy, "linear_mse", "logistic_mse", or dict
        :param classification: boolean, whether to use classification error rather than loss for selection
        """
        if isinstance(training, torchdata.Dataset):
            training = torchdata.DataLoader(
                training, batch_size=16, shuffle=True)
        if isinstance(testing, torchdata.Dataset):
            testing = torchdata.DataLoader(
                testing, batch_size=128, shuffle=False)
        self.make_model = make_model
        self.training = RebatchingLoader(training)
        self.testing = testing
        self.parameters = parameters
        self.initial_popsize = initial_popsize
        self.popsize = popsize
        self.initial = int(initial)
        self.epoch = int(epoch)
        self.mintrain = int(mintrain)
        self.maxtrain = int(maxtrain)
        self.maxrounds = int(max(0, maxtrain - initial) / epoch)
        self.selection_noise = selection_noise
        self.eval_batch_size = eval_batch_size
        self.stop_no_improvement = stop_no_improvement
        self.selection_key = selection_key
        self.mode = mode
        self.classification = classification
        self.after_training = after_training if after_training is not None else lambda x: x
        self.momentum = 0.9
        self.decay = 0.0
        self.best_model = None

    def initial_population(self, make_model):
        """Create the initial population.
        """
        ex = params.Exploration(self.parameters)
        population = []
        for lr in range(self.initial_popsize):
            model = make_model()
            if not hasattr(model, "META"):
                model.META = {}
            model.cpu()
            model.META["params"] = ex.pick_farthest(record=True)
            population.append(model)
        return population

    def selection(self, population):
        """Perform selection on the population.
        """
        if len(population) == 0:
            return []
        population = [model for model in population
                      if model.META["loss"] < float('inf')]
        for model in population:
            model.KEY = (1.0 + randn() * self.selection_noise) * \
                model.META["loss"]
        population = sorted(population, key=lambda m: m.KEY)
        while len(population) > self.popsize:
            del population[-1]
        return population

    def mutation(self, old_population, variants=1, variation=0.8):
        """Perform mutation on the population.
        """
        population = []
        for model in old_population:
            population += [model]
            for _ in range(variants):
                cloned = pycopy.deepcopy(model)
                cloned.META["params"] = self.parameters.mutate(
                    cloned.META["params"])
                population += [cloned]
        return population

    def is_better(self, model, other):
        """Check whether model is better than the other model.
        """
        if model is None:
            return False
        if other is None:
            return True
        return model.META["loss"] < other.META["loss"]

    def train_population(self, population, ntrain=50000, verbose=False):
        """Train the entire population for the given number of steps.
        """
        sys.stdout.write("training")
        sys.stdout.flush()
        for i, model in enumerate(population):
            sys.stdout.write(" {}".format(len(population) - i))
            sys.stdout.flush()
            ntrained = model.META.get("ntrain", 0)
            model.cuda()
            trainer = Trainer(model, mode=self.mode)
            trainer.set_optimizer(model.META["params"])
            batch_size = model.META["params"].get("batch_size")
            self.training.set_batch_size(batch_size)
            training_loss = trainer.train_dataloader(
                self.training, ntrain=ntrain)
            if isnan(training_loss):
                training_loss = float('inf')
                logging.info("{}: nan training_loss".format(dict(model.META["params"])))
                model.META['loss'] = float('inf')
                model.cpu()
                continue
            test_loss = trainer.evaluate_dataloader(
                self.testing, classification=self.classification)
            if isnan(test_loss):
                test_loss = float('inf')
                logging.info("{}: nan test_loss".format(dict(model.META["params"])))
                model.META['loss'] = float('inf')
                model.cpu()
                continue
            model.cpu()
            if self.is_better(model, self.best_model):
                self.best_model = model
        sys.stdout.write("\n")

    def train(self):
        """Perform automlp training.

        :returns: best model (also in self.best_model)
        :rtype: model instance

        """
        self.population = self.initial_population(self.make_model)
        self.train_population(self.population, ntrain=self.initial)
        self.population = self.selection(self.population)
        self.best_model = self.population[0]
        for r in xrange(self.maxrounds):
            old_population = [pycopy.deepcopy(model)
                              for model in self.population]
            self.population = self.mutation(self.population)
            self.train_population(self.population, ntrain=self.epoch)
            self.population = self.selection(self.population + old_population)
            best = self.best_model
            latest = self.population[0]
            logging.info("round {} best {}@{} latest {}@{}".format(
                r,
                best.META["loss"],
                best.META["ntrain"],
                latest.META["loss"],
                latest.META["ntrain"]))
            if latest.META["ntrain"] > self.maxtrain:
                break
            if latest.META["ntrain"] > self.mintrain and latest.META["ntrain"] - \
                    best.META["ntrain"] > self.stop_no_improvement * best.META["ntrain"]:
                break
            if self.after_training is not None:
                self.after_training(self)
        for model in self.population:
            model.cpu()
        self.after_training(None)
        return self.best_model


def plot_gridsearch(amlp, **kw):
    """Plot function for gridsearch.
    """
    if amlp is None:
        display.clear_output(wait=True)
    else:
        plot_models(amlp.population, **kw)


def scatter_losses(models, **kw):
    kw["cmap"] = kw.get("cmap", plt.cm.plasma)
    data = [
        (m.META["params"]["lr"],
         m.META["params"]["batch_size"],
         m.META["loss"]) for m in models]
    data = array(data)
    plt.xscale('log')
    plt.yscale('log')
    plt.scatter(data[:, 0], data[:, 1], c=data[:, 2], **kw)


class GridSearch(object):
    """A gridsearch drop-in replacement for AutoMLP."""
    def __init__(self,
                 make_model,
                 training,
                 testing,
                 parameters=default_parameters,
                 popsize=128,
                 initial=100000,
                 epoch=50000,
                 maxtrain=2e6,
                 eval_batch_size=200,
                 after_training=plot_gridsearch,
                 mode="crossentropy",
                 classification=False):
        """Initialize.

        :param make_model: function to create a model
        :param training: training loader
        :param testing: test loader
        :param parameters: a params.ParameterSet describing the search space
        :param popsize: population size
        :param initial: initial training
        :param epoch: per epoch training
        :param maxtrain: maximum number of training steps
        :param eval_batch_size: evaluation batch size
        :param after_training: function called after training with self as argument (for plotting)
        :param mode: training mode (crossentropy, logistic_mse, linear_mse)
        :param classification: use classification error instead of loss for evaluation

        """
        if isinstance(training, torchdata.Dataset):
            training = torchdata.DataLoader(
                training, batch_size=8, shuffle=True)
        if isinstance(training, torchdata.DataLoader):
            training = RebatchingLoader(training)
        if isinstance(testing, torchdata.Dataset):
            testing = torchdata.DataLoader(
                testing, batch_size=128, shuffle=False)
        self.make_model = make_model
        self.training = training
        self.testing = testing
        self.parameters = parameters
        self.popsize = int(popsize)
        self.initial = int(initial)
        self.epoch = int(epoch)
        self.maxtrain = int(maxtrain)
        self.eval_batch_size = eval_batch_size
        self.mode = mode
        self.classification = classification
        self.after_training = after_training if after_training is not None else lambda x: x
        self.momentum = 0.9
        self.decay = 0.0
        self.best_model = None

    def is_better(self, model, other):
        """Compare two models.
        """
        if model is None:
            return False
        if other is None:
            return True
        return model.META["loss"] < other.META["loss"]

    def train(self):
        """Perform grid search training.
        """
        self.population = []
        self.exploration = params.Exploration(self.parameters)
        while len(self.population) < self.popsize:
            model = self.make_model()
            model.META = {}
            model.META["params"] = self.exploration.pick_farthest(record=True)
            self.population.append(model)
            trainer = Trainer(model, mode=self.mode)
            trainer.set_optimizer(model.META["params"])
            model.cuda()
            while model.META.get("ntrain", 0) < self.maxtrain:
                ntrain = self.initial if model.META.get(
                    "ntrain", 0) == 0 else self.epoch
                batch_size = model.META["params"].get("batch_size")
                trainer = Trainer(model, mode=self.mode)
                trainer.set_optimizer(model.META["params"])
                self.training.set_batch_size(batch_size)
                training_loss = trainer.train_dataloader(
                    self.training, ntrain=ntrain)
                test_loss = trainer.evaluate_dataloader(
                    self.testing, classification=self.classification)
                self.after_training(self)
                logging.info("{} {}".format(len(self.population), dict(model.META["params"])))
            model.cpu()
            if self.is_better(model, self.best_model):
                self.best_model = model
        for model in self.population:
            model.cpu()
        self.after_training(None)
        return self.best_model
