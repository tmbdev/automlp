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
import parameters as params

training_figsize = (4, 4)


class ArrayDataset(object):
    def __init__(self, inputs, targets):
        assert len(inputs) == len(targets)
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return from_numpy(self.inputs[index]), from_numpy(self.targets[index])


def repeated_iterator(loader, epochs):
    for epoch in xrange(epochs):
        for sample in loader:
            yield sample

def unbatching_iterator(loader, epochs):
    samples = []
    for epoch in xrange(epochs):
        for batch in loader:
            for i in xrange(len(batch[0])):
                sample = [x[i] for x in batch]
                yield sample

class RebatchingLoader(torchdata.DataLoader):
    def __init__(self, loader, batch_size=1, epochs=1000000000):
        assert not isinstance(loader, torchdata.Dataset)
        assert isinstance(loader, torchdata.DataLoader), type(loader)
        self.loader = loader
        self.epochs = epochs
        self.batch_size = batch_size
        self.source = None

    def set_batch_size(self, bs):
        self.batch_size = bs

    def __len__(self):
        return (self.epochs * len(self.loader) * self.loader.batch_size) // self.batch_size

    def __iter__(self):
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


def deprecated(f):
    def g():
        raise Exception("{}: deprecated".format(f))
    return g


def memlog(gpu=None, msg=""):
    return
    import torch.cuda
    import inspect
    usage = int(torch.cuda.memory_allocated(gpu) * 1e-6)
    prev = inspect.currentframe().f_back
    fname, lineno, fun, lines, index = inspect.getframeinfo(prev)


def guess_input_device(model):
    if hasattr(model, "get_input_device"):
        return model.get_input_device()
    for p in model.parameters():
        if p.device is not None:
            return p.device
    return None


def from_numpy(a):
    if isinstance(a, (int, float)):
        return a
    if isinstance(a, torch.Tensor):
        return a
    if isinstance(a, np.ndarray):
        return torch.from_numpy(a)
    return a


def smart_target(classes, outputs, mode, value=1.0):

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


def make_batch(l):
    if isinstance(l[0], int):
        return torch.LongTensor(l)
    if isinstance(l[0], float):
        return torch.FloatTensor(l)
    if isinstance(l[0], torch.Tensor):
        return torch.stack(l)
    raise ValueError("unknown batch type: {}".format(type(l[0])))


@deprecated
def as_tensor(images):
    if isinstance(images, np.ndarray):
        images = torch.from_numpy(images)
    if images.dtype == torch.uint8:
        return images.type(torch.float) / 255.0
    elif images.dtype in [torch.float32, torch.float64]:
        return images.to(torch.float32)
    else:
        raise ValueError("unknown dtype: {}".format(images.dtype))


@deprecated
def one_hot_tensor(classes, nclasses=None, value=1.0):
    classes = as_class_tensor(classes)
    if nclasses is None:
        nclasses = 1 + np.amax(classes)


def getbatch_random(dataset, batch_size, convert=from_numpy):
    n = len(dataset)
    batch = [dataset[randint(0, n - 1)] for i in range(batch_size)]
    inputs = [convert(sample[0]) for sample in batch]
    targets = [convert(sample[1]) for sample in batch]
    return inputs, targets


def dataset_batches(dataset, batch_size, convert=from_numpy):
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


def log_model(model, key, **kw):
    if "log" not in model.META:
        model.META["log"] = []
    kw["time"] = time.time()
    kw["ntrain"] = model.META.get("ntrain", -1)
    kw["key"] = key
    model.META["log"].append(kw)


def make_sgd_optimizer(model, options):
    return optim.SGD(model.parameters(),
                     lr=options.get("lr", 0.01),
                     momentum=options.get("momentum", 0.9),
                     weight_decay=options.get("weight_decay", 0.0))


def generic_make_optimizer(model, options):
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


class Trainer(object):

    def __init__(self, model, mode, device=None,
                 make_optimizer=generic_make_optimizer):
        self.model = model
        if not hasattr(model, "META"):
            model.META = {}
        if "ntrain" not in model.META:
            model.META["ntrain"] = 0
        if "loss" not in model.META:
            model.META["loss"] = 1e37
        self.make_optimizer = make_optimizer

        self.device = device or guess_input_device(model)
        self.mode = mode

        if mode.lower() == "custom":
            self.normalizer = None
            self.criterion = None
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

    def forward_batch(self, images):
        inputs = from_numpy(images).to(self.device)
        outputs = self.model.forward(inputs)
        outputs = self.normalizer(outputs)
        return outputs

    def train_batch(self, images, classes):
        self.optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            outputs = self.forward_batch(images)
            target = smart_target(classes, outputs, self.mode)
            loss = self.criterion(outputs, target.to(self.device))
            loss.backward()
            self.optimizer.step()
        self.model.META["ntrain"] += len(outputs)
        return float(loss.cpu())

    def classify(self, images, batch_size=200):
        results = []
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                outputs = self.forward_batch(images[i:i + batch_size])
                total += outputs.size(0)
                _, indexes = outputs.cpu().max(1)
                results.append(indexes.numpy())
        return np.concatenate(results)

    def set_optimizer(self, options):
        log_model(self.model, "options", options=options)
        self.optimizer = self.make_optimizer(self.model, options)

    def train_dataloader(self, loader, ntrain=100000, options=None):
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
                    loss = float((pred != targets).sum())
                else:
                    target = smart_target(
                        targets, outputs, self.mode).to(self.device)
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
        assert batch_size is not None
        loader = torchdata.DataLoader(
            dataset, batch_size=batch_size, shuffle=True)
        return self.train_dataloader(loader, ntrain=ntrain, options=options)

    def evaluate_dataset(self, dataset, classification=False, batch_size=200):
        loader = torchdata.DataLoader(
            dataset, batch_size=batch_size, shuffle=False)
        return self.evaluate_dataloader(loader, classification=classification)


default_parameters = params.ParameterSet(
    params.LogParameter("lr", 1e-6, 1e2),
    params.QuantizedLogParameter("batch_size", 5, 500)
)


def strpar(p):
    def f(x):
        if isinstance(x, float) or (isinstance(x, int) and x >= 1000000):
            x = "%.2e" % x
        else:
            x = str(x)[:10]
        return x
    return " ".join(
        ["{}={}".format(f(k), f(v)) for k, v in p.items()])


def plot_log(log, ax=None, value="loss", key="ntrain", selector="train", **kw):
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
    ax = ax or plt.gca()
    selected = [(r[key], r["options"][value])
                for r in log if r["key"] == selector]
    plt.plot(*zip(*selected), **kw)


def plot_models(models, **kw):
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
    if amlp is None:
        display.clear_output(wait=True)
    else:
        plot_models(amlp.population, **kw)


class AutoMLP(object):
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
        self.progress = progress
        self.mode = mode
        self.classification = classification
        self.after_training = after_training if after_training is not None else lambda x: x
        self.progress = progress
        self.momentum = 0.9
        self.decay = 0.0
        self.verbose = False
        self.best_model = None

    def initial_population(self, make_model):
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
        if model is None:
            return False
        if other is None:
            return True
        return model.META["loss"] < other.META["loss"]

    def train_population(self, population, ntrain=50000, verbose=False):
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
            logging.info("{} best {} {} latest {} {}".format(
                r,
                "best",
                best.META["ntrain"],
                best.META["loss"],
                "latest",
                latest.META["ntrain"],
                latest.META["loss"]))
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
                 progress=2,
                 after_training=plot_gridsearch,
                 mode="crossentropy",
                 verbose=False,
                 classification=False):
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
        self.progress = progress
        self.mode = mode
        self.classification = classification
        self.after_training = after_training if after_training is not None else lambda x: x
        self.progress = progress
        self.momentum = 0.9
        self.decay = 0.0
        self.verbose = False
        self.best_model = None

    def is_better(self, model, other):
        if model is None:
            return False
        if other is None:
            return True
        return model.META["loss"] < other.META["loss"]

    def train(self):
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
