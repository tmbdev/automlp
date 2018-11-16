from __future__ import print_function

import copy
import io
import pickle
import random as pyrand
import time
from collections import OrderedDict
from math import ceil, floor, log10

import numpy as np
from numpy import clip

import scipy.spatial.distance as distance
import simplejson
from pylab import randn


class Parameter(object):
    """Base class for all parameters."""
    pass


class Constant(Parameter):
    """A parameter that is constant (not sampled or evolved). Any value is
    permitted."""

    def __init__(self, name, value):
        """Initialize a Constant.

        :param name: name of the constant
        :param value: any type
        """
        self.name = name
        self.value = value
        self.weight = 0.0

    def pick(self):
        """Pick a value (always returns constant).

        :returns: constant

        """
        return self.value

    def mutate(self, x):
        """Mutate the parameter value (noop).

        :param x: value to be mutated
        :returns: same value

        """
        return x

    def nn_value(self, x):
        """Value for nn lookups.

        :param x: constant
        :returns: 0.0

        """
        return 0.0


class UniformParameter(Parameter):
    """A parameter with a uniform distribution."""

    def __init__(self, name, lo, hi, weight=1.0, scale=0.05):
        """Parameters with a uniform distribution.

        :param name: name of the parameter
        :param lo: lower bound
        :param hi: higher bound
        :param weight: weight when picking a single parameter to mutate
        :param scale: scale on which mutation operates (relative to range)
        """
        self.name = name
        self.lo = lo
        self.hi = hi
        self.weight = weight
        self.scale = scale
        self.mdist = randn

    def pick(self):
        """Pick a random value according to distribution.

        :returns: value
        """
        return pyrand.uniform(self.lo, self.hi)

    def mutate(self, x):
        """Returns mutate value.

        :param x: value to be mutated
        :returns: mutated value

        """
        sigma = (self.hi - self.lo) * self.scale
        delta = self.mdist() * sigma
        return clip(x + delta, self.lo, self.hi)

    def clip(self, x):
        """Clips the value to the range associated with the parameter.

        :param x: value
        :returns: clipped value

        """
        return clip(x, self.lo, self.hi)

    def quant_clip(self, x):
        """Clip and quantize the value.

        :param x: value
        :returns: clipped and quantized value
        """
        x = floor(x)
        return int(clip(x, ceil(self.lo), round(self.hi) - 1))

    def nn_value(self, x):
        """Value for nn lookups.

        :param x: constant
        :returns: 0.0

        """
        return (x - self.lo) / float(self.hi - self.lo) / self.scale


class QuantizedParameter(UniformParameter):
    """A parameter with a uniform distribution and integer values."""

    def __init__(self, *args, **kw):
        """Parameters with a uniform distribution, quantized to integers.

        :param name: name of the parameter
        :param lo: lower bound
        :param hi: higher bound
        :param weight: weight when picking a single parameter to mutate
        :param scale: scale on which mutation operates (relative to range)
        """
        super(QuantizedParameter, self).__init__(*args, **kw)

    def pick(self):
        """Pick a random value according to distribution.

        :returns: value
        """
        x = super(QuantizedParameter, self).pick()
        return self.quant_clip(x)

    def mutate(self, x):
        """Returns mutate value.

        :param x: value to be mutated
        :returns: mutated value

        """
        x = super(QuantizedParameter, self).mutate(x)
        return self.quant_clip(x)


class LogParameter(UniformParameter):
    """A parameter with a log-uniform distribution."""

    def __init__(self, *args, **kw):
        """Parameter with a log-uniform distribution.

        :param name: name of the parameter
        :param lo: lower bound
        :param hi: higher bound
        :param weight: weight when picking a single parameter to mutate
        :param scale: scale on which mutation operates (relative to range)
        """
        super(LogParameter, self).__init__(*args, **kw)

    def pick(self):
        """Pick a random value according to distribution.

        :returns: value
        """
        return 10**pyrand.uniform(log10(self.lo), log10(self.hi))

    def mutate(self, x):
        """Returns mutate value.

        :param x: value to be mutated
        :returns: mutated value

        """
        sigma = (log10(self.hi) - log10(self.lo)) * self.scale
        delta = self.mdist() * sigma
        return self.clip(10 ** (log10(x) + delta))

    def nn_value(self, x):
        """Value for nn lookups.

        :param x: constant
        :returns: 0.0

        """
        return (log10(x) - log10(self.lo)) / \
            (log10(self.hi) - log10(self.lo)) / self.scale


class QuantizedLogParameter(LogParameter):
    """A parameter with a log-uniform distribution and integer values."""

    def __init__(self, *args, **kw):
        """Parameter with a log-uniform distribution, quantized to integers.

        :param name: name of the parameter
        :param lo: lower bound
        :param hi: higher bound
        :param weight: weight when picking a single parameter to mutate
        :param scale: scale on which mutation operates (relative to range)
        """
        super(QuantizedLogParameter, self).__init__(*args, **kw)

    def pick(self):
        """Pick a random value according to distribution.

        :returns: value
        """
        y = super(QuantizedLogParameter, self).pick()
        return self.quant_clip(y)

    def mutate(self, x):
        """Returns mutate value.

        :param x: value to be mutated
        :returns: mutated value

        """
        y = super(QuantizedLogParameter, self).mutate(x)
        return self.quant_clip(y)


def pick_weighted(w):
    """Given an array of weights, pick an index with probability
    proportional to the weight."""
    w = np.array(w, dtype='f')
    w /= sum(w)
    w = np.add.accumulate(w)
    return np.argmax(w > pyrand.uniform(0, 1))


class ParameterSet(object):
    """A collection of parameters; behaves much like an OrderedDict."""

    def __init__(self, *args):
        """Initialize a parameter set.

        :param args: list of parameters in order
        """
        self.parameters = OrderedDict()
        for p in args:
            self.parameters[p.name] = p

    def __iter__(self):
        """Iterate through the parameters.

        :returns: iterator
        """
        return iter(self.parameters)

    def get(self, key, dflt=None):
        """Get the parameter with the given key.

        :param key: parameter name
        :returns: parameter
        """
        return self.parameters.get(key, dflt)

    def __getitem__(self, key):
        """Get the parameter with the given key.

        :param key: parameter name
        :returns: parameter

        """
        return self.parameters[key]

    def items(self):
        """An iterator over all (name, parameter) pairs.

        :returns: iterator

        """
        return self.parameters.items()

    def __iadd__(self, p):
        """Add a parameter to the parameter set.

        :param p: parameter
        :returns: self

        """
        assert isinstance(p, Parameter)
        self.parameters[p.name] = p
        return self

    def add(self, p):
        """Add a parameter to the parameter set.

        :param p: parameter
        :returns: self
        """
        
        assert isinstance(p, Parameter)
        self.parameters[p.name] = p
        return self

    def pick(self):
        """Pick a set of values according to the parameter descriptions.

        :returns: OrderedDict of values.
        """
        
        result = OrderedDict([(p.name, p.pick())
                              for p in self.parameters.values()])
        return result

    def mutate(self, x):
        """Given an OrderedDict of parameter values, returns a mutated one.

        :param x: initial parameter values
        :returns: new parameter values
        """
        assert isinstance(x, OrderedDict)
        weights = [p.weight for p in self.parameters.values()]
        index = pick_weighted(weights)
        key = [p.name for p in self.parameters.values()][index]
        y = copy.copy(x)
        y[key] = self.parameters[key].mutate(y[key])
        return y

    def nn_vector(self, v):
        """Convert a set of parameter values into a floating point vector.

        This is used for nearest neighbor lookups.

        :param v: OrderedDict of parameters
        :returns: ndarray of float
        """
        assert len(v) == len(self.parameters)
        assert isinstance(v, OrderedDict)
        assert tuple(self.parameters.keys()) == tuple(v.keys())
        return np.array([p.nn_value(x)
                         for p, x in zip(self.parameters.values(), v.values())])


class Exploration(object):

    """Exploring a parameter space with random sampling. Optionally distributed."""

    def __init__(self, parameters):
        """Explore a parameter space using random maximum discrepancy sequences.

        :param parameters: ParameterSet to be explored.
        """
        self.parameters = parameters
        self.results = []
        self.ncandidates = 10
        self.fcandidates = 2.0
        self.initial = 10
        self.nnearest = 20
        self.red = None
        self.last_reload = 0

    def redis_connect(self, *args, **kw):
        """For exploration in parallel, connect to a Redis instance.

        :param *args: Redis connection args
        :param clear: If True (default) start a new exploration.
        :param key: Redis key prefix for this exploration.
        :param **kw: Redis connection args
        :returns: 
        :rtype: 

        """
        
        import redis
        self.red_key = kw.get("key", "exploration")
        self.red_connection = kw
        clear = kw.get("clear", True)
        if "key" in kw:
            del kw["key"]
        if "clear" in kw:
            del kw["clear"]
        self.red = redis.StrictRedis(*args, **kw)
        if clear:
            self.red.delete(self.key("results"))
            self.red.delete(self.key("models"))
            self.red.delete(self.key("metas"))
            self.red.set(self.key("parameters"), pickle.dumps(self.parameters))
        else:
            self.parameters = pickle.loads(
                self.red.get(self.key("parameters")))
            self.redis_reload()

    def save_model(self, model, key=None):
        """Save a model, either locally or in Redis.

        :param model: PyTorch model
        :param key: name to save under
        """
        import uuid
        import torch
        if key is None:
            key = uuid.uuid1().hex
        if self.red is None:
            torch.save(model, key + ".pth")
            with open(key + ".meta", "wb") as stream:
                simplejson.dump(model.META, stream, indent=4)
            return key
        stream = io.BytesIO()
        torch.save(model, stream)
        self.red.hset(self.key("models"), key, stream.getvalue())
        if hasattr(model, "META"):
            self.red.hset(
                self.key("metas"),
                key,
                simplejson.dumps(
                    model.META,
                    indent=4))
        return key

    def load_model(self, key):
        """Load a model, either from Redis or from disk.

        :param key: model name
        :returns: loaded PyTorch model
        """
        import io
        import torch
        if self.red is None:
            return torch.load(key + ".pth")
        model = self.red.hget(self.key("models"), key)
        with io.BytesIO(model) as stream:
            return torch.load(stream)

    def redis_reload(self, cache_time=-1):
        """Reload Redis data for exploration.

        :param cache_time: time to cache loaded data (-1=no cache)
        """
        if not self.red:
            return
        if time.time() - self.last_reload < cache_time:
            return
        results = self.red.lrange(self.key("results"), 0, 1000000)
        self.last_reload = time.time()
        if results is not None:
            self.results = [pickle.loads(result) for result in results]
        else:
            self.results = []

    def redis_add_result(self, q, v, model=None):
        """Add a result to the Redis server.

        :param q: quality of the result
        :param v: parameter vector of the result
        :param model: model name of the result
        :returns: 
        :rtype: 

        """
        if not self.red:
            return
        result = pickle.dumps((q, v, model))
        self.red.rpush(self.key("results"), result)

    def key(self, suffix):
        """Compute a Redis key.

        :param suffix: suffix to be combined with prefix
        :returns: Redis key

        """
        assert isinstance(suffix, str)
        return "{}.{}".format(self.red_key, suffix)

    def add_result(self, result, quality=1e37, model=None):
        """Add a result to the exploration.

        :param result: set of parameters
        :param quality: quality of the parameters
        :param model: associated model name
        """
        assert isinstance(quality, (int, float))
        assert isinstance(result, OrderedDict)
        self.results.append((quality, result, model))
        self.redis_add_result(quality, result, model)

    def __len__(self):
        """Number of parameter vectors in exploration.

        :returns: # parameter vectors
        :rtype: int

        """
        self.redis_reload(cache_time=1.0)
        return len(self.results)

    def nn_results(self):
        """Return all parameter vectors for nn lookup.

        :returns: matrix containing parameter vectors as rows
        :rtype: ndarray

        """
        return np.array([self.parameters.nn_vector(r[1])
                         for r in self.results])

    def pick_farthest(self, choices=None, record=False):
        """

        :param choices: optional set of choices (otherwise provided by self.pick) 
        :param record: record vector internally (with infinite cost)
        :returns: parameter vector

        """
        self.redis_reload()
        if len(self.results) == 0:
            result = self.parameters.pick()
            if record:
                self.add_result(result)
            return result
        if choices is None:
            choices = [self.parameters.pick()
                       for _ in range(self.ncandidates)]
        nn_choices = np.array([self.parameters.nn_vector(v) for v in choices])
        nn_existing = self.nn_results()
        dists = distance.cdist(nn_existing, nn_choices)
        dists = np.amin(dists, axis=0)
        index = np.argmax(dists)
        result = choices[index]
        if record:
            self.add_result(result)
        return result

    def find_nearest(self, p):
        """Find the nearest existing parameter vector to a given one.

        :param p: parameter vector
        :returns: index of nearest parameter vector
        :rtype: int

        """
        self.redis_reload()
        nn_choice = np.array([self.parameters.nn_vector(p)])
        nn_existing = self.nn_results()
        dists = distance.cdist(nn_existing, nn_choice)
        return np.argmin(dists[:, 0])

    def num_candidates(self):
        return max(self.ncandidates, int(self.fcandidates * len(self)))

    def pick_top(self):
        """Nonparametric optimizer.

        Picks new parameter vector by maximum discrepancy among nearest neighbors.
        (This can take a long time for small target regions.)

        :returns: 
        :rtype: 

        """
        self.redis_reload()
        threshold = sorted([r[0] for r in self.results])[self.nnearest - 1]
        choices = []
        while len(choices) < self.ncandidates:
            p = self.parameters.pick()
            index = self.find_nearest(p)
            if self.results[index][0] > threshold:
                continue
            choices.append(p)
        return self.pick_farthest(choices)

    def pick(self):
        """Nonparameteric optimizer
.
        Randomly exlpores the entire space for self.initial steps with
        self.pick_farthest, then switches to self.pick_top

        :returns: parameter vector

        """
        self.redis_reload()
        if len(self.results) < max(self.initial, self.nnearest):
            return self.pick_farthest()
        else:
            return self.pick_top()
