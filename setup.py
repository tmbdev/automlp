#!/bin/env python
# Copyright (c) 2017 NVIDIA CORPORATION. All rights reserved.
# See the LICENSE file for licensing terms (BSD-style).

from __future__ import print_function

import sys
from distutils.core import setup  # , Extension, Command

setup(
    name='automlp',
    version='v0.0',
    author="Thomas Breuel",
    description="Input pipelines for deep learning.",
    packages=["automlp"],
    # data_files= [('share/ocroseg', models)],
    # scripts=scripts,
)
