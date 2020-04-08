#!/usr/bin/env python3

import numpy as np
from scipy.linalg import norm

from lincov import LinCov, progress_bar
from lincov.spice_loader import SpiceLoader
from lincov.yaml_loader import YamlLoader

import sys
import os

if __name__ == '__main__':

    if len(sys.argv) < 2:
        raise SyntaxError("expected run name")

    label = sys.argv[1]
    loader = SpiceLoader('spacecraft')

    if len(sys.argv) > 2:
        copy_from = sys.argv[2]
    else:
        copy_from = 'f9'

    if len(sys.argv) > 3:
        snapshot_label = sys.argv[3]
    else:
        snapshot_label = 'init'

    if os.path.exists(os.path.join("output", label)):
        raise IOError("output directory already exists")

    l = LinCov.start_from(loader, label, copy_from = copy_from, snapshot_label = snapshot_label)
    while not l.finished:
        for step, mode in l.run():
            progress_bar(60, step.time - loader.start, loader.end - loader.start)


    
