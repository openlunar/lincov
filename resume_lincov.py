#!/usr/local/bin/python3

import numpy as np
from scipy.linalg import norm

from lincov import LinCov, progress_bar
from lincov.spice_loader import SpiceLoader
from lincov.yaml_loader import YamlLoader

import sys


if __name__ == '__main__':

    if len(sys.argv) < 2:
        raise SyntaxError("expected run name")

    label = sys.argv[1]    
    loader = SpiceLoader('spacecraft')

    l = LinCov.start_from(loader, label)
    while not l.finished:
        for step, mode in l.run():
            progress_bar(60, step.time - loader.start, loader.end - loader.start)


