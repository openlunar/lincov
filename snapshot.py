#!/usr/bin/env python3

"""This script uses an existing linear covariance analysis as
input. Suppose you have covariances saved at 0, 600, 1200, and 1800
seconds (mission elapsed time), but you want one at 300 seconds. You
can run this script with the arguments 'label 300 three_hundred' and
it will generate a time, covariance, and metadata snapshot named
three_hundred.

Probably you want to do something like this:

  ./start.py my_lincov f9 init

which will start the analysis named 'my_lincov' from an analysis named
'f9' that has a snapshot called 'init'. When that's finished,

  ./snapshot.py my_lincov 360021.0 loi

to create a snapshot of the covariance at LOI.

The only thing you can't really do with this script is to create a
snapshot a few seconds after another snapshot. For that you'd need to
write a custom script that combines start.py with snapshot.py.

"""


import numpy as np
from scipy.linalg import norm

from lincov import LinCov, progress_bar
from lincov.spice_loader import SpiceLoader
from lincov.yaml_loader import YamlLoader
from lincov.reader import find_block

import sys


if __name__ == '__main__':

    if len(sys.argv) < 2:
        raise SyntaxError("expected run name")

    label                = sys.argv[1]
    mission_elapsed_time = float(sys.argv[2])
    snapshot_label       = sys.argv[3]

    config = YamlLoader(label)
    count  = find_block(mission_elapsed_time, config.block_dt) - 1
    loader = SpiceLoader('spacecraft')

    l = LinCov.start_from(loader, label, count)
    while not l.finished:
        for step, mode in l.run():
            if step.time >= loader.start + mission_elapsed_time:
                LinCov.save_covariance(label, step.P, step.time, snapshot_label = snapshot_label)
                step.save_metadata(snapshot_label)
                print("Snapshot saved.")
                l.finished = True
                break


