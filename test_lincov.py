#!/usr/local/bin/python3

import numpy as np
from scipy.linalg import norm

from lincov import LinCov
from lincov.spice_loader import SpiceLoader


if __name__ == '__main__':
    loader = SpiceLoader('spacecraft')
    dt     = 0.1
    meas_dt = {'att': 0.1,
               'horizon': 0.5,
               'radial_test': 1.0}
    meas_last = {'att': 0,
                 'horizon': 0,
                 'radial_test': 0}

    l = LinCov(loader.start, dt, meas_dt, meas_last, loader = loader)

    
