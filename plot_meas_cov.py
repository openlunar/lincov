#!/usr/local/bin/python3

from spiceypy import spiceypy as spice
from lincov.spice_loader import SpiceLoader

import pandas as pd
import numpy as np
from scipy.linalg import norm
from scipy.stats import chi2

import sys

import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D, art3d

import lincov.frames as frames
from lincov.plot_utilities import *
from lincov.reader import *
from lincov import LinCov

if __name__ == '__main__':

    if len(sys.argv) < 4:
        raise SyntaxError("expected run name, index number, body name")

    label = sys.argv[1]
    meas  = sys.argv[2]
    time  = float(sys.argv[3])
    
    loader = SpiceLoader('spacecraft')

    P, time = LinCov.load_covariance(name, count)
    plot_lvlh_covariance(P, time, body)


    plt.show()
