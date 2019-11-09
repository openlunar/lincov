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

from plot_lincov import plot_inrtl, plot_lvlh, plot_environment, plot_R

if __name__ == '__main__':

    label = sys.argv[1]
    name  = sys.argv[2]
    if len(sys.argv) > 3:
        first = int(sys.argv[3])
    else:
        first = 1
        
    config = YamlLoader(label)
    loader = SpiceLoader('spacecraft')
    count = LinCov.find_latest_count(label)
    
    d = load_sample(label, first, count, name)
    
    time = np.array(d['time'] - loader.start)

    if name == 'state_sigma':
        plot_inrtl(time, d)
        plot_lvlh(time, d, 'moon')
        plot_lvlh(time, d, 'earth')
    elif name == 'environment':
        plot_environment(np.array(d['time'] - loader.start), d)
    else:
        plot_R(time, d, name)
        

    plt.show()
