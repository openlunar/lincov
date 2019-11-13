#!/usr/bin/env python3

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

from plot import plot_inrtl, plot_lvlh, plot_environment, plot_R

def print_usage():
    print("Plot a single entry from each block (and first and last from the final block) to summarize a dataset.")
    print("")
    print("usage: {} config dataset [first [last]]".format(sys.argv[0]))
    print("  options:")
    print("    config refers to the configuration YAML file (without .yml extension)")
    print("    dataset is one of state_sigma, environment, or any measurement type")
    print("      with a covariance which changes (e.g. moon_horizon, earth_horizon)")
    print("      over the course of the analysis")
    print("    first is the mission time to start at for loading the dataset blocks")
    print("    last is the mission time to end at (will try to find the last if not")
    print("      given")
    

if __name__ == '__main__':

    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        print_usage()

    label = sys.argv[1]
    name  = sys.argv[2]
    
    config = YamlLoader(label)
    loader = SpiceLoader('spacecraft')
    
    if len(sys.argv) > 3:
        first_time = float(sys.argv[3])
        first      = find_block(first_time, config.block_dt)
    else:
        first      = 1

    if len(sys.argv) > 4:
        last_time = float(sys.argv[4])
        last      = find_block(last_time, config.block_dt)
    else:
        last      = LinCov.find_latest_count(label)
        

    
    d = load_sample(label, first, last, name)
    
    time = np.array(d['time'] - loader.start)

    if name == 'state_sigma':
        plot_inrtl(time, d, label)
        plot_lvlh(time, d, 'moon', label)
        plot_lvlh(time, d, 'earth', label)
    elif name == 'environment':
        plot_environment(np.array(d['time'] - loader.start), d, label)
    else:
        plot_R(time, d, name, label)
        

    plt.show()
