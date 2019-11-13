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

def apply_styles(axes, time = None):
    for ax in axes:
        ax.set_ylim(bottom=0.0)
        #ax.set_xlim(time[0], time[-1])
        ax.grid(True)


def plot_environment(time, d):
    fig, axes = plt.subplots(3,1,sharex=True)

    fig.suptitle("Environment variables")

    axes[0].set_title("apparent angle of planet")
    axes[0].plot(time, d['earth_angle'] * 180/np.pi, label='earth', alpha=0.7)
    axes[0].plot(time, d['moon_angle'] * 180/np.pi, label='moon', alpha=0.7)
    axes[0].set_ylabel("deg")

    axes[1].set_title("planet phase angle from spacecraft perspective")
    axes[1].plot(time, d['earth_phase_angle'] * 180/np.pi, label='sun/earth', alpha=0.7)
    axes[1].plot(time, d['moon_phase_angle'] * 180/np.pi, label='sun/moon', alpha=0.7)
    axes[1].set_ylabel("deg")

    axes[2].set_title("ground station elevations")
    for key in d:
        if 'elevation' in key:
            axes[2].plot(time, d[key] * 180/np.pi, label=key[10:], alpha=0.7)
    axes[2].set_ylabel("deg")
    
    
    axes[2].set_xlabel("mission elapsed time (s)")

    axes[0].grid(True)
    axes[1].grid(True)
    axes[2].grid(True)
    axes[0].legend()
    axes[1].legend()
    axes[2].legend()

    return fig, axes


def plot_R(time, d, title, label):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("{}: {}".format(title, label))

    ax.scatter(time, np.sqrt(d['Rxx']), s=2, label='x', alpha=0.8)
    ax.scatter(time, np.sqrt(d['Ryy']), s=2, label='y', alpha=0.8)
    ax.scatter(time, np.sqrt(d['Rzz']), s=2, label='z', alpha=0.8)
    
    ax.set_xlabel("mission elapsed time (s)")
    apply_styles([ax], time)

    return fig, [ax]


def plot_inrtl(time, d, label):
    fig, axes = plt.subplots(5,1,sharex=True)

    fig.suptitle("inertial 1-sigma covariance: {}".format(label))

    axes[0].plot(time, d['srx'], label='rx', alpha=0.7)
    axes[0].plot(time, d['sry'], label='ry', alpha=0.7)
    axes[0].plot(time, d['srz'], label='rz', alpha=0.7)
    axes[0].set_ylabel('m')

    axes[1].plot(time, d['svx'], label='vx', alpha=0.7)
    axes[1].plot(time, d['svy'], label='vy', alpha=0.7)
    axes[1].plot(time, d['svz'], label='vz', alpha=0.7)
    axes[1].set_ylabel("m/s")

    axes[2].plot(time, d['sattx'] * 180/np.pi, label='about x', alpha=0.7)
    axes[2].plot(time, d['satty'] * 180/np.pi, label='about y', alpha=0.7)
    axes[2].plot(time, d['sattz'] * 180/np.pi, label='about z', alpha=0.7)
    axes[2].set_ylabel("degrees")

    axes[3].plot(time, d['sbax'], alpha=0.7)
    axes[3].plot(time, d['sbay'], alpha=0.7)
    axes[3].plot(time, d['sbaz'], alpha=0.7)
    axes[3].set_ylabel("m/s2")

    axes[4].plot(time, d['sbgx'] * 180/np.pi, alpha=0.7)
    axes[4].plot(time, d['sbgy'] * 180/np.pi, alpha=0.7)
    axes[4].plot(time, d['sbgz'] * 180/np.pi, alpha=0.7)
    axes[4].set_ylabel("deg/s")
    axes[4].set_xlabel("mission elapsed time (s)")
    
    apply_styles(axes, time)

    return fig, axes


def plot_lvlh(time, d, body, label):
    fig, axes = plt.subplots(2,1,sharex=True)

    fig.suptitle("LVLH ({}) 1-sigma covariance: {}".format(body, label))

    if body == 'moon':
        frame = 'llvlh'
    elif body == 'earth':
        frame = 'elvlh'

    labels=('downtrack', 'crosstrack', 'radial')
    
    axes[0].plot(time, d[frame+'_srx'], label=labels[0], alpha=0.7)
    axes[0].plot(time, d[frame+'_sry'], label=labels[1], alpha=0.7)
    axes[0].plot(time, d[frame+'_srz'], label=labels[2], alpha=0.7)
    axes[0].legend()
    axes[0].set_ylabel('m')    

    axes[1].plot(time, d[frame+'_svx'], label=labels[0], alpha=0.7)
    axes[1].plot(time, d[frame+'_svy'], label=labels[1], alpha=0.7)
    axes[1].plot(time, d[frame+'_svz'], label=labels[2], alpha=0.7)
    axes[1].set_ylabel("m/s")

    axes[1].set_xlabel("mission elapsed time (s)")

    apply_styles(axes, time)

    return fig, axes

    
if __name__ == '__main__':

    if len(sys.argv) < 2:
        raise SyntaxError("expected run name")

    label = sys.argv[1]
    name  = sys.argv[2]
    start = float(sys.argv[3])
    end   = float(sys.argv[4])

    config = YamlLoader(label)
    loader = SpiceLoader('spacecraft')
    start_block = find_block(start, config.block_dt)
    end_block   = find_block(end,   config.block_dt)
    
    print("start block is {}".format(start_block))
    print("end block is {}".format(end_block))
    
    d    = load_window(loader, label, start, end, name = name)
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

    
