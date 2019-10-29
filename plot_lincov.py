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
from lincov import LinCov

def apply_styles(axes, time):
    for ax in axes:
        ax.set_ylim(bottom=0.0)
        #ax.set_xlim(time[0], time[-1])
        ax.grid(True)


def plot_inrtl(time, d):
    fig, axes = plt.subplots(5,1,sharex=True)

    fig.suptitle("inertial 1-sigma covariance")

    axes[0].plot(time, d['srx'], label='rx', alpha=0.5)
    axes[0].plot(time, d['sry'], label='ry', alpha=0.5)
    axes[0].plot(time, d['srz'], label='rz', alpha=0.5)
    axes[0].set_ylabel('m')

    axes[1].plot(time, d['svx'], label='vx', alpha=0.5)
    axes[1].plot(time, d['svy'], label='vy', alpha=0.5)
    axes[1].plot(time, d['svz'], label='vz', alpha=0.5)
    axes[1].set_ylabel("m/s")

    axes[2].plot(time, d['sattx'] * 180/np.pi, label='about x', alpha=0.5)
    axes[2].plot(time, d['satty'] * 180/np.pi, label='about y', alpha=0.5)
    axes[2].plot(time, d['sattz'] * 180/np.pi, label='about z', alpha=0.5)
    axes[2].set_ylabel("degrees")

    axes[3].plot(time, d['sbax'], alpha=0.5)
    axes[3].plot(time, d['sbay'], alpha=0.5)
    axes[3].plot(time, d['sbaz'], alpha=0.5)
    axes[3].set_ylabel("m/s2")

    axes[4].plot(time, d['sbgx'] * 180/np.pi, alpha=0.5)
    axes[4].plot(time, d['sbgy'] * 180/np.pi, alpha=0.5)
    axes[4].plot(time, d['sbgz'] * 180/np.pi, alpha=0.5)
    axes[4].set_ylabel("deg/s")
    axes[4].set_xlabel("mission elapsed time (s)")
    
    apply_styles(axes, time)

    return fig, axes


def plot_lvlh(time, d, body = 'moon'):
    fig, axes = plt.subplots(2,1,sharex=True)

    fig.suptitle("LVLH ({}) 1-sigma covariance".format(body))

    if body == 'moon':
        frame = 'llvlh'
    elif body == 'earth':
        frame = 'elvlh'

    labels=('v-bar', 'h-bar', 'r-bar')
    
    axes[0].plot(time, d[frame+'_srx'], label=labels[0], alpha=0.5)
    axes[0].plot(time, d[frame+'_sry'], label=labels[1], alpha=0.5)
    axes[0].plot(time, d[frame+'_srz'], label=labels[2], alpha=0.5)
    axes[0].legend()
    axes[0].set_ylabel('m')    

    axes[1].plot(time, d[frame+'_svx'], label=labels[0], alpha=0.5)
    axes[1].plot(time, d[frame+'_svy'], label=labels[1], alpha=0.5)
    axes[1].plot(time, d[frame+'_svz'], label=labels[2], alpha=0.5)
    axes[1].set_ylabel("m/s")

    axes[1].set_xlabel("mission elapsed time (s)")

    apply_styles(axes, time)

    return fig, axes


def plot_lvlh_covariance(name, body_id = 399, object_id = -5440):
    P, time = LinCov.load_covariance(name)

    if body_id == 'earth':
        body_id = 399
    elif body_id == 'moon':
        body_id = 301

    # Get LVLH frame
    x_inrtl = spice.spkez(object_id, time, 'J2000', 'NONE', body_id)[0] * 1000.0
    T_inrtl_to_lvlh = frames.compute_T_inrtl_to_lvlh( x_inrtl )

    # Transform covariance to LVLH frame
    P_lvlh = T_inrtl_to_lvlh.dot(P[0:6,0:6]).dot(T_inrtl_to_lvlh.T)
    
    fig1, pos_axes = error_ellipsoid(P_lvlh[0:3,0:3], dof=3, xlabel='downtrack (m)', ylabel='crosstrack (m)', zlabel='radial (m)')
    fig2, vel_axes = error_ellipsoid(P_lvlh[3:6,3:6], dof=3, xlabel='downtrack (m/s)', ylabel='crosstrack (m/s)', zlabel='radial (m/s)')

    return (fig1, fig2), (pos_axes, vel_axes)

def plot_covariance(P, **kwargs):
    fig, axes = error_ellipsoid(P, dof=P.shape[0], **kwargs)

    return fig, axes


if __name__ == '__main__':

    loader = SpiceLoader('spacecraft')

    name = 'state'#sys.argv[1]
    d    = pd.read_csv('output/{}.csv'.format(name))
    time = np.array(d['time'] - d['time'][0])

    plot_inrtl(time, d)
    plot_lvlh(time, d, 'moon')
    plot_lvlh(time, d, 'earth')
    plot_lvlh_covariance('f9', 'earth')

    plt.show()

    
