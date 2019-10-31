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

def plot_lvlh_covariance(name, count, body_id = 399, object_id = -5440):
    P, time = LinCov.load_covariance(name, count)

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

    if len(sys.argv) < 4:
        raise SyntaxError("expected run name, index number, body name")

    label = sys.argv[1]
    count = int(sys.argv[2])
    body  = sys.argv[3]
    
    loader = SpiceLoader('spacecraft')
    
    plot_lvlh_covariance(label, count, 'earth')

    plt.show()
