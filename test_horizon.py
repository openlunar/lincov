#!/usr/bin/python3

import numpy as np
from scipy.linalg import norm

from spiceypy import spiceypy as spice
from lincov import SpiceLoader
from lincov.horizon import covariance

import matplotlib.pyplot as plt


if __name__ == '__main__':

    body_id = 301
    sc_id   = -5440
    loader = SpiceLoader('spacecraft')
    start, end = loader.coverage(sc_id)

    distances = []
    rx = []
    ry = []
    rz = []
    sun = []
    r_pix = []
    
    for tt in np.arange(start, end, 100.0):
        R, d = covariance(tt, 301, fov = 18 * np.pi/180.0, fpa_size = 2048, spacecraft_id = sc_id, statistics = True)
        distances.append(d['rho'])
        rx.append(R[0,0])
        ry.append(R[1,1])
        rz.append(R[2,2])
        sun.append(d['sun_angle'])
        r_pix.append(d['r_pix'])

    fig, axes = plt.subplots(4,1, squeeze=True, sharex=True)

    axes[0].scatter(distances, np.sqrt(np.array(rx)), s=1)
    axes[1].scatter(distances, np.sqrt(np.array(ry)), s=1)
    axes[2].scatter(distances, np.sqrt(np.array(rz)), s=1)
    #axes[3].scatter(distances, np.array(sun) * 180/np.pi, s=1)
    axes[3].scatter(distances, r_pix, s=1)

    for ii in range(0,3):
        axes[ii].grid(True)
        #axes[ii].set_yscale('log')
    axes[0].set_ylabel('x sigma (km)')
    axes[1].set_ylabel('y sigma (km)')
    axes[2].set_ylabel('z sigma (km)')
    #axes[3].set_ylabel('sun angle (deg)')
    axes[3].set_ylabel('pixels taken up')
    axes[3].grid(True)
    axes[3].set_xlabel("distance from moon (km)")

    plt.show()
