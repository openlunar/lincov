#!/usr/local/bin/python3

import pandas as pd
import numpy as np
from scipy.linalg import norm

import sys

import matplotlib.pyplot as plt

def apply_styles(axes, time):
    for ax in axes:
        ax.set_ylim(bottom=0.0)
        ax.set_xlim(time[0], time[-1])
        ax.grid(True)


def plot_inrtl(time, d):
    fig, axes = plt.subplots(5,1,sharex=True)

    fig.suptitle("inertial 1-sigma covariance")

    axes[0].plot(time, d['srx'] / 1000.0, label='rx')
    axes[0].plot(time, d['sry'] / 1000.0, label='ry')
    axes[0].plot(time, d['srz'] / 1000.0, label='rz')
    axes[0].set_ylabel('km')

    axes[1].plot(time, d['svx'] / 1000.0, label='vx')
    axes[1].plot(time, d['svy'] / 1000.0, label='vy')
    axes[1].plot(time, d['svz'] / 1000.0, label='vz')
    axes[1].set_ylabel("km/s")

    axes[2].plot(time, d['sattx'] * 180/np.pi, label='about x')
    axes[2].plot(time, d['satty'] * 180/np.pi, label='about y')
    axes[2].plot(time, d['sattz'] * 180/np.pi, label='about z')
    axes[2].set_ylabel("degrees")

    axes[3].plot(time, d['sbax'])
    axes[3].plot(time, d['sbay'])
    axes[3].plot(time, d['sbaz'])
    axes[3].set_ylabel("m/s2")

    axes[4].plot(time, d['sbgx'] * 180/np.pi)
    axes[4].plot(time, d['sbgy'] * 180/np.pi)
    axes[4].plot(time, d['sbgz'] * 180/np.pi)
    axes[4].set_ylabel("deg/s")
    axes[4].set_xlabel("mission elapsed time (s)")
    
    apply_styles(axes, time)

    return fig, axes

def plot_other_frame(time, d, frame = 'llvlh'):
    fig, axes = plt.subplots(2,1,sharex=True)

    fig.suptitle("{} 1-sigma covariance".format(frame))

    if frame in ('llvlh', 'elvlh'):
        labels=('v-bar', 'h-bar', 'r-bar')
    
    axes[0].plot(time, d[frame+'_srx'] / 1000.0, label=labels[0])
    axes[0].plot(time, d[frame+'_sry'] / 1000.0, label=labels[1])
    axes[0].plot(time, d[frame+'_srz'] / 1000.0, label=labels[2])
    axes[0].legend()
    axes[0].set_ylabel('km')

    axes[1].plot(time, d[frame+'_svx'] / 1000.0, label=labels[0])
    axes[1].plot(time, d[frame+'_svy'] / 1000.0, label=labels[1])
    axes[1].plot(time, d[frame+'_svz'] / 1000.0, label=labels[2])
    axes[1].set_ylabel("km/s")

    axes[1].set_xlabel("mission elapsed time (s)")

    apply_styles(axes, time)

    return fig, axes

if __name__ == '__main__':

    name = 'state'#sys.argv[1]
    d    = pd.read_csv('output/{}.csv'.format(name))
    time = np.array(d['time'] - d['time'][0])

    plot_inrtl(time, d)
    plot_other_frame(time, d, 'llvlh')
    plot_other_frame(time, d, 'elvlh')
    
    
    plt.show()
