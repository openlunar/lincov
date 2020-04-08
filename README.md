# lincov

Linear covariance analysis tool for understanding navigation uncertainty over the course of a mission.

* [Open Lunar Foundation](https://www.openlunar.org/)

## Description

This tool, which was written for cislunar mission design, aims to
assist in understanding how changing different sensor parameters
affects the spacecraft's knowledge about its state over the course of
a nominal trajectory. Eventually, it should also help in understanding
other types of dispersions.

Currently, it includes measurement models for:

* star tracker
* horizon recognition
* two-way radiometric ranging
* two-way radiometric range-rate (Doppler)

Note that the final two measurement models are typically used in a
ground-based filter rather than one onboard the spacecraft.

In the future, additional measurement models to be developed include:

* terrain
* altimeter
* others

## Requirements

* Python 3.x
* Numpy
* SciPy
* [SpiceyPy](https://github.com/AndrewAnnex/SpiceyPy)
* [ruamel.yaml](https://bitbucket.org/ruamel/yaml/src)
* Matplotlib

## Installation

Clone this repository:

    git clone https://github.com/openlunar/lincov.git

Currently, this tool cannot be installed as a package. You must run it
out of the repository directory.

## Detailed Usage

Most scripts are in the project root directory. Each analysis is
represented by a configuration YAML file in the `config/` directory,
of which there are several examples included.

To run an analysis, you first need to place a SPICE kernel describing
your reference trajectory in the `kernels/` directory. This is loaded
by the `SpiceLoader` class in `lincov/spice_loader.py`, which expects
it to be called `spacecraft.bsp` and to use -5440 as the integer
identifier for your spacecraft. The `SpiceLoader` class also takes
care of loading other kernels; a default set is located in the
`kernels/` directory, and the kernels to be loaded can be changed by
altering the constructor of `SpiceLoader`. Though it is untested, a
user could likely also perform analyses outside of the
earth&ndash;moon system by altering these kernels.

The analysis also requires as input an initial covariance and time,
which currently has as its default the documented dispersions of a
Falcon 9 launch to GTO. These files are in `output/f9/P.init.npy` and
`output/f9/time.init.npy`. The time should align to the beginning of
the reference trajectory. The covariance is always given in an
inertial frame.

### Running and resuming

A run may be initiated using

    ./start.py twoway_range f9 init

The first argument refers to the configuration file (in this case,
`twoway_range.yml`), and also sets the output directory in which to
place the time and covariance snapshots as it runs the analysis.

The second argument tells in which `output/` subdirectory to look for
the initial time and covariance; as mentioned, it defaults to
`f9`. The third argument indicates which snapshot index to load;
`init` is the default, but most snapshots are given numerical indices
(e.g.  `P.0825.npy` and its corresponding `time` file). Snapshot
intervals are controlled in the configuration file (the `block_dt`
parameter). If Python runs out of memory, try decreasing this time
step. If you have limited hard drive space but lots of RAM, you can
increase it. The snapshots exist for two purposes:

1. In the event of an error, one often wants to resume from an
   intermediate point, without redoing the entire analysis.
2. It might be desirable to run an analysis with one set of sensors,
   then pause it, and continue with a different set of sensors.

If a run is cancelled or halts (such as because of an error), it may
be resumed using

    ./resume.py <config-file>

in which case it should start from the highest-indexed snapshot in the
`output/<config-file>` directory.

    ./snapshot.py <config-file> <time> <label>

The snapshot script is similar to resume, except that it runs from the
snapshot index just prior to the given mission elapsed time (*not*
ephemeris time) until it gets to the desired time, and then spits out
a snapshot with the given label. You might use this tool to create a
snapshot containing dispersions at a midcourse correction, from which
you can begin other analyses (using the `start.py` script).

### Plotting

There are a variety of plotting tools included.

    ./plot.py <config> <item> [start-time [end-time]]

This tool plots a segment of the analysis, and defaults to plotting
the entire thing (which is extremely memory-consuming). The item is
one of `state_sigma` (for the state and covariance), `environment`
(for properties such as ground station elevations over time), and `R`
(for sensor measurement information).

    ./plot_sample.py <config> <item> [start-time [end-time]]

This tool has the same syntax as `plot.py`, but instead of loading the
entire time series from each set of outputs, it only loads the first
and last item in each index. This allows an entire analysis to be
plotted in a somewhat representative form without taking up enormous
amounts of memory.

    ./plot_cov.py <list-of-labels> <snapshot-label> <spice-body-id>

The list of labels (referring to the configuration files) should be
comma-separated without spaces. It allows you to compare the
covariance at a given time between different analyses. The
`snapshot-label` is a snapshot index or other identifier (e.g. `init`
or `320`) telling it which snapshot to load in the relevant
configuration directory. The <spice-body-id> controls which LVLH frame
is used for plotting (use 399 for earth and 301 for moon).

### Configuration files

The configuration files consist of, at minimum, a covariance
propagation time step (`dt`), a snapshot interval (`block_dt`),
measurement intervals for each measuremen type (the `meas_dt`
section), and general parameters (the `params` section).

General parameters of non-obvious nature are:

* **tau**: the time constants for the biases on the IMU (accelerometer
  and gyroscope), in seconds. These biases are
  exponentially-correlated random variables.

* **q_a_psd_imu**, **q_a_psd_dynamics**, **q_w_psd**: the process
  noise power spectral densities. The larger of the two acceleration
  process noises is generally used; the former is from the
  accelerometer specifications and the latter represents the dynamics
  uncertainty. The w in `q_w_psd` refers to omega; this represents the
  gyroscope process noise.

Look in the various measurement model `.py` files in `lincov/` to
understand the options on the different measurement types; these
options are usually prefixed by the measurement name.

## Unit tests

Some rather limited unit tests are in the `test/` directory and can be
run using

    python3 setup.py test

## Developers

If you find a bug or wish to make a contribution, use the project's
[github issue tracker](https://github.com/openlunar/lincov/issues).

## License

Copyright (c) 2019--2020, Open Lunar Foundation and John O. "Juno"
Woods, Ph.D.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the <organization> nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.