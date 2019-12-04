# lincov

Linear covariance analysis tool for lunar missions.

* [Open Lunar Foundation](https://www.openlunar.org/)

## Description

This tool helps in understanding how changing different sensor
parameters affects the spacecraft's knowledge about its
state. Eventually, it should also help in understanding other types of
dispersions.

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

## Usage

Most scripts are in the project root directory. Each analysis is
represented by a configuration YAML file in the `config/` directory.

## Developers

If you find a bug or wish to make a contribution, use the project's
[github issue tracker](https://github.com/openlunar/lincov/issues).

## License

Copyright (c) 2019--2020, Open Lunar Foundation.
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