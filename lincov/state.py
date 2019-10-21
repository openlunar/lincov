from spiceypy import spiceypy as spice

import numpy as np
from scipy.linalg import norm

import lincov.horizon as horizon

class State(object):
    """State information for the linear covariance analysis"""

    def __init__(self, time, loader = None):
        self.loader = loader
        self.mu_earth = loader.mu_earth * 1e9
        self.mu_moon  = loader.mu_moon * 1e9

        self.eci = spice.spkez(loader.object_id, time, 'J2000', 'NONE', 399)[0] * 1000.0
        self.lci = spice.spkez(loader.object_id, time, 'J2000', 'NONE', 301)[0] * 1000.0

        # FIXME: Need measurements here
        self.a_meas_inrtl = np.zeros(3)
        self.w_meas_inrtl = np.zeros(3)

        # Get distance to earth and moon
        self.d_earth = norm(self.eci[0:3])
        self.d_moon  = norm(self.lci[0:3])

        # Get angular size of each
        self.theta_earth = 2 * np.arctan(self.loader.r_earth[2] / self.d_earth)
        self.theta_moon  = 2 * np.arctan(self.loader.r_moon[2]  / self.d_moon)

        if self.theta_earth < self.loader.fov:
            self.horizon_earth = True
        self.range_earth = norm(self.eci[0:3])
            
        if self.theta_moon < self.loader.fov:
            self.horizon_moon = True
        self.range_moon = norm(self.lci[0:3])

        self.time = time

    @property
    def object_id(self):
        return self.loader.object_id

    @property
    def T_body_to_att(self):
        return self.loader.T_body_to_att

    def range(self, rel):
        if rel == 'earth':
            return self.range_earth
        elif rel == 'moon':
            return self.range_moon

    def radii(self, rel):
        if rel == 'earth': return self.loader.r_earth * 1000.0
        elif rel == 'moon': return self.loader.self.r_moon * 1000.0

    def T_pa_to_cam(self, rel):
        """Transformation from cone principal axis frame to opnav camera
        frame, where rel tells which planet cone is oriented
        towards.

        """
        if rel in ('earth', 399):
            body_id = 399
        elif rel in ('moon', 301):
            body_id = 301
        else:
            raise NotImplemented("expected 'moon' or 'earth' based opnav, got '{}'".format(rel))
        return horizon.compute_T_pa_to_cam(self.eci, self.time, body_id)

    @property
    def T_epa_to_cam(self):
        return self.T_pa_to_cam('earth')

    @property
    def T_mpa_to_cam(self):
        return self.T_pa_to_cam('moon')
