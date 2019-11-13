from spiceypy import spiceypy as spice

import numpy as np
from scipy.linalg import norm

import lincov.horizon as horizon


def sun_spacecraft_angle(body, time, object_id):
    if body == 'earth':
        frame = 'ITRF93'
    elif body == 'moon':
        frame = 'MOON_ME'
    sun,t1,t2 = spice.subslr('INTERCEPT/ELLIPSOID', body, time, frame, 'NONE', str(object_id))
    sc, t1,t2  = spice.subpnt('INTERCEPT/ELLIPSOID', body, time, frame, 'NONE', str(object_id))
    sun /= norm(sun)
    sc /= norm(sc)
    return np.arccos(np.dot(sun, sc))


class State(object):
    """State information for the linear covariance analysis"""

    ground_stations = { 'DSS-24': 399024,
                        'DSS-34': 399034,
                        'DSS-54': 399054 }

    r_station_ecef = {}
    
    def __init__(self, time, loader = None, params = None):
        self.loader = loader
        self.params = params
        self.mu_earth = loader.mu_earth * 1e9
        self.mu_moon  = loader.mu_moon * 1e9

        # Ensure that ground station locations are loaded
        if len(State.r_station_ecef) == 0:
            for station in State.ground_stations:
                gid = self.ground_stations[station]
                State.r_station_ecef[station] = spice.spkezr(station, self.loader.start, 'ITRF93', 'NONE', 'earth')[0][0:3] * 1000.0

        self.eci = spice.spkez(loader.object_id, time, 'J2000', 'NONE', 399)[0] * 1000.0
        self.lci = spice.spkez(loader.object_id, time, 'J2000', 'NONE', 301)[0] * 1000.0

        # FIXME: Need measurements here
        self.a_meas_inrtl = np.zeros(3)
        self.w_meas_inrtl = np.zeros(3)

        # Get distance to earth and moon
        self.d_earth = norm(self.eci[0:3])
        self.d_moon  = norm(self.lci[0:3])

        # Get angular size of each
        self.earth_angle = 2 * np.arctan(self.loader.r_earth[2] * 1000.0 / self.d_earth)
        self.moon_angle  = 2 * np.arctan(self.loader.r_moon[2] * 1000.0 / self.d_moon)
        
        self.earth_phase_angle = sun_spacecraft_angle('earth', time, loader.object_id)
        self.moon_phase_angle  = sun_spacecraft_angle('moon', time, loader.object_id)

        # We need to be able to clearly see the planet in order to do
        # horizon detection.
        planet_occult_code = spice.occult('earth', 'ellipsoid', 'ITRF93', 'moon', 'ellipsoid', 'MOON_ME', 'NONE', str(loader.object_id), time)

        self.horizon_moon_enabled = False
        self.horizon_earth_enabled = False
        
        if planet_occult_code == 0:
            if self.earth_angle < self.params.horizon_fov and self.earth_phase_angle < self.params.horizon_max_phase_angle:
                self.horizon_earth_enabled = True
            
            if self.moon_angle < self.params.horizon_fov and self.moon_phase_angle < self.params.horizon_max_phase_angle:
                self.horizon_moon_enabled = True
        else:
            self.earth_angle = 0.0
            self.moon_angle  = 0.0

        self.elevation_from = {}
        self.visible_from = []
        self.r_station_inrtl = {}
        for ground_name in self.ground_stations:
            obj_str = str(self.loader.object_id)
            moon_occult_code = spice.occult(obj_str, 'point', ' ', 'moon', 'ellipsoid', 'MOON_ME', 'NONE', str(self.ground_stations[ground_name]), time)

            elevation = float('nan')
            if moon_occult_code >= 0:
                # get spacecraft elevation
                x, lt = spice.spkcpo(obj_str, time, ground_name + '_TOPO', 'OBSERVER', 'NONE', self.r_station_ecef[ground_name] / 1000.0, 'earth', 'ITRF93')
                r, lon, lat = spice.reclat(x[0:3])

                if lat >= self.params.radiometric_min_elevation:
                    self.visible_from.append(ground_name)
                    elevation = lat
                    
            # store elevation of spacecraft for logging purposes
            self.elevation_from[ground_name] = elevation

        self.range_earth = norm(self.eci[0:3])
        self.range_moon  = norm(self.lci[0:3])

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
