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

    # FIXME: Eventually make this have something to do with how largez
    # the planet is in the FOV.
    max_sun_planet_angle = 150.0 * np.pi/180.0
    M2 = 240/221.0 # https://descanso.jpl.nasa.gov/monograph/series2/Descanso2_S13.pdf p. 13-13
    C3 = 96 * M2

    ground_stations = { 'DSS-24': 399024,
                        'DSS-34': 399034,
                        'DSS-54': 399054 }

    r_station_ecef = {}

    min_elevation = 5 * np.pi/180.0 # spacecraft must be above horizon
                                    # by this much to be visible to
                                    # ground tracking
    signal_duration = 1.02070501723677 # seconds
    c = 299792458.0 # m/s
    f_T = 2.1102e9 # frequency of transmission (Hz)
    twoway_doppler_sigma = 0.001 * C3 * f_T / c # meters/s / m/s
    twoway_range_sigma   = 2.0 / c # meters / m/s
    
    def __init__(self, time, loader = None):
        self.loader = loader
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
        self.theta_earth = 2 * np.arctan(self.loader.r_earth[2] * 1000.0 / self.d_earth)
        self.theta_moon  = 2 * np.arctan(self.loader.r_moon[2] * 1000.0 / self.d_moon)
        
        self.angle_sun_earth = sun_spacecraft_angle('earth', time, loader.object_id)
        self.angle_sun_moon  = sun_spacecraft_angle('moon', time, loader.object_id)

        # We need to be able to clearly see the planet in order to do
        # horizon detection.
        planet_occult_code = spice.occult('earth', 'ellipsoid', 'ITRF93', 'moon', 'ellipsoid', 'MOON_ME', 'NONE', str(loader.object_id), time)

        self.horizon_moon_enabled = False
        self.horizon_earth_enabled = False
        
        if planet_occult_code == 0:
            if self.theta_earth < self.loader.fov and self.angle_sun_earth < self.max_sun_planet_angle:
                self.horizon_earth_enabled = True
            
            if self.theta_moon < self.loader.fov and self.angle_sun_moon < self.max_sun_planet_angle:
                self.horizon_moon_enabled = True

        self.elevation_from = {}
        self.visible_from = []
        self.r_station_inrtl = {}
        for ground_name in self.ground_stations:
            obj_str = str(self.loader.object_id)
            moon_occult_code = spice.occult(obj_str, 'point', ' ', 'moon', 'ellipsoid', 'MOON_ME', 'NONE', str(self.ground_stations[ground_name]), time)
            if moon_occult_code >= 0: # moon not blocking spacecraft
                pass
            
            # get spacecraft elevation
            x, lt = spice.spkcpo(obj_str, time, ground_name + '_TOPO', 'OBSERVER', 'NONE', self.r_station_ecef[ground_name] / 1000.0, 'earth', 'ITRF93')
            r, lon, lat = spice.reclat(x[0:3])
            if lat >= self.min_elevation:
                self.visible_from.append(ground_name)
            self.elevation_from[ground_name] = lat # store elevation of spacecraft for logging purposes
               

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
