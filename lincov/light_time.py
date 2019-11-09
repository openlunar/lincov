from spiceypy import spiceypy as spice

import numpy as np
from scipy.linalg import norm

C = 299792.458 # km/s

def light_time(x1, x2, gamma = 1.0, mu = 3.9860043543609598e5):
    r2    = norm(x2[0:3])
    r1    = norm(x1[0:3])
    r12   = norm(x2[0:3] - x1[0:3])
    tau12 = r12 / C + (1.0 + gamma) * (mu / C**3) * np.log((r1 + r2 + r12) / (r1 + r2 - r12))
    return tau12


def send_time(station_id, sc_id, sc_time, max_error = 1e-7, **kwargs):
    t2 = sc_time
    if type(sc_id) == int:
        x2 = spice.spkez(sc_id, t2, 'J2000', 'NONE', 399)[0]
    else:
        x2 = sc_id
    x1 = spice.spkez(station_id, t2, 'J2000', 'NONE', 399)[0]
    t1 = t2 - light_time(x1, x2, **kwargs)
    dt = max_error * 2
    
    while np.abs(dt) > max_error:
        x1 = spice.spkez(station_id, t1, 'J2000', 'NONE', 399)[0]
        df_dt = -1 + np.dot((x2[0:3] - x1[0:3]) / norm(x2[0:3] - x1[0:3]), x1[3:6])
        f = t2 - t1 - light_time(x1, x2, **kwargs)
        dt = -f / df_dt
        t1 += dt

    return t1

def receive_time(station_id, sc_id, sc_time, max_error = 1e-7, **kwargs):
    t2 = sc_time
    if type(sc_id) == int:
        x2 = spice.spkez(sc_id, t2, 'J2000', 'NONE', 399)[0]
    else:
        x2 = sc_id
    x3 = spice.spkez(station_id, t2, 'J2000', 'NONE', 399)[0]
    t3 = t2 + light_time(x2, x3, **kwargs)
    dt = max_error * 2

    while np.abs(dt) > max_error:
        x3 = spice.spkez(station_id, t3, 'J2000', 'NONE', 399)[0]
        df_dt = -1 + np.dot((x3[0:3] - x2[0:3]) / norm(x3[0:3] - x2[0:3]), x3[3:6])
        f = t3 - t2 - light_time(x2, x3, **kwargs)
        dt = -f / df_dt
        t3 += dt

    return t3
    

def ltime(t1, id1, dir, x2, gamma = 1.0, mu = 3.9860043543609598e5):
    x1 = spice.spkez(id1, t1, 'J2000', 'NONE', 399)

    r2    = norm(x2[0:3])
    if dir == '->':
        r1    = norm(x1[0:3])
        r12   = norm(x2[0:3] - x1[0:3])
        tau12 = r12 / C + (1.0 + gamma) * (mu / C**3) * np.log((r1 + r2 + r12) / (r1 + r2 - r12))
        return t1 + tau12, tau12

    elif dir == '<-':
        t3    = t1
        r3    = norm(x1[0:3])
        r23   = norm(x2[0:3] - x1[0:3])
        tau23 = r23 / C + (1.0 + gamma) * (mu / C**3) * np.log((r2 + r3 + r23) / (r2 + r3 - r23))
        return t3 - tau23, tau23

    else:
        raise ValueError("expected '<-' or '->' for dir")

