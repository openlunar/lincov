"""

References:

* Hikes, Liounis, Christian (2017). Parametric covariance model for
  horizon-based optical navigation. Journal of Guidance, Control, and
  Dynamics 40(1): 169-177.

"""

import numpy as np
from scipy.linalg import norm

from spiceypy import spiceypy as spice


def compute_T_pa_to_cam(time, body_id = 301, spacecraft_id = -5440):
    """Compute a principal axis frame (see equations 63-66).

    Args:
      time           ephemeris time (s)
      body_id        NAIF body identifier (defaults to 301, moon)
      spacecraft_id  NAIF ID of spacecraft

    Returns:
      A 3x3 orthonormal rotation matrix.
    """

    r_sc_wrt_body = spice.spkezp(spacecraft_id, time, 'J2000', 'NONE', body_id)[0]
    r_sc_wrt_sun  = spice.spkezp(spacecraft_id, time, 'J2000', 'NONE', 10)[0]
    e_sun = r_sc_wrt_sun
    e_sun /= norm(e_sun)
    
    e_z = -r_sc_wrt_body / norm(r_sc_wrt_body)
    e_y = np.cross(e_z, e_sun)
    e_y /= norm(e_y)

    e_x = np.cross(e_y, e_z)
    e_x /= norm(e_x)

    return np.vstack((e_x, e_y, e_z))
    
    

def covariance(time,
               body_id,
               fpa_size      = 4096,
               fov           = 30.0 * np.pi/180.0,
               theta_max     = 70 * np.pi/180.0,
               sigma_pix     = 0.063,
               n_max         = 1000,
               spacecraft_id = -5440,
               statistics    = False):
    """Compute parameterized covariance model for horizon recognition.

    Args:
      time           ephemeris time (s)
      body_id        NAIF id for planet we're using for navigation
      fpa_size       focal plane array size (pixels)
      fov            field of view of camera
      theta_max      half-angle of horizon which is visible (where max is pi/2)
      sigma_pix      horizon-finding method's subpixel accuracy
      n_max          maximum number of fit points to use for the horizon
      spacecraft_id  NAIF id for the spacecraft doing the navigating
      statistics     if True, return information used in model

    Returns:
      A 3x3 covariance matrix (for camera frame, with z-axis toward
    the body).

    """
    r_p = spice.bodvcd(body_id, 'RADII', 3)[1][2]
    d_x = 1 / (2.0 * np.tan(fov / 2))
    r_sc_wrt_body = spice.spkezp(spacecraft_id, time, 'J2000', 'NONE', body_id)[0]
    rho = norm(r_sc_wrt_body)
    r_pix = r_p * fpa_size / (2 * rho * np.tan(fov / 2))

    # FIXME: find theta_max instead of specifying it
    
    # Don't allow n to exceed our processing capabilities, if applicable
    n = 2 * theta_max * r_pix
    if n > n_max:
        n = n_max

    D = (theta_max / 4) * (2 * theta_max + np.sin(2*theta_max)) - np.sin(theta_max)**2

    T_p_to_c = compute_T_pa_to_cam(time, body_id, spacecraft_id)

    corner = np.sqrt(rho**2 - r_p**2) * np.sin(theta_max) / (D * r_p)
    M = np.array([[theta_max / D, 0.0, corner],
                  [0.0, 4 / (2 * theta_max - np.sin(2*theta_max)), 0.0],
                  [corner, 0.0, (rho**2 - r_p**2) * (2 * theta_max + np.sin(2*theta_max)) / (4 * D * r_p**2)]])
    scalar = sigma_pix**2 * rho**4 * theta_max / (n * d_x**2 * (rho**2 - r_p**2))
    P = T_p_to_c.T.dot(M).dot(T_p_to_c) * scalar

    if statistics:
        r_sun_wrt_body = spice.spkezp(body_id, time, 'J2000', 'LT', 10)[0]
        e_sun = r_sun_wrt_body / norm(r_sun_wrt_body)
        e_cam = r_sc_wrt_body / norm(r_sc_wrt_body)
        
        data = {
            'rho':       rho,
            'sun_angle': np.arccos(np.dot(e_sun, e_cam)),
            'r_pix':     r_pix,
            'n':         n
        }

        return P, data

    return P


