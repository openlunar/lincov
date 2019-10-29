import numpy as np
from scipy.linalg import norm

SMALL = 1e-12

def gradient(r, mu):
    """Compute acceleration over position gradient (3x3 matrix) for a
    given position."""

    r1 = norm(r)
    return np.outer(r, r) * (3.0 * mu / r1**5) - np.identity(3) * (mu / r1**3) # CHECK ME


def point_gravity(r, mu):
    """Compute acceleration due to gravity given a point mass."""
    r2 = r.T.dot(r)
    r1 = np.sqrt(r2)
    r3 = r1 * r2
    r5 = r2 * r3

    if r2 < SMALL:
        raise ValueError("position for gravity calculation too close to origin")

    return r * -mu / r3

def j2_gravity(r_pcpf, mu, j2, r_eq):
    """Compute acceleration due to gravity given spherical harmonics."""

    z2 = r_pcpf[2]**2
    a  = point_gravity(r_pcpf, mu)
    r1 = norm(r_pcpf)

    k = 3 * mu * j2 * r_eq**2 / (2.0 * r1**5)
    l = 5.0 * z2 / r1**2

    a[0:2] += k * (l - 1.0) * r_pcpf[0:2]
    a[2]   += k * (l - 3.0) * r_pcpf[2]

    return a
