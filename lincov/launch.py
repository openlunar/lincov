import numpy.random as npr
import numpy as np

from spiceypy import spiceypy as spice

def sample_f9_gto_covariance(x,
                             sigma_rp    = 7400,   # m
                             sigma_ra    = 130000, # m
                             sigma_inc   = 0.1 * np.pi/180,
                             sigma_lnode = 0.75 * np.pi/180,
                             sigma_argp  = 0.3 * np.pi/180,
                             sigma_att   = 3.0 * np.pi/180,
                             sigma_ba    = 0.005 * 9.81,
                             sigma_bg    = 0.1 * np.pi/180,
                             N           = 10000):
    """Generate a fake launch covariance so we have a starting point for
    our linear covariance analysis.
    
    Falcon 9 GTO: 185 x 35786 by 27 degrees
    
    Based on outdated information:
    
    * https://www.spaceflightnow.com/falcon9/001/f9guide.pdf
      (see p. 29)
    """

    mu    = x.loader.mu_earth
    state = x.eci / 1000.0

    elements = spice.oscelt(state, x.time, mu) # orbital elements from cartesian
    
    rp    = elements[0]
    ecc   = elements[1]
    ra    = rp * (1 + ecc) / (1 - ecc)
    inc   = elements[2]
    lnode = elements[3]
    argp  = elements[4]
    m0    = elements[5]

    sigma = np.array([sigma_rp / 1000.0, sigma_ra / 1000.0, sigma_inc, sigma_lnode, sigma_argp])
    mean  = np.array([rp, ra, inc, lnode, argp])
    els   = npr.randn(N, 5) * sigma + mean

    state_errors = []

    for ii in range(0, N):
        elements = np.hstack((els[ii,:], m0, x.time, mu))
        rp = elements[0]
        ra = elements[1]
        ecc = (ra - rp) / (ra + rp)

        random_state = spice.conics(elements, x.time)
        random_state[0] *= 1000.0
        random_state[1] *= 1000.0
        
        state_error = random_state - state
        state_errors.append( state_error )

    state_errors = np.vstack( state_errors )
    P_full = np.zeros((15,15))
    P_sampled = state_errors.T.dot(state_errors) / (N - 1)
    
    P_full[0:6,0:6] = P_sampled
    P_full[6:9,6:9] = np.diag(np.array([sigma_att, sigma_att, sigma_att])**2)
    P_full[9:15,9:15] = np.diag(np.array([sigma_ba, sigma_ba, sigma_ba,
                                          sigma_bg, sigma_bg, sigma_bg])**2)

    return P_full
    
