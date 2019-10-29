import numpy as np
from scipy.linalg import norm, inv

from lincov.spice_loader import SpiceLoader
from lincov.state import State
import lincov.horizon as horizon
from lincov.launch import sample_f9_gto_covariance
from lincov.frames import compute_T_inrtl_to_lvlh
from lincov.gravity import gradient as G

import pyquat as pq
import pandas as pd

import time
import math

def progress_bar(bar_length, completed, total):
    # https://stackoverflow.com/a/50108192/170300
    bar_length_unit_value = (total / bar_length)
    completed_bar_part = math.ceil(completed / bar_length_unit_value)
    progress = "*" * completed_bar_part
    remaining = " " * (bar_length - completed_bar_part)
    percent_done = "%.2f" % ((completed / total) * 100)
    print(f'[{progress}{remaining}] {percent_done}%', end='\r')


class LinCov(object):
    N = 15
    max_duration = 600.0

    order = ('att', 'horizon')
    
    # process noise stuff
    tau   = np.array([600.0, 600, 600, 600, 600, 600]) # FIXME: bias time constants
    beta  = 1 / tau
    sigma = np.array([0.005 * 9.81,    # accelerometer bias sigma
                      0.005 * 9.81,
                      0.005 * 9.81,
                      0.1 * np.pi/180, # gyro bias siga
                      0.1 * np.pi/180,
                      0.1 * np.pi/180])**2

    # CHECK ME
    q_a_psd_coast = 1e-7  # unmodeled forces PSD
    # Reference suggesting above is reasonable is:
    #
    # * Liounis, Daniel, Christian (2013). Autonomous navigation
    #   system performance in the Earth--Moon system. In SPACE
    #   Conferences and Exposition.
    #
    # However, I get somewhat different values when not yet in lunar
    # orbit. This may require further examination.
    
    q_a_psd_accelerometer = 0.025**2 / 3600.0 # accelerometer noise PSD
    
    q_a_psd = max(q_a_psd_accelerometer, q_a_psd_coast) # accelerometer noise may dominate -- PSD
    q_w_psd = (0.09 * np.pi/180.0)**2 / 3600.0 # angular velocity (gyroscope noise) PSD
    

    def process_noise(self, dt):
        q_acc = self.q_a_psd * dt
        q_vel = q_acc * dt * 0.5
        q_pos = q_vel * dt * 2/3.0
        q_w   = self.q_w_psd * dt

        return np.diag([q_pos, q_pos, q_pos,
                        q_vel, q_vel, q_vel,
                        q_w,   q_w,   q_w,
                        0.0,   0.0,   0.0,
                        0.0,   0.0,   0.0])

    def state_transition(self, x):
        T_body_to_inrtl = np.identity(3)
        
        F = np.zeros((self.N, self.N))
        F[0:3,3:6] = np.identity(3)
        F[3:6,0:3] = G(x.eci[0:3], x.mu_earth) + G(x.lci[0:3], x.mu_moon)
        F[3:6,6:9] = -pq.skew(x.a_meas_inrtl).dot(T_body_to_inrtl)
        F[3:6,9:12] = -T_body_to_inrtl

        F[6:9,6:9] = -pq.skew(x.w_meas_inrtl)
        F[6:9,12:15] = -np.identity(3)

        Phi = np.identity(self.N) + F * self.dt
        Phi[9:self.N,9:self.N] = np.diag(-self.beta * self.dt)

        return Phi

    def att_update(self, x, P, plot=False):
        # measurement covariance:
        # FIXME: Needs to be adjusted for star tracker choice
        R = np.diag((np.array([5.0, 5.0, 70.0]) * np.pi/(180*3600))**2) # arcseconds to radians

        if plot:
            from plot_lincov import plot_covariance
            import matplotlib.pyplot as plt
            plot_covariance(R, xlabel='x (rad)', ylabel='y (rad)', zlabel='z (rad')
            plt.show()
        
        # measurement linearization / measurement sensitivity matrix (a Jacobian)
        H = np.zeros((3, self.N))
        H[0:3,6:9] = x.T_body_to_att
        
        return H, R

    def horizon_update(self, x, P, rel, plot=False):
        if rel in ('earth', 399):
            body_id = 399
        elif rel in ('moon', 301):
            body_id = 301
            
        R = horizon.covariance(x.time, body_id)

        if plot:
            from plot_lincov import plot_covariance
            import matplotlib.pyplot as plt
            T_lvlh = frames.compute_T_inrtl_to_lvlh(x.lci)[0:3,0:3]
            plot_covariance(T_lvlh.dot(R).dot(T_lvlh.T), xlabel='downtrack (m)', ylabel='crosstrack (m)', zlabel='radial (m)')
            plt.show()
        
        H = np.zeros((3, self.N))
        H[0:3,0:3] = np.identity(3)

        return H, R

    def radial_test_update(self, x, P):
        R = np.array([[2.0 * norm(x.eci[0:3]) * 0.001]])**2
        H = np.zeros((1, self.N))
        H[0,0:3] = x.eci[0:3] / norm(x.eci[0:3])

        return H, R
        

    def update(self, meas_type, time, x, P):
        """Attempt to process a measurement update for some measurement type"""
        updated = False
        if time > self.meas_last[meas_type] + self.meas_dt[meas_type]:
            
            if meas_type == 'att':
                H, R = self.att_update(x, P)
                updated = True
            elif meas_type == 'horizon':
                if x.horizon_moon:
                    H, R = self.horizon_update(x, P, 'moon')
                    updated = True
            elif meas_type == 'radial_test':
                H, R = self.radial_test_update(x, P)
                updated = True
            else:
                raise NotImplemented("unrecognized update type '{}'".format(meas_type))
            self.meas_last[meas_type] = time

            PHt = P.dot(H.T)

            if len(H.shape) == 1: # perform a scalar update
                W = H.dot(PHt) + R
                K = PHt / W[0,0]

                # Scalar joseph update
                P_post = P - K.dot(H.dot(P)) - PHt.dot(K.T) + (K*W).dot(K.T)

                import pdb
                pdb.set_trace()
            else: # perform a vector update
                K = PHt.dot(inv(H.dot(PHt) + R))

                # Vector Joseph update
                I_minus_KH = np.identity(K.shape[0]) - K.dot(H)
                P_post     = I_minus_KH.dot(P).dot(I_minus_KH.T) + K.dot(R).dot(K.T)
        else:
            P_post = P
            
        return P_post, updated

    def propagate(self, x, P, Q):
        """Propagate covariance matrix forward in time by dt"""
        Phi = self.state_transition(x)
        P = Phi.dot(P.dot(Phi.T)) + Q
        return P

    def save(self, name, time, cols):
        d = {'time': time}
        for key in cols:
            if len(cols[key].shape) == 1:
                d[key] = cols[key]
            elif cols[key].shape[1] == 3:
                d[key + 'x'] = cols[key][:,0]
                d[key + 'y'] = cols[key][:,1]
                d[key + 'z'] = cols[key][:,2]
            elif cols[key].shape[1] == 4:
                d[key + 'w'] = cols[key][:,0]
                d[key + 'x'] = cols[key][:,1]
                d[key + 'y'] = cols[key][:,2]
                d[key + 'z'] = cols[key][:,3]
                
        frame = pd.DataFrame(d)
        return frame.to_csv("output/{}.csv".format(name))


    def save_covariance(self, name, P, time):
        with open("output/{}.P.npy".format(name), 'wb') as P_file:
            np.save(P_file, P)
        with open("output/{}.time.npy".format(name), 'wb') as time_file:
            np.save(time_file, time)

    @classmethod
    def load_covariance(self, name):
        with open("output/{}.P.npy".format(name), 'rb') as P_file:
            P = np.load(P_file)
        with open("output/{}.time.npy".format(name), 'rb') as time_file:
            time = np.load(time_file)
        return P, time

    def init_covariance(self, loader, time):
        x = State(self.start, loader = loader)
        if time > self.start:
            raise NotImplemented("cannot initialize covariance from the future")
        try:
            P, cov_time = self.load_covariance("f9")
        except IOError:
            P = sample_f9_gto_covariance(x)
            self.save_covariance("f9", P, self.start)
        return P, time
        
    
    def __init__(self, time, dt, meas_dt, meas_last,
                 end_time = None,
                 loader   = None,
                 P        = None):
        self.start   = time
        if end_time is None:
            end_time = loader.end
            
        if end_time > self.start + self.max_duration:
            self.end = self.start + self.max_duration
        else:
            self.end = end_time
        
        self.dt        = dt
        self.meas_dt   = meas_dt
        self.meas_last = meas_last

        times  = []
        
        sr       = [] # position sigma
        sv       = [] # velocity sigma
        satt     = []
        sba      = []
        sbg      = []
        elvlh_sr = []
        elvlh_sv = []
        llvlh_sr = []
        llvlh_sv = []

        # Generate process noise matrix
        Q = self.process_noise(dt)

        # Generate launch covariance if none was given
        if P is None:
            x = State(time, loader = loader)
            P, time = self.init_covariance(loader, time)            

        # Loop until completion
        while time < self.end:
            progress_bar(60, time - loader.start, loader.end - loader.start)
            x = State(time, loader = loader)
            P = self.propagate(x, P, Q)

            # These don't change pre- and post-update in a LinCov (but
            # they do in a Kalman filter)
            T_inrtl_to_elvlh = compute_T_inrtl_to_lvlh(x.eci)
            T_inrtl_to_llvlh = compute_T_inrtl_to_lvlh(x.lci)
            
            # Pre-update logging
            times.append( time )
            sr.append(np.sqrt(np.diag(P[0:3,0:3])))
            sv.append(np.sqrt(np.diag(P[3:6,3:6])))
            satt.append(np.sqrt(np.diag(P[6:9,6:9])))
            sba.append(np.sqrt(np.diag(P[9:12,9:12])))
            sbg.append(np.sqrt(np.diag(P[12:15,12:15])))

            P_elvlh = T_inrtl_to_elvlh.dot(P[0:6,0:6]).dot(T_inrtl_to_elvlh.T)
            P_llvlh = T_inrtl_to_llvlh.dot(P[0:6,0:6]).dot(T_inrtl_to_llvlh.T)
            
            elvlh_sr.append(np.sqrt(np.diag(P_elvlh[0:3,0:3])))
            elvlh_sv.append(np.sqrt(np.diag(P_elvlh[3:6,3:6])))

            llvlh_sr.append(np.sqrt(np.diag(P_llvlh[0:3,0:3])))
            llvlh_sv.append(np.sqrt(np.diag(P_llvlh[3:6,3:6])))

            updated = False
            for meas_type in self.order:
                if time >= self.meas_last[meas_type] + self.meas_dt[meas_type]:
                    #print("{}: updating {}".format(time, meas_type))
                    P, updated = self.update(meas_type, time, x, P)



            # Post-update logging
            times.append(time)
            sr.append(np.sqrt(np.diag(P[0:3,0:3])))
            sv.append(np.sqrt(np.diag(P[3:6,3:6])))
            satt.append(np.sqrt(np.diag(P[6:9,6:9])))
            sba.append(np.sqrt(np.diag(P[9:12,9:12])))
            sbg.append(np.sqrt(np.diag(P[12:15,12:15])))

            P_elvlh = T_inrtl_to_elvlh.dot(P[0:6,0:6]).dot(T_inrtl_to_elvlh.T)
            P_llvlh = T_inrtl_to_llvlh.dot(P[0:6,0:6]).dot(T_inrtl_to_llvlh.T)
            elvlh_sr.append(np.sqrt(np.diag(P_elvlh[0:3,0:3])))
            elvlh_sv.append(np.sqrt(np.diag(P_elvlh[3:6,3:6])))
            
            llvlh_sr.append(np.sqrt(np.diag(P_llvlh[0:3,0:3])))
            llvlh_sv.append(np.sqrt(np.diag(P_llvlh[3:6,3:6])))
            
            
            time += self.dt

        if time > self.end: # Make sure we don't go past the end of the SPICE kernel
            time = self.end

        times = np.hstack(times)
        #xs_eci = np.vstack(xs_eci)
        #xs_lci = np.vstack(xs_lci)

        self.save('state', times, {
            'sr': np.vstack(sr),
            'sv': np.vstack(sv),
            'satt': np.vstack(satt),
            'sba': np.vstack(sba),
            'sbg': np.vstack(sbg),
            'elvlh_sr': np.vstack(elvlh_sr),
            'elvlh_sv': np.vstack(elvlh_sv),
            'llvlh_sr': np.vstack(llvlh_sr),
            'llvlh_sv': np.vstack(llvlh_sv)
        })

        self.save_covariance("test", P, time)

