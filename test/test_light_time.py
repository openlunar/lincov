import unittest

from .context import *

from lincov.light_time import light_time, send_time, receive_time, C

class Test_LightTime(unittest.TestCase):


    def setUp(self):
        self.loader = SpiceLoader('spacecraft')
    
    def test_send_and_receive_time(self):
        t2         = self.loader.start + 1000.0
        station_id = 399024
        sc_id      = self.loader.object_id

        # Want to find time signal leaves station (t1) to get to sc_id
        # at t2.
        t1 = send_time(station_id, sc_id, t2)
        t3 = receive_time(station_id, sc_id, t2)

        # Compare our results to the SPICE method's results
        spice_t1 = spice.ltime(t2, sc_id, '<-', station_id)[0]
        spice_t3 = spice.ltime(t2, sc_id, '->', station_id)[0]

        np.testing.assert_allclose([t1, t3], [spice_t1, spice_t3])

        # Make sure it still works if we give a state instead
        x2 = spice.spkez(sc_id, t2, 'J2000', 'NONE', 399)[0]
        t1b = send_time(station_id, x2, t2)
        t3b = receive_time(station_id, x2, t2)
        
        np.testing.assert_equal([t1, t3], [t1b, t3b])
        
