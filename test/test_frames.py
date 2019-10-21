import unittest

from .context import *

class Test_Frames(unittest.TestCase):

    def test_compute_T_inrtl_to_lvlh(self):

        # Target state
        xt = np.array([-2301672.24489839, -5371076.10250925, -3421146.71530212,
                       6133.8624555516, 306.265184163608, -4597.13439017524])

        xc = np.array([-2255213.51862763, -5366553.94133467, -3453871.15040494,
                       6156.89588163809, 356.79933181917, -4565.88915429063])

        x_c_wrt_t = xc - xt
        Til = frames.compute_T_inrtl_to_lvlh(xt)
        xc_lvlh = Til.dot(x_c_wrt_t)

        np.testing.assert_allclose(xc_lvlh, np.array([56935.52933486611, 38.16029598938621, 2845.326754409645,
                                                      4.890395234717321, -0.09759947085768417, -0.8044815052666578]))
