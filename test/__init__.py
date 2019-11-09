import unittest
from test import test_frames, test_light_time

def lincov_test_suite():
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromModule(test_frames))
    suite.addTests(loader.loadTestsFromModule(test_light_time))

    return suite
