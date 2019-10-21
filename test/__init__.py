import unittest
from test import test_frames

def lincov_test_suite():
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromModule(test_frames))

    return suite
