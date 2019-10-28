'''
'''
import unittest

from tests.test_counting import VVCTestCase
from tests.test_others import OtherTestCase
from tests.test_mot_metrics import MOTMetricsTestCase
    
def suite():
    suite = unittest.TestSuite()
    suite.addTest(MOTMetricsTestCase('test_motchallenge_files'))
    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
