'''
'''
import unittest

from tests.test_counting import VVCTestCase
from tests.test_others import OtherTestCase
    
def suite():
    suite = unittest.TestSuite()
    suite.addTest(VVCTestCase('test_yolo_naive'))
    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
