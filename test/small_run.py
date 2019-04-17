'''
'''
import unittest

from test.test_counting import CountingTestCase
    
def suite():
    suite = unittest.TestSuite()
    suite.addTest(CountingTestCase('test_yolo_naive'))
    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
