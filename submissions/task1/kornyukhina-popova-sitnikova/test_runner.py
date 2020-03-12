import unittest
from mypackage import *
from nose.tools import assert_equals

suite = unittest.TestSuite()
suite.addTest(unittest.makeSuite(test.TestNash))
runner = unittest.TextTestRunner()
testResult = runner.run(suite)
print(testResult)