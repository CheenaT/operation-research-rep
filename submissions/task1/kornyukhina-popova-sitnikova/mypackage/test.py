import numpy as np
from fractions import Fraction
from nose.tools import assert_equals
import unittest
from .nash import nash_equilibrium
from scipy.optimize import linprog

class TestNash(unittest.TestCase):
    #def setUp(self):
    #    print ("setup() before each methods in this class")
    #def tearDown(self):
    #    print ("teardown() after each methods in this class")
    def test_1(self):
        a = np.array(([4,0,6,2,2,1],[3,8,4,10,4,4],[1,2,6,5,0,0],[6,6,4,4,10,3],[10,4,6,4,0,9],[10,7,0,7,9,8]))
        func_p,func_q,func_price = nash_equilibrium(a)
        expected_price = 151/31
        expected_p = np.array([0 , 4/31 , 3/31 , 27/62 , 21/62 , 0])
        expected_q = np.array([0, 0, 257/372, 9/62, 55/372, 1/62])
        expected_res = np.array([expected_price, expected_p, expected_q])
        self.assertAlmostEqual(func_price, expected_price)
        self.assertAlmostEqual(func_p.all(), expected_p.all())
        self.assertAlmostEqual(func_q.all(), expected_q.all())
    def test_2(self):
        a = np.array(([-1,1],[1,-1]))
        func_p,func_q,func_price = nash_equilibrium(a)
        expected_price = 0
        expected_p = np.array([1/2,1/2])
        expected_q = np.array([1/2,1/2])
        expected_res = np.array([expected_price, expected_p, expected_q])
        self.assertAlmostEqual(func_price, expected_price)
        self.assertAlmostEqual(func_p.all(), expected_p.all())
        self.assertAlmostEqual(func_q.all(), expected_q.all())
    def test_3(self):
        a = np.array(([3,1],[1,1]))
        func_p,func_q,func_price = nash_equilibrium(a)
        expected_price = 1
        expected_p = np.array([1,0])
        expected_q = np.array([0,1])
        self.assertAlmostEqual(func_price, expected_price)
        self.assertAlmostEqual(func_p.all(), expected_p.all())
        self.assertAlmostEqual(func_q.all(), expected_q.all())
    def test_4(self):
        a = np.array(([5,-1],[2,3]))
        func_p,func_q,func_price = nash_equilibrium(a)
        expected_price = 17/7
        expected_p = np.array([1/7, 6/7])
        expected_q = np.array([4/7, 3/7])
        self.assertAlmostEqual(func_price, expected_price)
        self.assertAlmostEqual(func_p.all(), expected_p.all())
        self.assertAlmostEqual(func_q.all(), expected_q.all())
    def test_5(self):
        a = np.array(([5,6,7],[3,3,3],[2,1,0]))
        func_p,func_q,func_price = nash_equilibrium(a)
        expected_price = 5
        expected_p = np.array([1,1,0])
        expected_q = np.array([1,1,0])
        self.assertAlmostEqual(func_price, expected_price)
        self.assertAlmostEqual(func_p.all(), expected_p.all())
        self.assertAlmostEqual(func_q.all(), expected_q.all())
    def test_6(self):
        a = np.array(([51234,2134,71234],[2243,11243,1231],[32,21,1531],[432,21,124]))
        func_p,func_q,func_price = nash_equilibrium(a)
        expected_price = 5712373/581
        expected_p = np.array([90/581,491/581,0,0])
        expected_q = np.array([9109/58100,48991/58100,0])
        self.assertAlmostEqual(func_price, expected_price)
        self.assertAlmostEqual(func_p.all(), expected_p.all())
        self.assertAlmostEqual(func_q.all(), expected_q.all())
    def test_7(self):
        a = np.array(([0.5,0.6,0.8],[0.9,0.7,0.8],[0.7,0.6,0.6]))
        func_p,func_q,func_price = nash_equilibrium(a)
        expected_price = 7/10
        expected_p = np.array([0,1,0])
        expected_q = np.array([0,1,0])
        self.assertAlmostEqual(func_price, expected_price)
        self.assertAlmostEqual(func_p.all(), expected_p.all())
        self.assertAlmostEqual(func_q.all(), expected_q.all())
    def test_8(self):
        a = np.array(([1,-1,-1],[-1,-1,-1]))
        func_p,func_q,func_price = nash_equilibrium(a)
        expected_price = -1
        expected_p = np.array([1,0])
        expected_q = np.array([0,1,0])
        self.assertAlmostEqual(func_price, expected_price)
        self.assertAlmostEqual(func_p.all(), expected_p.all())
        self.assertAlmostEqual(func_q.all(), expected_q.all())