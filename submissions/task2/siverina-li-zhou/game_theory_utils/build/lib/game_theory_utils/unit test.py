import numpy as np
import unittest
import scipy.linalg as sla
from random import randrange
class test_matrix(unittest.TestCase):
    def test_p(self):
        res=np.array([0, 4/31, 3/31, 27/62, 21/62, 0])
        for i in range (1, 6, 1):
            self.assertAlmostEqual(res[i], p[i], 1)
    def test_q(self):
        res=np.array([0, 0, 257/372, 9/62, 55/372, 1/62])
        for i in range (1, 6, 1):
            self.assertAlmostEqual(res[i], q[i], 1)
    def test_p1(self):
        res=np.array([1/4,1/2,1/4])
        for i in range (1, 6, 1):
            self.assertAlmostEqual(res[i], p[i], 1)
    def test_q1(self):
        res=np.array([1/4,1/2,1/4])
        for i in range (1, 6, 1):
            self.assertAlmostEqual(res[i], q[i], 1)
    def test_p2(self):
        res=np.array([19/35, 6/35, 2/7])
        for i in range (1, 6, 1):
            self.assertAlmostEqual(res[i], p[i], 1)
    def test_q2(self):
        res=np.array([9/35, 14/35, 12/35])
        for i in range (1, 6, 1):
            self.assertAlmostEqual(res[i], q[i], 1)
if __name__ == '__main__':
    unittest.main()
