import unittest
import nash
from nash import *



class TestNashEq(unittest.TestCase):
    def Compare(self,p1, v1, v2, p2, str1, str2, msg):
        self.assertEqual(p1,p2,msg)
        for i in range(len(v1)):
            self.assertEqual(v1[i], str1[i], msg)
        for j in range(len(v2)):
            self.assertEqual(v2[j], str2[j], msg)
    def test_ones(self):
        matrix= [[1,1],[1,1]]
        price = 1.0
        p,q =[1.0, 0.0], [1.0, 0.0]
        p1, v1, v2 = nash.nash_equilibrium(matrix, 0)
        self.Compare(p1, v1, v2, price, p, q, 'Error1')
    def test_task(self):
        matrix =[[4, 0, 6, 2, 2, 1],
                [3, 8, 4, 10, 4, 4],
				[1, 2, 6, 5, 0, 0],
                [6, 6, 4, 4, 10, 3],
				[10, 4, 6, 4, 0, 9],
                [10, 7, 0, 7, 9, 8]]
        price =0
        p,q= 0,0
    def test_3(self):
        matrix = [[1,3],[3,1]]
        price= 2.0
        p, q = [0.5, 0.5], [0.5,0.5]
        p1, v1, v2 = nash.nash_equilibrium(matrix, 0)
        self.Compare(np.round_(p1,1), np.round_(v1,1), np.round_(v2,1), price, p, q, 'Error3')
    def test_4(self):
        matrix=[[3,9,2,1],
        [7,8,5,6],
        [4,7,3,5],
        [5,6,1,7]]
        price= 0.5
        p, q = [0.1, 0.3, 0.2, 0.1], [0.3, 0.1, 0.2, 0.4]
        p1, v1, v2 = nash.nash_equilibrium(matrix, 0)
        self.Compare(np.round_(p1,1), np.round_(v1,1), np.round_(v2,1), price, p, q, 'Error4')

    
        


        
if __name__ == '__main__':
    unittest.main()