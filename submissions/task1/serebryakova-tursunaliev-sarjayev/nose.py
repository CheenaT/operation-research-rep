import numpy as np
from scipy.optimize import linprog
from nose.tools import assert_equals
from superrnash import nash

def nash_equilibrium1(a):
    c = [-1 for i in range(0,a.shape[1])]
    b = [1 for i in range(0,a.shape[0])]
    q = linprog(c, a, b).x
    p = linprog(b, -a.transpose(),c).x
    opt_sum = 0
    for i in p:
        opt_sum+=i
    cost = 1/opt_sum
    return (p*cost, q*cost,cost)

class TestNash:
    def test_1(self):
        matrix = np.array([
        [8, 4, 7],
        [6, 5, 9],
        [7, 6, 8],
        ])
        
        func_p,func_q,func_cost = nash_equilibrium1(matrix)
        cost = 6
        p = np.array([0, 0, 1])
        q = np.array([0, 1, 0])
        assert_equals(abs(func_cost - cost) < 0.00001, True)
        assert_equals(sum(abs(x - y) < 0.00001 for x,y in zip(func_p,p)), len(func_p))
        assert_equals(sum(abs(x - y) < 0.00001 for x,y in zip(func_q,q)), len(func_q))
    
    def test_2(self):
        matrix = np.array([
        [4, 7, 2],
        [7, 3, 2],
        [2, 1, 8],
        ])
        func_p,func_q,func_cost = nash_equilibrium1(matrix)
        cost = 4.02941
        p = np.array([0.42647, 0.23529, 0.33824])
        q = np.array([0.35294, 0.26471, 0.38235])
        assert_equals(abs(func_cost - cost) < 0.00001, True)
        assert_equals(sum(abs(x - y) < 0.00001 for x,y in zip(func_p,p)), len(func_p))
        assert_equals(sum(abs(x - y) < 0.00001 for x,y in zip(func_q,q)), len(func_q))
    def test_3(self):
        matrix = np.array([
        [4, 0, 6, 2, 2, 1],
        [3, 8, 4, 10, 4, 4],
        [1, 2, 6, 5, 0, 0],
        [6, 6, 4, 4, 10, 3],
        [10, 4, 6, 4, 0, 9],
        [10, 7, 0, 7, 9, 8]
        ])
        func_p,func_q,func_cost = nash_equilibrium1(matrix)
        cost = 151/31
        p = np.array([0, 4/31, 3/31, 27/62, 21/62, 0])
        q = np.array([0, 0, 257/372, 9/62, 55/372, 1/62])
        assert_equals(abs(func_cost - cost) < 0.00001, True)
        assert_equals(sum(abs(x - y) < 0.00001 for x,y in zip(func_p,p)), len(func_p))
        assert_equals(sum(abs(x - y) < 0.00001 for x,y in zip(func_q,q)), len(func_q))
        
    def test_4(self):
        matrix = np.array([
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5],
        ])
        func_p,func_q,func_cost = nash_equilibrium1(matrix)
        cost = 3
        p = np.array([0, 0, 1])
        q = np.array([1, 0, 0])
        assert_equals(abs(func_cost - cost) < 0.00001, True)
        assert_equals(sum(abs(x - y) < 0.00001 for x,y in zip(func_p,p)),
            len(func_p))
        assert_equals(sum(abs(x - y) < 0.00001 for x,y in zip(func_q,q)), len(func_q))
    
    def test_5(self):
        matrix = np.array([
        [4, 5, 6, 7],
        [1, 2, 3, 4],
        [3, 4, 5, 9],
        [8, 6, 2, 4],
        ])
        func_p,func_q,func_cost = nash_equilibrium1(matrix)
        cost = 5
        p = np.array([0.75, 0, 0, 0.25])
        q = np.array([0.5 , 0, 0.5, 0])
        assert_equals(abs(func_cost - cost) < 0.00001, True)
        assert_equals(sum(abs(x - y) < 0.00001 for x,y in zip(func_p,p)),
            len(func_p))
        assert_equals(sum(abs(x - y) < 0.00001 for x,y in zip(func_q,q)), len(func_q))
        
    def test_6(self):
        matrix = np.array([
        [8, 4, 7],
        [6, 5, 9],
        [7, 6, 8],
        ])
        func_p,func_q,func_cost = nash_equilibrium1(matrix)
        p,q,cost=nash.nash_equilibrium(matrix)
        assert_equals(abs(func_cost - cost) < 0.00001, True)
        assert_equals(sum(abs(x - y) < 0.00001 for x,y in zip(func_p,p)), len(func_p))
        assert_equals(sum(abs(x - y) < 0.00001 for x,y in zip(func_q,q)), len(func_q))
        
    def test_7(self):
        matrix = np.array([
        [4, 7, 2],
        [7, 3, 2],
        [2, 1, 8],
        ])
        func_p,func_q,func_cost = nash_equilibrium1(matrix)
        p,q,cost=nash.nash_equilibrium(matrix)
        assert_equals(abs(func_cost - cost) < 0.00001, True)
        assert_equals(sum(abs(x - y) < 0.00001 for x,y in zip(func_p,p)), len(func_p))
        assert_equals(sum(abs(x - y) < 0.00001 for x,y in zip(func_q,q)), len(func_q))
        
    def test_8(self):
        matrix = np.array([
        [4, 0, 6, 2, 2, 1],
        [3, 8, 4, 10, 4, 4],
        [1, 2, 6, 5, 0, 0],
        [6, 6, 4, 4, 10, 3],
        [10, 4, 6, 4, 0, 9],
        [10, 7, 0, 7, 9, 8]
        ])
        func_p,func_q,func_cost = nash_equilibrium1(matrix)
        p,q,cost=nash.nash_equilibrium(matrix)
        assert_equals(abs(func_cost - cost) < 0.00001, True)
        assert_equals(sum(abs(x - y) < 0.00001 for x,y in zip(func_p,p)), len(func_p))
        assert_equals(sum(abs(x - y) < 0.00001 for x,y in zip(func_q,q)), len(func_q))
        
    def test_9(self):
        matrix = np.array([
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5],
        ])
        func_p,func_q,func_cost = nash_equilibrium1(matrix)
        p,q,cost=nash.nash_equilibrium(matrix)
        assert_equals(abs(func_cost - cost) < 0.00001, True)
        assert_equals(sum(abs(x - y) < 0.00001 for x,y in zip(func_p,p)), len(func_p))
        assert_equals(sum(abs(x - y) < 0.00001 for x,y in zip(func_q,q)), len(func_q))


    def test_10(self):
        matrix = np.array([
        [4, 5, 6, 7],
        [1, 2, 3, 4],
        [3, 4, 5, 9],
        [8, 6, 2, 4],
        ])
        func_p,func_q,func_cost = nash_equilibrium1(matrix)
        p,q,cost=nash.nash_equilibrium(matrix)
        assert_equals(abs(func_cost - cost) < 0.00001, True)
        assert_equals(sum(abs(x - y) < 0.00001 for x,y in zip(func_p,p)),
            len(func_p))
        assert_equals(sum(abs(x - y) < 0.00001 for x,y in zip(func_q,q)), len(func_q))

        

