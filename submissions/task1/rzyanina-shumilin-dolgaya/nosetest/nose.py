import numpy as np
import prak
from nose.tools import assert_equals
eps = 0.000001

def test1():    #матрица с чистыми стратегиями
	m1 = np.array([
	    [3,9,2,1],
	    [7,8,5,6],    
	    [4,7,3,5],
	    [5,6,1,7]
	])
	p1 = np.array([0, 1, 0, 0])
	q1 = np.array([0, 0, 1, 0])
	p, q, value = prak.nash_equilibrium(m1)
	assert_equals(value, 5)
	for i in range(0, len(p1)):
		assert_equals((p[i]-p1[i]) < eps, True)
	for i in range(0, len(q1)):
		assert_equals((q[i]-q1[i]) < eps, True)
	return
def test2():   #матрица с полным спектром
	m1 = np.array([
		[4, 7, 2],
		[7, 3, 2],
		[2, 1, 8],
		])
	p1 = np.array([0.42647059, 0.23529412, 0.33823529])
	q1 = np.array([0.35294118, 0.26470588, 0.38235294])
	p, q, value = prak.nash_equilibrium(m1)
	assert_equals(value, 4.029411764705882)
	for i in range(0, len(p1)):
		assert_equals((p[i]-p1[i]) < eps, True)
	for i in range(0, len(q1)):
		assert_equals((q[i]-q1[i]) < eps, True)
	return
def test3():   #матрица с неполным спектром
	m1 = np.array([
	    [4,0,6,2,2,1],
	    [3,8,4,10,4,4],
	    [1,2,6,5,0,0],
	    [6,6,4,4,10,3],
	    [10,4,6,4,0,9],
	    [10,7,0,7,9,8]
	])
	p1 = np.array([0, 0.12903226, 0.09677419, 0.43548387, 0.33870968, 0])
	q1 = np.array([0, 0, 0.69086022, 0.14516129, 0.14784946, 0.01612903])
	p, q, value = prak.nash_equilibrium(m1)
	assert_equals(value, 4.870967741935484)
	for i in range(0, len(p1)):
		assert_equals((p[i]-p1[i]) < eps, True)
	for i in range(0, len(q1)):
		assert_equals((q[i]-q1[i]) < eps, True)
	return