#coding utf-8
from .matrix_game import nash_equilibrium
import numpy as np
import math

def isclose(a, b, rel_tol=1e-3, abs_tol=0.1):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def test_1():
	matrix = np.array([[1, -3, -2],[0, 5, 4],[2, 3, 2]])
	p_test = [0, 0, 1]
	q_test = [1, 0, 0]
	price_test = 1/2

	price, p, q = nash_equilibrium(matrix)

	for k in range(len(p_test)):
		assert isclose(p_test[k], p[k])
	for i in range(len(q_test)):
		assert isclose(q_test[i], q[i])
	assert price == price_test

def test_2():
	matrix = np.array([[3, 6, 1, 4],[5, 2, 4, 2],[1, 4, 3, 5],[4, 3, 4, -1]])
	p_test = [1/8, 25/52, 19/52, 3/104]
	q_test = [1/8, 37/104, 23/52, 1/13]
	price_test = 339/104

	price, p, q = nash_equilibrium(matrix)

	for k in range(len(p_test)):
		assert isclose(p_test[k], p[k])
	for i in range(len(q_test)):
		assert isclose(q_test[i], q[i])
	assert isclose(price, price_test)

def test_3():
	matrix = np.array([[1, 2], [2, 1]])
	p_test = [1/2, 1/2]
	q_test = [1/2, 1/2]
	price_test = 3/2

	price, p, q = nash_equilibrium(matrix)

	for k in range(len(p_test)):
		assert isclose(p_test[k], p[k])
	for i in range(len(q_test)):
		assert isclose(q_test[i], q[i])
	assert isclose(price, price_test)

