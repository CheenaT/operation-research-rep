from .game import nash_equilibrium
from fractions import Fraction as frc # Так как сранивать float без eps невозможно
import numpy as np

def test_game_internet():
	internet_ex = [[3, 6, 1, 4],[5, 2, 4, 2],[1, 4, 3, 5],[4, 3, 4, -1]] # Тест из интернета
	p = [1/8, 25/52, 19/52, 3/104]
	q = [1/8, 37/104, 23/52, 1/13]
	price = 3.2596153846153846
	p1,q1,price1 = nash_equilibrium(internet_ex)
	for k in range(4):
		assert frc(q1[k]).limit_denominator(1000) == frc(q[k]).limit_denominator(1000)
		assert frc(p1[k]).limit_denominator(1000) == frc(p[k]).limit_denominator(1000)
	assert frc(price1).limit_denominator(1000) == frc(price).limit_denominator(1000)

def test_game_task():
	task_test_matrix = [[4, 0, 6, 2, 2, 1],[3, 8, 4, 10, 4, 4],
						[1, 2, 6, 5, 0, 0],[6, 6, 4, 4, 10, 3],
						[10, 4, 6, 4, 0, 9],[10, 7, 0, 7, 9, 8]] # Тест из задания прака
	p = [0, 4/31, 3/31, 27/62, 21/62, 0]
	q = [0, 0, 257/372, 9/62, 55/372, 1/62]
	price = 151/31
	p1,q1,price1 = nash_equilibrium(task_test_matrix)
	for k in range(6):
		assert frc(q1[k]).limit_denominator(1000) == frc(q[k]).limit_denominator(1000)
		assert frc(p1[k]).limit_denominator(1000) == frc(p[k]).limit_denominator(1000)
	assert frc(price1).limit_denominator(1000) == frc(price).limit_denominator(1000)

def test_game_fake():
	fake_test = [[3, 1],[1, 3]]    # Тест Миши
	p = [1/2, 1/2]
	q = [1/2, 1/2]
	price = 2
	p1,q1,price1 = nash_equilibrium(fake_test)
	for k in range(2):
		assert frc(q1[k]).limit_denominator(1000) == frc(q[k]).limit_denominator(1000)
		assert frc(p1[k]).limit_denominator(1000) == frc(p[k]).limit_denominator(1000)
	assert frc(price1).limit_denominator(1000) == frc(price).limit_denominator(1000)

def test_game_saddle():
	saddle_test = [[1, 2],[3, 4]]  # Седловая точка 1
	p = [0, 1]
	q = [1, 0]
	price = 3
	p1,q1,price1 = nash_equilibrium(saddle_test)
	for k in range(2):
		assert frc(q1[k]).limit_denominator(1000) == frc(q[k]).limit_denominator(1000)
		assert frc(p1[k]).limit_denominator(1000) == frc(p[k]).limit_denominator(1000)
	assert frc(price1).limit_denominator(1000) == frc(price).limit_denominator(1000)

def test_game_saddle2():
	saddle2_test = [[2, 2],[2, 2]] # Седловая точка 2
	p = [1/2, 1/2]
	q = [1/2, 1/2]
	price = 2
	p1,q1,price1 = nash_equilibrium(saddle2_test)
	for k in range(2):
		assert frc(q1[k]).limit_denominator(1000) == frc(q[k]).limit_denominator(1000)
		assert frc(p1[k]).limit_denominator(1000) == frc(p[k]).limit_denominator(1000)
	assert frc(price1).limit_denominator(1000) == frc(price).limit_denominator(1000)

def test_game_not_square_saddle():
	not_square_saddle = [[1, 2, 3],[1, 1, 1]]  # Прямоугольная и седловая точка
	p = [1/2, 1/2]
	q = [1, 0, 0]
	price = 1
	p1,q1,price1 = nash_equilibrium(not_square_saddle)
	for k in range(2):
		assert frc(q1[k]).limit_denominator(1000) == frc(q[k]).limit_denominator(1000)
		assert frc(p1[k]).limit_denominator(1000) == frc(p[k]).limit_denominator(1000)
	assert frc(q1[2]).limit_denominator(1000) == frc(q[2]).limit_denominator(1000)
	assert frc(price1).limit_denominator(1000) == frc(price).limit_denominator(1000)

def test_game_not_square_not_saddle():
	not_square_not_saddle = [[-1, -2, 3],[2, 4, 1]]  # Прямоугольная и не седловая точка
	p = [1/5, 4/5]
	q = [2/5, 0, 3/5]
	price = 7/5
	p1,q1,price1 = nash_equilibrium(not_square_not_saddle)
	for k in range(2):
		assert frc(q1[k]).limit_denominator(1000) == frc(q[k]).limit_denominator(1000)
		assert frc(p1[k]).limit_denominator(1000) == frc(p[k]).limit_denominator(1000)
	assert frc(q1[2]).limit_denominator(1000) == frc(q[2]).limit_denominator(1000)
	assert frc(price1).limit_denominator(1000) == frc(price).limit_denominator(1000)

def test_game_not_square_not_saddle2():
	not_square_not_saddle = [[-1, 2],[-2, 4],[3, 1]]   # Прямоугольная и не седловая точка
	p = [0, 1/4, 3/4]
	q = [3/8, 5/8]
	price = 7/4
	p1,q1,price1 = nash_equilibrium(not_square_not_saddle)
	for k in range(2):
		assert frc(q1[k]).limit_denominator(1000) == frc(q[k]).limit_denominator(1000)
		assert frc(p1[k]).limit_denominator(1000) == frc(p[k]).limit_denominator(1000)
	assert frc(p1[2]).limit_denominator(1000) == frc(p[2]).limit_denominator(1000)
	assert frc(price1).limit_denominator(1000) == frc(price).limit_denominator(1000)
