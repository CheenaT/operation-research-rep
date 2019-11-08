from fractions import Fraction
import numpy as np
import pytest

from nash_equilibrium.simplex_method import nash_equilibrium


@pytest.mark.parametrize("matrix,p_ans,q_ans,price_ans,flag_ans", [
    (
        np.array([
            [4, 0, 6, 2, 2, 1],
            [3, 8, 4, 10, 4, 4],
            [1, 2, 6, 5, 0, 0],
            [6, 6, 4, 4, 10, 3],
            [10, 4, 6, 4, 0, 9],
            [10, 7, 0, 7, 9, 8]
        ]),
        np.array([Fraction(0, 1), Fraction(4, 31), Fraction(3, 31),
                  Fraction(27, 62), Fraction(21, 62), Fraction(0, 1)]),
        np.array([Fraction(0, 1), Fraction(0, 1), Fraction(257, 372),
                  Fraction(9, 62), Fraction(55, 372), Fraction(1, 62)]),
        Fraction(151, 31),
        False
    ),
    (
        np.array([
            [1,2,1,2],
            [2,1,2,4],
            [3,3,2,2],
            [4,1,3,3],
        ]),
        np.array([Fraction(0, 1), Fraction(0, 1), Fraction(2, 3), Fraction(1, 3)]),
        np.array([Fraction(0, 1), Fraction(1, 3), Fraction(1, 3), Fraction(1, 3)]),
        Fraction(7, 13),
        False
    ),
    (
        np.array([
            [1,2,1,2],
            [2,1,2,1],
            [3,2,3,2],
            [4,3,4,3],
        ]),
        np.array([Fraction(0, 1), Fraction(1, 1), Fraction(0, 1), Fraction(0, 1)]),
        np.array([Fraction(0, 1), Fraction(0, 1), Fraction(0, 1), Fraction(1, 1)]),
        Fraction(3, 1),
        True
    ),
    (
        np.array([
            [2,5,3],
            [3,1,7],
            [8,0,2],
        ]),
        np.array([Fraction(11, 17), Fraction(5, 34), Fraction(7, 34)]),
        np.array([Fraction(6, 17), Fraction(25, 68), Fraction(19, 68)]),
        Fraction(151, 34),
        False
    ),
    (
        np.array([
            [2,1,2],
            [1,2,1],
            [3,3,3],
        ]),
        np.array([Fraction(1, 1), Fraction(0, 1), Fraction(0, 1)]),
        np.array([Fraction(0, 1), Fraction(0, 1), Fraction(1, 1)]),
        Fraction(3, 1),
        False
    ),
    (
        np.array([
            [2,1,2],
            [1,2,1],
            [1,1,2],
        ]),
        np.array([Fraction(1, 2), Fraction(1, 2), Fraction(0, 1)]),
        np.array([Fraction(0, 1), Fraction(1, 2), Fraction(1, 2)]),
        Fraction(3, 2),
        False
    ),
    (
        np.array([
            [3,9,2,1],
            [7,8,5,6],
            [4,7,3,5],
            [5,6,1,7],
        ]),
        np.array([Fraction(0, 1), Fraction(0, 1), Fraction(1, 1), Fraction(0, 1)]),
        np.array([Fraction(0, 1), Fraction(1, 1), Fraction(0, 1), Fraction(0, 1)]),
        Fraction(5, 1),
        True
    ),
    (
        np.array([
            [4,5,9,3],
            [8,4,3,7],
            [7,6,8,9],
        ]),
        np.array([Fraction(0, 1), Fraction(1, 1), Fraction(0, 1), Fraction(0, 1)]),
        np.array([Fraction(0, 1), Fraction(0, 1), Fraction(1, 1)]),
        Fraction(6, 1),
        True
    ),
    (
        np.array([
            [0,2,7],
            [12,11,1],
        ]),
        np.array([Fraction(11, 18), Fraction(7, 18)]),
        np.array([Fraction(1, 3), Fraction(0, 1), Fraction(2, 3)]),
        Fraction(14, 3),
        False
    ),
    (
        np.array([
            [6,5,7],
            [10,4,7],
            [13,10,4],
            [7,11,5],
        ]),
        np.array([Fraction(24, 35), Fraction(2, 35), Fraction(0, 1), Fraction(9, 35)]),
        np.array([Fraction(2, 35), Fraction(8, 35), Fraction(5, 7)]),
        Fraction(227, 35),
        False
    ),

])
def test_nash_test_equilibrium(matrix, p_ans, q_ans, price_ans, flag_ans):
    p_res, q_res, price_res, flag_res = nash_equilibrium(matrix)
    assert (p_res == p_ans).all()
    assert (q_res == q_ans).all()
    assert price_res == price_ans
    assert flag_res == flag_ans