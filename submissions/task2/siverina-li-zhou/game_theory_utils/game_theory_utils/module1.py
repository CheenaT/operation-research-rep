import numpy as np
import unittest
import scipy.linalg as sla
from random import randrange
def nash_equilibrium(a):
    i = 0
    k = 1000
    summ1 = np.zeros(n)
    summ2 = np.zeros(m)
    v = 0
    V = 0
    u = []
    q = [0] * m
    p = [0] * n
    c = 1
    t = randrange(0, n, 1)
    print(t)
    while c <= k:
        summ1 = summ1 + a[t]
        MIN = int(min(summ1))
        for i in range (m):
            if (MIN == summ1[i]):
                t = i
                q[t] += 1
                break
        summ2 = summ2 + A[t]
        MAX = int(max(summ2))
        for i in range (n):
            if (MAX == summ2[i]):
                t = i
                p[t] += 1
                break
        v = MIN / c
        V = MAX / c
        u.append((v + V) / 2)
        c = c + 1
    w = sum(u) / k
    Kp = np.array([k] * n)
    Kq = np.array([k] * m)
    p = p / Kp
    q = q / Kq
    return w, p, q

def sedlo(a):
    q = [0] * m
    p = [0] * n
    for i in range(n):
        minn.append(min(a[i]))
    for i in range(m):
        maxx.append(max(A[i]))
    if (max(minn) == min(maxx)):
        w = max(minn)
    else:
        w, p, q = nash_equilibrium(a)
    return w, p, q
    c = 0
    for i in range (n):
        if (w != minn[i]):
            c += 1
        else:
            p[c] = 1
    c = 0
    for i in range (m):
        if (w != maxx[i]):
            c += 1
        else:
            q[c] = 1
    return w, p, q


def func1(A):
    indeces = np.unique(A, axis = 1, return_index=True)[1]
    A = A[:, np.sort(indeces)]
    used = np.zeros(A.shape[1])
    for i in range(1, A.shape[1]):
        B_roll = np.roll(A, shift = i, axis = 1)
        diff_A_B = A - B_roll
        qwe = np.all(diff_A_B >= 0, axis = 0)
        used = np.logical_or(used, qwe)
    A = np.delete(A, np.arange(used.shape[0])[(used) > 0], 1)
    return A, np.sum(used)


def func2(A):
    indeces = np.unique(A, axis = 0, return_index=True)[1]
    A = A[np.sort(indeces),:]
    used = np.zeros(A.shape[0])
    for i in range(1, A.shape[0]):
        B_roll = np.roll(A, shift = i, axis = 0)
        diff_A_B = A - B_roll
        qwe = np.all(diff_A_B <= 0, axis = 1)
        used = np.logical_or(used, qwe)
    A = np.delete(A, np.arange(used.shape[0])[(used) > 0], 0)
    return A, np.sum(used)
