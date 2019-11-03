#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#game solution

import numpy as np
import matplotlib.pyplot as plt
from nash_equilibrium_pkg.plot import plot_output


def minD_string(a,j):
    i = 0
    while (a[i][j] == 0):
       i +=1
    min = (a[i][0])/(a[i][j])
    str = i
    for i in range(len(a)-1):
        if ((a[i][j]) != 0) and (a[i][j] > 0):
            if  (a[i][0])/(a[i][j]) <= min:
                min = (a[i][0])/(a[i][j])
                str = i
    return str


def change_string (a,r,s):
    for i in range (len(a[r])):
        if (i != s):
            a[r][i] = a[r][i]/(a[r][s])
    a[r][s] = 1
    return a


def change_others (a,r,s):
    for i in range(len(a)):
        for j in range (len(a[0])):
            if ((i != r) and (j != s)):
                a[i][j] = (a[i][j])-((a[r][j])*(a[i][s])/(a[r][s]))
    return a


def change_col (a,r,s):
    for i in range (len(a)):
        if ( i != r):
            a[i][s] = 0
    return a


def changing (a,r,s):
    a = change_others(a,r,s)
    a = change_col(a,r,s)
    a = change_string(a,r,s)
    return a


def find_min_in_last_string(a):
    min = a[len(a)-1][0]
    column = 0
    for i in range (len(a[0]-1)):
        if ((a[len(a)-1][i]) <= min): 
            min = (a[len(a)-1][i])
            column = i
    return column


def simptab (a):
    n = len(a)
    m = len(a[0])
    a = a.transpose()
    a = np.concatenate((np.ones((1,len(a[0]))),a))
    E = np.eye(n)
    res = np.concatenate((a,E))
    res = res.transpose()
    vs = np.zeros((1,len(res[0])))
    for i in range(m):
        vs[0][i+1] = -1
    res = np.concatenate((res,vs))
    return res


def saddle(arr):
    n = len(arr[0])
    m = len(arr)
    a = [0]*n
    b = [0]*m
    for j in range(n):
        currentmax = arr[0][j]
        for i in range(m):
             if arr[i][j] > currentmax:
                 currentmax = arr[i][j]
        a[j] = currentmax
    for i in range(m):
        currentmin = arr[i][0]
        for j in range(n):
            if arr[i][j] < currentmin:
                currentmin = arr[i][j]
            b[i] = currentmin
    min = a[1]
    max = b[1]
    I,J = 1,1
    for j in range(n):
        if a[j] < min:
            min = a[j]
            J = j
    for i in range(m):
        if b[i] > max:
            max = b[i]
            I = i
    if max == min:
        print("Game price: ", min, ";", "Strategies:", I," ", J)
        return min, [I], [J]
    else:
        return None, None, None


def second_gamer(arr,a):
    m = len(arr)-1
    vector = np.zeros((1,m))
    for i in range(m):
       vector[0][-i-1] = (arr[-1][-1-i])*a
    return vector


def first_gamer(arr,base,a):
    m = len(base)               
    n = len(arr[0])-m-1         
    vector = np.zeros((1,n))
    for i in range(m):
        if base[i] <= n:
            vector[0][base[i]-1] = (arr[i][0])*a
    return vector


def get_vector(a):
    v = [0 for i in range(len(a[0]))]
    for i in range(len(a[0])):
        v[i] = a[0][i]
    return v


def nash_equilibrium(a):
    np.set_printoptions(precision = 5)
    game_p, v1, v2 = saddle(a)
    if game_p == None:
        res = simptab(a)
#        print(res)
        m = len(a)
        n = len(a[0])
        base_v = [int(i) for i in range(m+1,2*m+1 )]
        while any(res[-1, :] < 0):
            r=find_min_in_last_string(res)  
            str_of_minD = minD_string(res,r)
            res = changing(res,str_of_minD,r)
            base_v[str_of_minD] = r;
#            print(res)
#        print (res)
        game_p = 1/(res[-1][0])
        p_vector = second_gamer(res,game_p)
        q_vector = first_gamer(res,base_v,game_p)
        print("game price = ",game_p)
        v1 = get_vector(p_vector)
        print("p = ", v1)
        v2 = get_vector(q_vector)
        print("q = ", v2)
        plot_output(v1,'blue')
        plot_output(v2,'red')
        return game_p, v1, v2
    else:
        return game_p, v1, v2
