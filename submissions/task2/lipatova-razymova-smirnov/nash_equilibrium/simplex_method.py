# coding: utf-8

import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import numpy as np
from fractions import Fraction


def make_matrix_positve(matrix):  # Делаем положительными элементы матрицы
    min = min_search(matrix)
    add_elem(matrix, min)
    return matrix, abs(min)  # Возвращаем матрицу и прибавленный элемент


def min_search(matrix):  # Минимум в таблице
    min = 0
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] < min:
                min = matrix[i][j]
    return min


def add_elem(matrix, elem):  # Добавляет модуль числа всем элементам матрицы
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            matrix[i][j] += abs(elem)


def check(matrix):  # Проверяет строку значений Z в таблице на наличие отрицательных
    return min(matrix[matrix.shape[0] - 1]) >= 0


def make_fract(a, b):  # Делаем дробь не боясь делить на 0
    if b != 0:
        return Fraction(a, b)
    else:
        return 0


def print_tab(tab):  # Вывод таблицы
    n = tab.shape[1]  # кол-во столбцов
    m = tab.shape[0]  # кол-во строк
    for i in range(m):
        for j in range(n):
            print(tab[i][j], end='  ')
        print()


def simplex_table(matrix):
    n = matrix.shape[1]  # кол-во столбцов
    m = matrix.shape[0]  # кол-во строк
    tab = np.full((m + 1, n + m + 2), Fraction(0))

    for i in range(m):
        for j in range(n):
            tab[i][j] = matrix[i][j]  # заполняем часть таблицы значениями матрицы

    for i in range(m):
        tab[i][i + n] = 1  # заполняем значения добавочных переменных

    for i in range(m):
        tab[i][n + m] = 1  # заполняем столбец значений (предпосдений столбец таблицы)

    for i in range(m):
        tab[i][n + m + 1] = i + n + 1  # заполняем номера базисных переменных (посдений столбец таблицы)

    for i in range(n):
        tab[m][i] = -1  # заполняем значения функции Z равные -1 на основных переменных (последняя строка)

    return tab


def new_simplex_matrix(tab):
    n = tab.shape[1]  # кол-во столбцов
    m = tab.shape[0]  # кол-во строк
    row, column, elem = best_correspondence(tab)  # Ищем подходящие столбец и строку

    for i in range(n-1):
        tab[row][i] = make_fract(tab[row][i], elem)  # Делим строку на значения элемента для перехода
                                                     # к новой базисной переменной
    for j in range(m):
        coeff = tab[j][column]
        for i in range(n - 1):
            if j != row:
                tab[j][i] -= tab[row][i] * coeff  # Вычитаем строку умноженную на коэф. из остальных строк

    tab[row][n - 1] = column + 1  # Меняем переменную базиса по номеру столбца
    return tab


def best_correspondence(tab):  # Ищем строку и столбец
    n = tab.shape[1]  # кол-во столбцов
    m = tab.shape[0]  # кол-во строк
    max_abs = abs(min(tab[m - 1]))  # Максимальный модуль в строке Z определяет столбец

    column = 0
    for i in range(n):
        if abs(tab[m - 1][i]) == max_abs and tab[m - 1][i] < 0:  # Смотрим на каком стобце достигается максимум
            column = i

    f = True
    for j in range(m - 1):  # Ищем строку с минимальным элементом в найденом столбце
        if tab[j][column] * tab[j][n - 2] > 0:  # Условие неравенства эл-тов нулю и совпадения их знаков
            fract = make_fract(tab[j][n - 2], tab[j][column])
            if f:
                row = j
                f = False  # Встретили первый подходящий стобец
                min_elem = fract
            else:
                if fract < min_elem:
                    min_elem = fract
                    row = j

    elem = tab[row][column]
    return row, column, elem


def get_op_plan(tab):
    n = tab.shape[1]  # кол-во столбцов
    m = tab.shape[0]  # кол-во строк

    straight = np.full((n - m - 1), Fraction(0, 1))
    reverse = np.full((m - 1), Fraction(0, 1))

    for i in range(m - 1):  # Формируем оптимальный план для второго игрока
        if tab[i][n - 1] <= n - m - 1:
            straight[tab[i][n - 1] - 1] = tab[i][n - 2]

    for i in range(m - 1):  # Формируем оптимальный план для первого игрока
        reverse[i] = tab[m - 1][i + n - m - 1]

    return straight, reverse


def spectr(a):
    y = a
    x = np.arange(1, np.size(a) + 1)

    fig, ax = plt.subplots()
    ax.set_title('Визуализация спектров')
    ax.set_ylabel('Вероятности')
    ax.set_xlabel('Стратегии')

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.02))

    ax.bar(x, y, width=0.02, color=(0.2, 0.4, 0.6, 0.6))
    ax.plot(x, y, marker='o', linestyle='')


def simplex_method(matrix):
    matrix, addition = make_matrix_positve(matrix)
    tab = simplex_table(matrix)  # Формируем первичную симплес таблицу
    while not check(tab):  # Основной цикл симплекс метода
        tab = new_simplex_matrix(tab)
        # print_tab(tab)
        # print()

    straight, reverse = get_op_plan(tab)  # Планы для второго и первого игрока

    z = sum(straight)  # Функция которую минимизируем
    price = make_fract(1, z)
    straight = straight * price
    reverse = reverse * price

    price -= addition  # Вычитаем добавку к цене

    return straight, reverse, price


def print_result(s, r, p, f):

    print("Цена игры ", p)

    if not f:
        print('Оптимальная смешанная стратегия 1 игрока | p | |', end="")
    else:
        print("                 Седловая точка \nПервый игрок  ", end="")

    for i in range(len(r)):
        print(str(r[i]).center(7), end='|')

    print()

    if not f:
        print('\nОптимальная смешанная стратегия 2 игрока | q | |', end="")
    else:
        print("Второй игрок  ", end="")

    for i in range(len(s)):
        print(str(s[i]).center(7), end='|')

    spectr(r)
    spectr(s)


def find_saddle_point(matrix):

    p = [0 for _ in range(len(matrix))]
    q = [0 for _ in range(len(matrix[0]))]

    for i in range(len(matrix)):
        # ищем минимальное в строке и максимальное в столбце
        min_value = min(matrix[i])
        indexes = filter(lambda x: matrix[i][x] == min_value, range(len(matrix[i])))

        for min_index in indexes:
            for j in range(len(matrix)):
                if matrix[j][min_index] > min_value:
                    break
            else:
                p[i] = 1
                q[min_index] = 1
                return p, q, min_value, 1

    return p, q, min_value, 0


def nash_equilibrium(matrix):
    p, q, price, flag = find_saddle_point(matrix)

    if not flag:
        p, q, price = simplex_method(matrix)

    return p, q, price, flag
