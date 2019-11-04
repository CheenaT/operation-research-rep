import numpy as np
import matplotlib.pyplot as plt



def print_matrix(matrix):
    max_len = max([len(str(e)) for r in matrix for e in r])
    for row in matrix:
        print(*list(map('{{:>{length}}}'.format(length=max_len).format, row)))


def first_simplex_table(i, j, A):
    matrix = np.ones((i+2,j+3)) #создание единичной матрицы 
    matrix[0] = 0
    matrix[1:i+2, 0] = range(j+1,i+j+2) #индексы базисных неизвестных
    matrix[i+1,0] = -1 #вместо обозначения F
    matrix[0, 2:j+2] = range(1,j+1) #индексы свободных неизвестных   
    matrix[1:i+1, 2:j+2] = A #заполняем матрицей выигрышей
    matrix[i+1,1] = 0 #свободный член функции цели
    matrix[i+1,2:j+2] *= -1 #коэффициенты целевой функции
    matrix[:,j+2] = 0 #обнуляем столбец для вспомогательных коэффициентов 
##    print(matrix)
    return matrix



def simplex_find(i, j, A):
    a = (A[i+1,2:j+2]).copy() #копируем значения свободных неизвестных индексной строки
    for n in range (0,j):
        if a[n] > 0: a[n] = 0 
    a = abs(a)        
##    print("Max",a)    
    b = np.where(a == np.amax(a))[0] #массив индексов макс.элементов инд. стр.
    if len(b) > 1: #выбираем индекс ведущего столбца:
        lead_column = b[len(b)-1]+2 #индекс ведущего столбца: +2 тк перед ним два служебных столбца
    else:
        lead_column = b[0]+2
##    print("Ведущий столбец: X" , int(A[0, lead_column])) 

    c = np.zeros(i)
    
    for n in range (0,i):
       if A[n+1,1] > 0 and A[n+1,lead_column] < 0 or A[n+1,lead_column] == 0: #проверка на неограниченность линейной формы
            c[n] = float("inf")
       else: 
            c[n] = A[n+1,1] / A[n+1,lead_column]
       
##    print("Min",c) #печатаем отношение свободных членов к ведущему столбцу
    
    d = np.where(c == np.amin(c))[0] #массив индексов мин.элементов отношения
    if abs(c[d[0]]) == np.inf:
        raise Exception("Пространство допустимых решений неограниченно. Решения не существует.")
    lead_line = d[0]+1 #индекс ведущей строки: +1 тк нумерация свободных переменных с 1
##    print("Ведущая строка: X" , int(A[lead_line, 0]) )
    
    return lead_column, lead_line


def next_simplex_table(i, j, A):
    lead_elements = simplex_find(i, j, A) #находим ведущие столбец и строку
    lead_column = lead_elements[0]
    lead_line = lead_elements[1]
    lead_element = A[lead_line, lead_column]
##    print(lead_elements)
    
    A[lead_line, 0], A[0, lead_column] = A[0, lead_column], A[lead_line, 0] # ведущие переменные меняем местами
    
    for n in range(1,i+2):
            if n != lead_line: A[n, j+2] = -1 * A[n,lead_column] #отсутсвие у ведущей строки вспомогательного коэффициента 
                
##    print("Вспомогательные коэффициенты: ", A[:,j+2])
                
    for n in range(1,j+2): #заполняем новый базисный элемент
            if n == lead_column: 
                A[lead_line, n] = 1/lead_element
            else:
                A[lead_line, n] /=  lead_element
               
##    print("Новый базисный элемент: ", A[lead_line])
    
    for n in range(1,i+2): #внешний цикл по строкам
            if n != lead_line: #заполняем оставшиеся базисные элементы
                for m in range(1, j+2):
                    if(m == lead_column):
                        A[n,m] = A[lead_line,m]*A[n, j+2] #+0 тк в предыдущей таблице не было столбца у базисного элемента
                    else:
                        A[n,m] = A[lead_line,m]*A[n, j+2]+A[n,m]
##    print(A)



def optimality_criterion(i, j, A):
    for n in range(2, j+2):
        if A[i+1, n] < 0:
            return False
    return True




def nash_equilibrium(A,gr):

    a, b = (np.shape(A)) #размер матрицы: кол-во строк, кол-во столбцов 

    B = first_simplex_table(a, b, A) #построение первой симплексной таблицы

    count = 0 #стетчик итераций симплекс-метода
    while not optimality_criterion(a, b, B): #проверка выполнения критерия оптимальности
##        print("Итерация №", count)
        next_simplex_table(a, b, B) #построение следующей симплексной таблицы
        count += 1
    
    solution_to_primal = np.zeros(b)
    solution_to_dual = np.zeros(a)
    for n in range(a): #выписываем решение прямой задачи
        if b >= B[n+1,0] > 0:
            solution_to_primal[int(B[n+1,0])-1] = B[n+1,1]
    for n in range(b): #выписываем решение двойственной задачи
        if a+b >= B[0,n+2] > b:
            solution_to_dual[int(B[0,n+2])-b-1] = B[a+1,n+2]

    price = 1/np.sum(solution_to_primal) #цена игры
    strategy1=price*solution_to_dual
    strategy2=price*solution_to_primal

    #print("Значение игры:", price)
    #print("Оптимальная стратегия первого игрока:", strategy1)
    #print("Оптимальная стратегия второго игрока:", strategy2)
    
    if (gr):
        x = np.arange(1, a+1)
        y = price*solution_to_dual
        fig, ax = plt.subplots()

        ax.scatter(x, y)

        plt.show()
    
        x = np.arange(1, b+1)
        y = price*solution_to_primal
        fig, ax = plt.subplots()

        ax.scatter(x, y)

        plt.show()

    return price, strategy1, strategy2
