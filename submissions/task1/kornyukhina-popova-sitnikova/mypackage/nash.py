import numpy as np
from fractions import Fraction
from math import fabs
import matplotlib.pyplot as plt

# Проверка на положительность элементов исходной матрицы
def check_natural(a):
    a = a.astype('double')
    return a if ((a>0).all() == True) else a+fabs(a.min())+1

# Красивый вывод решения
def output_decision(x):
    print("|",end=' ')
    for i in range(x.shape[0]):
        print (str(Fraction(x[i]).limit_denominator()),"| ", end='')
# Красивый вывод матрицы
def output_matrix(a):
    shift = np.zeros(a.shape[1])
    for i in range(a.shape[1]):
        s = a[:,i]
        max_digit = len(str(s[0]))
        for j in range(1,s.shape[0]):
            if (len(str(s[j])) > max_digit):
                max_digit = len(str(s[j]))
        shift[i] = max_digit
        
    for i in range(a.shape[0]):
        print("| ",end='')
        for j in range(a.shape[1]):
            for k in range(int(shift[j]-1)-len(str(a[i,j]))+1):
                print(" ",end='')
            print (a[i,j],"| ",end='')
        print()   
def nash_equilibrium(a):
    #Упрощаем матрицу выигрыша
    a_edit, mas_i, mas_j = simple_win_matrix(a)
    if (check_saddle_point(a_edit)):
        #Находим седловую точку
        p = np.zeros(a.shape[0])
        q = np.zeros(a.shape[1])
        p[np.argmax(a.min(axis=1))] = 1
        q[np.argmin(a.max(axis=0))] = 1
        return p,q,a[np.argmax(a.min(axis=1)),np.argmin(a.max(axis=0))]
    else:
        #Получаем оптимальный план прямой и двойственной задач
        x,y = simplex_metod(a_edit)
        price_game = 1/x.sum() #Цена игры
        #Оптимальная смешанная стратегия игрока 1:
        p = np.zeros(a.shape[0])
        j = 0
        for i in range(a.shape[0]):
            if (i in mas_i):
                j+=1
                continue
            else:
                p[i] = y[i-j]
        p *= price_game
        #Оптимальная смешанная стратегия игрока 2:
        j = 0
        q = np.zeros(a.shape[1])
        for i in range(a.shape[1]):
            if (i in mas_j):
                j+=1
                continue
            else:
                q[i] = x[i-j]
        q *= price_game
        
        shift = 0
        if ((a>0).all() != True):
            shift = fabs(a.min())+1
        
        price_game = 0
        for i in range(p.shape[0]):
            if ((p[i] > 0) and (y[i]>0)):
                price_game = p[i]/y[i]-shift
                break
        
        return p,q,price_game

# Проверка на седловую точку
def check_saddle_point(a):
    return 1 if (a.min(axis=1).max() == a.max(axis=0).min()) else 0

# Редуцирование матрицы выигрыша
def simple_win_matrix(a):
    mas_i = set()
    mas_j = set()
    #Проходим по строкам
    for i in range(a.shape[0]):
        for j in range(a.shape[0]):
            if (i == j):
                continue
            else:
                if ((a[i,:]==a[j,:]).all() == True):
                    if (j > i):
                        mas_i.add(j)  
                elif ((a[i,:]>=a[j,:]).all() == True):
                    mas_i.add(j)  
    #Удаляем дублирующие и доминируемые строки
    n=0
    for i in mas_i:        
        a = np.delete(a,(i-n),axis=0)
        n+=1

    #Проходим по столбцам
    for i in range(a.shape[1]):
        for j in range(a.shape[1]):
            if (i == j):
                continue
            else:
                if ((a[:,i]==a[:,j]).all() == True):
                    if (j > i):
                        mas_j.add(j)  
                elif ((a[:,i]<=a[:,j]).all() == True):
                    mas_j.add(j)
    #Удаляем дублирующие и доминируемые столбцы
    n=0
    for i in mas_j:        
        a = np.delete(a,(i-n),axis=1)
        n+=1
      
    return a, mas_i, mas_j

# Симплекс метод
def simplex_metod(a):
    a = check_natural(a)
    
    #Для первого игрока
    #Заполняем simplex таблицу
    simplex_table = np.zeros((a.shape[0]+1,a.shape[1]+a.shape[0]+1))
    for i in range(a.shape[0]):
        for j in range(1,a.shape[1]+1):
            simplex_table[i,j] = a[i,j-1]
    for i in range(a.shape[0]):
        simplex_table[i,0] = 1
    for j in range(1,a.shape[1]+1):
        simplex_table[a.shape[0],j] = -1    
    for i in range(a.shape[0]):
        simplex_table[i, i+a.shape[1]+1] = 1
    
    #Массив изменненых позиций
    mas_exch = np.arange(a.shape[0])
    mas_exch+=a.shape[1]+1
    pos_des = np.arange(2)
    
    #Цикл до того, пока есть отрицательные индексные значения
    while((simplex_table[simplex_table.shape[0]-1,1:]>=0).all() != True):
        lead = min(simplex_table[a.shape[0],1:1+a.shape[0]+a.shape[1]]) #Значение ведущего элемента
        for j in range(1,1+a.shape[0]+a.shape[1]):
            if (simplex_table[a.shape[0],j] == lead):
                pos_lead = [a.shape[0],j]
           
        #min(simplex_table[:a.shape[0],0]/simplex_table[:a.shape[0],pos_lead[1]])
        
        #Позиция разрешающего элемента
        srez = simplex_table[:a.shape[0],0]/simplex_table[:a.shape[0],pos_lead[1]]
        #print(srez)
        pos_des[0] = np.argmin(srez)
        pos_des[1] = pos_lead[1]
        des = simplex_table[pos_des[0],pos_des[1]] # Значение разрешаюющего элемента
        
        while (des < 0):
            srez = np.delete(srez, pos_des[0], axis=0)
            #print(srez)
            if (srez.shape[0] > 0):
                pos_des[0] = np.argmin(srez)
                des = simplex_table[pos_des[0],pos_des[1]] # Значение разрешаюющего элемента
            else:
                break
            
        #Метод прямоугольника
        for i in range(simplex_table.shape[0]):
            if (i == pos_des[0]):
                continue
            else:    
                for j in range(simplex_table.shape[1]):
                    if (j == pos_des[1]):
                        continue
                    else:
                        simplex_table[i,j] -= simplex_table[i,pos_des[1]] * simplex_table[pos_des[0],j] / des
        simplex_table[pos_des[0],:] = simplex_table[pos_des[0],:] / des
        for i in range(simplex_table.shape[0]):
            if (i == pos_des[0]):
                continue
            else:
                simplex_table[i,pos_des[1]] = 0
        
        mas_exch[pos_des[0]] = pos_des[1]
        
    #Находим решение прямой задачи (для 1 игрока)
    set_exch = set()
    for i in range(a.shape[0]):
        if (mas_exch[i] <= a.shape[0]):
            set_exch.add(mas_exch[i])
    
    y = np.zeros(a.shape[1])
    
    for i in range(1,a.shape[1]+1):
        if i in set_exch:
            for j in range(mas_exch.shape[0]):
                if (i == mas_exch[j]):
                    y[i - 1] = simplex_table[j,0]
                    break
    
    #Находим решение обратной задачи (для 2 игрока)
    x = np.zeros(a.shape[0])
    
    for i in range(a.shape[0]):
        x[i] = simplex_table[a.shape[0],1+a.shape[1]+i]    
        
    return y,x

# Визуализация решений
def visualize(p):
    plt.title("Спектр оптимальных стратегий")
    plt.axis([0,p.shape[0]+1,0,p.max()+0.1])
    for i in range(p.shape[0]):
        plt.axvline(x=i+1,ymax=p[i]/(p.max()+0.1))
    plt.plot(np.arange(p.shape[0])+1,p,'ro')    
    plt.ylabel("Вероятность стратегии")
    plt.xlabel("Номер стратегии")
    plt.show()