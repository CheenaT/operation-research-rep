from nash import *

help='/print -- для вывода текущей занесенной матрицы\n/enter -- для начала ввода матрицы\n/random -- для автосоздания и заполнения матрицы\n/test -- для тестирования\n/result -- для вычисления результата\n/graph -- для построения графика стратегий\n/help -- для вывода списка команд\n/exit -- для завершения программы'
print("print /help for getting help")
matr = []
a=0
B=[]
price=0

while(1):
    s=str(input(">>> "))
    if (s=="/help"):
        print(help)
    elif (s=="/print"):
        if (matr ==[]):
            print("Ошибка! Еще не введена матрица")
            continue
        else:
            print_matrix(matr)
    elif (s=='/enter'):
        print("Введите через пробел размеры матрицы")
        n,m=[int(i) for i in input().split()]
        print('Вводите по одному элементы матрицы')
        matr= [[int(input()) for j in range(m)] for i in range(n)]
        print_matrix(matr)
    elif (s=="/random"):
        a,b = eval(input('Введите колличество ст0лбцов и строк(через запятую): '))
        matr = np.random.randint(0,15,size=(a,b))
        print_matrix(matr)
    elif (s=='/result'):
        if (matr == []):
            print('Enter matrix first')
            continue
        else:
            nash_equilibrium(matr)
    elif (s=="/exit"):
        print("Ending! Bye")
        break
    else:
        print('Wrong command, try again')
