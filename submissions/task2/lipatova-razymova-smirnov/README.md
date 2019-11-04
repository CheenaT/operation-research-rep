# Nash Equilibrium

Simplex method implementation.
Требования к системе:
	1)Python версии не ниже 3.6
	2)Необходимые пакеты:
	 - matplotlib
	 - numpy
	 
Запуск
	nash_equilibrium:
		На вход подается матрица игры.
		На выходе:
			p-стратегия первого игрока
			q-стратегия второго игрока
			price-цена игры
			flag-индикатор седловых точек
	print_result:
		На вход подается параметры:
			s-стратегия первого игрока
			r-стратегия второго игрока
			p-цена игры
			f-индикатор седловых точек
		На выходе:
			Выводит значения этих пременных и строит визуализацию спектров.

	from nash_equilibrium.simplex_method import nash_equilibrium, print_result
		tab = np.array([
	    [4,0,6,2,2,1],
	    [3,8,4,10,4,4],
	    [1,2,6,5,0,0],
	    [6,6,4,4,10,3],
	    [10,4,6,4,0,9],
	    [10,7,0,7,9,8]
	])
	p, q, price, flag = nash_equilibrium(tab)
	print_result(p, q, price, flag)
