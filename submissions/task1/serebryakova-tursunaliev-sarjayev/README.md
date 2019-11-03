If jup.ipynb can't load you can see it [here](https://nbviewer.jupyter.org/github/CheenaT/game-theory--matrix-game/blob/task1-serebryakova-tursunaliev-sarjayev/submissions/task1/serebryakova-tursunaliev-sarjayev/jup.ipynb)

# Практическая работа №1

Команда "СТС": Серебрякова Софья, Турсуналиев Чингиз, Сарджаев Меред, 312 группа.

## Содержание:

1. Постановка задачи и подход к её решению
2. Инструкцией по запуску
3. Вклад каждого участника

## Постановка задачи

Постановка задачи состоит в численном решении антагонистической матричной игры.

## Инструкция по запуску

1. Установка [pip](https://pip.pypa.io/en/stable/installing/) если его нет:

* ### Linux или macOS

```sh
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py
```

* ### Windows

Скачать [get-pip.py](https://bootstrap.pypa.io/get-pip.py) в папку на вашем компьютере. Откройте окно командной строки и перейдите в папку, содержащую `get-pip.py`. Затем запустите `python get-pip.py`.

2. Скачать пакет и установить его:

```sh
cd superrnash-package
pip install .
```

3. Использование пакета:

```sh
python
>>> import superrnash
>>> suppernash.main()
2
2
5 0
6 0
```

<p align='center'>
  <img src='https://sun9-26.userapi.com/c857028/v857028320/349f5/RLZIvT0v63w.jpg' width='600' alt='npm start'>
</p>

<p align='center'>
  <img src='https://sun9-43.userapi.com/c857028/v857028320/349fd/xeNDQbxPAq0.jpg' width='600' alt='npm start'>
</p>

```sh
 Game value is : 0.0 
 optimal strategy for 1st player :  [1.0, 0.0] 
 optimal strategy for 2nd player :  [0.0, 1.0]
>>>
```

4. Запуск тестов:

```sh
nosetests nose.py
```

## Вклад каждого участника

* Серебрякова Софья реализовала функцию nash_equilibrium(a), которая принимает матрицу выигрыша и возвращает значение игры и оптимальные стратегии первого и второго игроков.

* Турсуналиев Чингиз оформил решение в виде пакета superrnash и реализовал функцию visualization(p), которая иллюстрирует работу функции nash_equilibrium(a) путем решения нескольких игр и визуализации спектров оптимальных стратегий игроков в Jupyter. В частности, были приведены игры в которых:
  + Спектр оптимальной стратегии состоит из одной точки (т.е. существует
равновесие Нэша в чистых стратегиях).
  + Спектр оптимальной стратегии неполон (т.е. некоторые чистые
стратегии не используются).
  + Спектр оптимальной стратегии полон.
* Сарджаев Меред написал unit-тесты для функции nash_equilibrium(a) с помощью утилиты nose.
