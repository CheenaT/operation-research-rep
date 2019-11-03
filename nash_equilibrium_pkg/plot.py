#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np


def plot_output(lis,color):
    x=np.linspace(1,len(lis),len(lis))
    y=lis
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Визуализация спектров оптимальных стратегий')
    plt.plot(x, y, marker='o', linestyle='', color=color)
    plt.savefig("image.png")
    pass


