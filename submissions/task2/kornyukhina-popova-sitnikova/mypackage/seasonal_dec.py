import pandas as pd
import numpy as np
import itertools
import warnings
warnings.filterwarnings('ignore')
from .visualization import *

def seasonal_component(df, n, model='additive'):
    Values = df.reset_index().copy()
    Values = centered_rolling_mean(rolling_mean(df,n))
    if (model == 'additive'):
        Values["Value"] = df["Value"][int(n / 2):df["Value"].size - int(n / 2)] - Values["Value"]
    else:
        Values["Value"] = df["Value"][int(n / 2):df["Value"].size - int(n / 2)] / Values["Value"]
    #средняя оценка сезонной компоненты    
    seas_com = np.zeros(n) 
    period = int(Values["Value"].size / n) 
    Values = Values.reset_index()
    for i in range(0, n):
        seas_com[i] = Values["Value"][i:Values["Value"].size:n].sum() / period
    #циклический сдвиг массива
    from collections import deque
    d = deque(seas_com)
    d.rotate(period)
    seas_com = np.array(list(d))
    Values = Values.set_index("Date")
    correction_factor = seas_com.sum() / n if model == 'additive' else n / seas_com.sum()
    adjusted_SC = seas_com - correction_factor if model == 'additive' else seas_com * correction_factor
    return adjusted_SC,Values

#Метод наименьших квадратов для нахождения коэффициентов T
def least_square_method(df, adjusted_SC, n):
    Values = df.reset_index().copy()
    for i in range(0, Values["Value"].size, n):
        for j in range(n):
            Values["Value"][j + i] -= adjusted_SC[j] 
    t = pd.DataFrame({ 
                    "Date": np.array(range(1,Values["Value"].size + 1)),
                    "Value":np.array(range(1,Values["Value"].size + 1))
    })
    t.set_index("Date") 
    n = Values["Value"].size
    const = n * (t["Value"] * t["Value"]).sum() - t["Value"].sum() * t["Value"].sum()
    a_1 = (n * ((Values["Value"] * t["Value"]).sum()) - Values["Value"].sum() * t["Value"].sum()) / const
    a_0 = (Values["Value"].sum() - a_1 * t["Value"].sum()) / n 
    return a_1, a_0

def get_trend(df, n):
    #Тренд
    #T=a_0+a_1*t
    SC, trend = seasonal_component(df, n)
    a_1, a_0 = least_square_method(df, SC, n)
    Values = df.copy()
    t = pd.DataFrame({ 
                        "Date": np.array(range(1,Values["Value"].size + 1)),
                        "Value":np.array(range(1,Values["Value"].size + 1))
    })
    t.reset_index()
    for i in range(Values["Value"].size):
        Values["Value"][i] = a_0 + a_1 * t["Value"][i]
    return Values

def seasonality(df, n, model='additive'):
    mod = df["Value"].size % n
    SC, value = seasonal_component(df, n, model)
    season = pd.Series(np.zeros(df["Value"].size))
    for i in range(0, df["Value"].size - mod, n):
        season[i:i + n] = SC[0:n]
    if (mod):
        season[df["Value"].size - mod: df["Value"].size] = SC[:mod]
    season_df = df.reset_index().copy()
    season_df["Value"] = season[:]
    return season_df

def seasonal_decompose(df, model='additive', trend='roll'):
    if (model == 'additive'):
        observed = df.copy()
        trend = rolling_mean(df, 12) if (trend == 'roll') else get_trend(df, 12) 
        seasonal = seasonality(df, 12).set_index("Date")
        resid = (df - (trend + seasonal)).dropna()
    if (model == 'multiplicative'):
        observed = df.copy()
        trend = rolling_mean(df, 12) if (trend == 'roll') else get_trend(df, 12) 
        seasonal = seasonality(df, 12, model='multiplicative').set_index("Date")
        resid = (df / (trend * seasonal)).dropna()
    return observed, trend, seasonal, resid

