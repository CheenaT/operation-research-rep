import matplotlib.pyplot as plt
import itertools
import warnings
warnings.filterwarnings('ignore')

#Визуализация ряда
def plot_series(DataFrame, a=1, x=15, y=5, color='blue', title='Time series'): 
    plt.figure(figsize=(x, y)) 
    if a == 1:
        plt.plot(DataFrame['Value'], color=color, label='Date', linewidth=2, markersize=15)  
    else:
        plt.plot(DataFrame['Value_pred'], color='red', label='Value_pred', linewidth=2, markersize=15)  
        plt.plot(DataFrame['Value'], color='blue', label='Value', linewidth=2, markersize=15)  
        
    plt.legend(loc='upper left')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.show()

#Скользящие статистики
def rolling_mean(df, n):
    if (n == 2):
         print("Error::n")
    stride = int(n / 2) if (n % 2) else int(n / 2) - 1
    delete = 2 * stride if (n % 2) else stride + 2
    Values = df.reset_index().copy()
    for i in range(df["Value"].size - delete):
        Values["Value"][i + stride] = df.reset_index()["Value"][i:n + i].sum() / n
    Values = Values.drop(range(Values["Value"].size - 1, Values["Value"].size - delete, -1))
    Values = Values.drop(range(stride))
    Values = Values.set_index("Date")
    return Values
def centered_rolling_mean(df):
    Values = df.copy()
    Values = Values.reset_index()
    for i in range(df["Value"].size):
        Values["Value"][i + 1] = df["Value"][i:2 + i].sum() / 2
    Values = Values.reset_index()
    Values = Values.drop(0)
    Values = Values.set_index("Date")
    del Values["index"]
    return Values
def rolling_std(df, n):
    stride = int(n / 2) if (n % 2) else int(n / 2) - 1
    Values = df.reset_index().copy()
    Mean_Values = (rolling_mean(df, n)).reset_index().copy()
    stds = (rolling_mean(df, n)).reset_index().copy()
    for i in range(0, Mean_Values["Value"].size):
        for j in range(i, i + n):
            x = (Values["Value"][j] - Mean_Values["Value"][i]) ** 2
        stds["Value"][i] = (x/(n-1)) ** 0.5
    return rolling_mean(stds, 5)

