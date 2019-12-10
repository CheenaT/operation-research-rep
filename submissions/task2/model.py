import numpy as np
import pandas as pd
import statsmodels.api as sm
import xlrd as xl
import matplotlib.pyplot as plt
import statistics as st
import seaborn as sns
from math import factorial
from statsmodels.tsa.seasonal import seasonal_decompose as sdecomp
from statsmodels.tsa.adfvalues import mackinnonp, mackinnoncrit
from statsmodels.compat.python import iteritems
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from pandas import datetime, DataFrame
from pandas.core.nanops import nanmean as pd_nanmean
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from statsmodels.tools import add_constant
import warnings

sns.set(style="darkgrid")


def permutation(m, n):
    return factorial(n) / (factorial(n - m) * factorial(m))


def diff_operator(set, k):
    size = len(set)
    sum = set[0] * (-1)**(size-1)
    for i in range(1, size):
        minus_counter = (-1)**(size-i-1)
        sum = sum + permutation(i, size-1)*minus_counter*set[i]
    return sum


def AIC_finder(test, predict, k, p, counter, q):
    if (counter > 0):
        c = 0
    else:
        c = 1
    n = p + q + c + 1
    predict = np.array(predict)
    ressid = test - predict.ravel()
    sse = sum(ressid**2)
    AIC = n*np.log(sse/n)+2*k
    return AIC


def arima_optimizer_AIC(training, testing, p, max_k, q, flag):
    print(training)
    print()
    print(testing)
    min_AIC_lib = min_our_AIC = 99999
    min_p_lib = min_p_our = 0
    min_q_lib = min_q_our = 0
    for i in range(p+1):
        for j in range(q+1):
            print("ARIMA(%d,%d,%d)" % (i, max_k, j))
            if flag == 0:
                AIC, our_AIC, error, r2, test, predict = arima_learn_forecast(
                    training, testing, i, max_k, j)
                r2_score = r2
            else:
                AIC, our_AIC, error, r2, test, predict = arima_learn_predict(
                    training, testing, i, max_k, j)
                r2_score = r2
            if (AIC < min_AIC_lib):
                min_AIC_lib = AIC
                min_test_lib = test
                min_predict_lib = predict
                min_p_lib = i
                min_q_lib = j
                r2_score_lib = r2_score
                error_lib = error
            if (our_AIC < min_our_AIC):
                min_our_AIC = our_AIC
                min_test_our = test
                min_predict_our = predict
                min_p_our = i
                min_q_our = j
                r2_score_our = r2_score
                error_our = error
    print()
    print("Best ARIMA by loglikehood AIC = ARIMA(%d,%d,%d)" %
          (min_p_lib, max_k, min_q_lib))
    print("Minimal loglikehood AIC = ", min_AIC_lib)
    print('Test MSE: %.3f' % error_lib)
    print("r2 score: ", r2_score_lib)
    print()
    print("Best ARIMA by SSE AIC = ARIMA(%d,%d,%d)" %
          (min_p_our, max_k, min_q_our))
    print("Minimal SSE AIC = ", min_our_AIC)
    print('Test MSE: %.3f' % error_our)
    print("r2 score: ", r2_score_our)
    plt.plot(min_test_lib)
    if (flag == 1):
        x = range(360, 360+len(min_predict_lib))
        plt.plot(x, min_predict_lib, color='red')
    else:
        plt.plot(min_predict_lib, color='red')
    plt.show()


def arima_learn_forecast(training, testing, p, max_k, q):
    warnings.simplefilter('ignore')
    x = training.values
    size = int(len(training.values))
    train, test = training.values, testing.values
    history = [x for x in train]
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=(p, max_k, q))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
        print('predicted=%f, expected=%f' % (yhat, obs))
    error = mean_squared_error(test, predictions)
    print('Test MSE: %.3f' % error)
    r2 = r2_score(test, predictions)
    print("r2 score: ", r2)
    regr = OLS(test, predictions).fit()
    our_aic = AIC_finder(test, predictions, len(test), p, max_k, q)
    print("OLS AIC: ", regr.aic)
    print("Our AIC: ", our_aic)
    return regr.aic, our_aic, error, r2, test, predictions


def arima_learn_predict(training, testing, p, max_k, q):
    warnings.simplefilter('ignore')
    x = training.values
    size = int(len(training.values))
    train, test = training.values, testing.values
    history = [x for x in train]
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=(p, max_k, q))
        model_fit = model.fit(disp=0)
        output = model_fit.predict(start=len(train), end=(
            len(train)+len(test)-1), typ='levels')
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(yhat)
        print('predicted=%f, expected=%f' % (yhat, obs))
    error = mean_squared_error(test, predictions)
    print('Test MSE: %.3f' % error)
    r2 = r2_score(test, predictions)
    print("r2 score: ", r2)
    x = range(len(train), len(train)+len(test))
    print(len(x), len(train))
    regr = OLS(test, predictions).fit()
    our_aic = AIC_finder(test, predictions, len(test), p, max_k, q)
    print("OLS AIC: ", regr.aic)
    print("Our AIC: ", our_aic)
    return regr.aic, our_aic, error, r2, np.append(train, test), predictions


def best_train_finder(training, p, max_k, q):
    X = training.values
    size = int(len(X) * 0.1)
    train, test = X[0:size], X[size:len(X)]
    history = [x for x in train]
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=(5, 1, 0))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
        print('predicted=%f, expected=%f' % (yhat, obs))
    error = mean_squared_error(test, predictions)
    print('Test MSE: %.3f' % error)
    print("r2 score: ", r2_score(test, predictions))
    # plot
    plt.plot(test)
    plt.plot(predictions, color='red')
    plt.show()


def integral_definer(df):
    values = df
    oper_values = np.array([0]).astype(float)
    counter = 0
    flag = 0
    max_k = 0
    for k in range(1, len(values)):
        if flag:
            continue
        for i in range(1, len(values)+1):
            if ((i-k-1) >= 0):
                oper_values = np.append(oper_values, 0)
                oper_values[i-1] = diff_operator(values[i-k-1:i], k)
        oper_values_cutted = oper_values[k:len(
            oper_values)+(1-k)*(len(values)-k) + counter]
        counter = counter - (k-1)
        print(k, " test: ")
        if df_test(oper_values_cutted):
            flag = 1
            max_k = k
            ans = oper_values_cutted
    return max_k, ans


def avg_data(df):  # скользящая средняя
    rows, columns = df.shape
    averages = []
    averages.append(df['Value'][0])
    for i in range(1, rows):
        elem = 0
        for j in range(i):
            elem = elem + df['Value'][j]
        elem = elem/i
        averages.append(elem)
    return averages


def white_noise(df):  # первые разности
    rows, columns = df.shape
    noise = []
    noise.append(df['Value'][0])
    for i in range(1, rows):
        noise.append(df['Value'][i]-df['Value'][i-1])
    noise[0] = noise[1]
    return noise


def get_lag(mod, endog, exog, start_lag, max_lag, method, model_args=()):
    results = {}  # dict
    method = method.lower()
    for lag in range(start_lag, start_lag + max_lag + 1):
        results[lag] = mod(endog, exog[:, :lag], *model_args).fit()
    if method == "aic":
        best_inf_crit, best_lag = min((v.aic, k) for k, v in iteritems(
            results))  # перебор по значениям из results
    return best_inf_crit, best_lag

def df_test(df):
    df_vect = df
    df_size = len(df_vect)
    autolag = 'AIC'
    max_lag = None
    regression = 'c'
    trend_size = len(regression)  # размер тренда
    # Максимальное запаздывание, вычисляется как ТВГ соотвествующего выражения
    max_lag = int(np.ceil(12. * np.power(df_size / 100., 1/2)))
    max_lag = min(df_size // 2 - trend_size, max_lag)
    if max_lag < 0:
        raise ValueError('Dataset is too short')
    # массив с первыми разностями: элем_i = a[i+1] - a[i]
    df_diff = np.diff(df_vect)
    # массив с лагами, где max_lag - число "сдвигов" вниз
    df_diff_all = sm.tsa.lagmat(
        df_diff[:, None], max_lag, trim='both', original='in')
    df_size = df_diff_all.shape[0]  # количество столбцов в массиве лагов
    # заменяем первый столбец df_diff_all на df_vect
    df_diff_all[:, 0] = df_vect[-df_size - 1:-1]
    df_diff_short = df_diff[-df_size:]  # оставляем последние df_size элементов
    df_diff_full = df_diff_all
    start_lag = df_diff_full.shape[1] - \
        df_diff_all.shape[1] + 1  # начальный лаг
    best_inf_crit, best_lag = get_lag(
        sm.OLS, df_diff_short, df_diff_full, start_lag, max_lag, autolag)
    best_lag -= start_lag  # оптимальное значение лага
    # массив с лагами, но уже при оптимальном значении лага
    df_diff_all = sm.tsa.lagmat(
        df_diff[:, None], best_lag, trim='both', original='in')
    df_size = df_diff_all.shape[0]
    # заменяем первый столбец df_diff_all на df_vect
    df_diff_all[:, 0] = df_vect[-df_size - 1:-1]
    df_diff_short = df_diff[-df_size:]
    use_lag = best_lag
    # аппроксимация ряда методом наименьших квадратов
    resols = sm.OLS(df_diff_short, sm.tsa.add_trend(
        df_diff_all[:, :use_lag + 1], regression)).fit()
    adfstat = resols.tvalues[0]  # получение необходимой статистики
    pvalue = mackinnonp(adfstat, regression=regression, N=1)
    critvalues = mackinnoncrit(N=1, regression=regression, nobs=df_size)
    if adfstat < critvalues[1]:
        print("Time series is stationary with crit value ", adfstat)
        return True
    else:
        print("Time series is not stationary with crit value ", adfstat)
        return False


def series_seasonal(df, window):
    seasonal = np.array([pd_nanmean(df[i::window], axis=0)
                         for i in range(window)])
    return seasonal


def series_decompose_sum(df, window):  # Аддитивная модель
    avg = df.Value.rolling(window=30).mean()  # trend, но по-другому
    no_trend = df.Value - avg
    seasonal = series_seasonal(no_trend, 30)
    seasonal = seasonal - np.mean(seasonal, axis=0)
    size = no_trend.shape[0]
    season = np.tile(seasonal.T, size // window + 1).T[:size]  # window = 30
    df['Season1'] = season
    sea_son = df.Season1
    residual = df.Value - avg - season
    return avg, sea_son, residual


def series_decompose_mul(df, window):  # Мультипликативная модель
    avg = df.Value.rolling(window=30).mean()  # trend, но по-другому
    no_trend = df.Value/avg
    seasonal = series_seasonal(no_trend, 30)
    seasonal = seasonal - np.mean(seasonal, axis=0)
    size = no_trend.shape[0]
    season = np.tile(seasonal.T, size // window + 1).T[:size]  # window = 30
    df['Season2'] = season
    sea_son = df.Season2
    residual = df.Value - avg - season
    return avg, sea_son, residual


# MAIN
training = pd.read_excel('training.xlsx')
training_decomp = pd.read_excel('training.xlsx')
print(training.columns)  # названия столбов

# добавляем новый столбец в наш dataframe
training['Average'] = avg_data(training)
resid = [training['Average'][0]]
# добавляем новый столбец в наш dataframe
training['Noise'] = white_noise(training)
for i in range(1, len(training['Average'])):
    resid.append(training['Average'][i] -
                 training['Average'][i - 1] - training['Noise'][i])
resid[0] = resid[1]
training['Residual'] = resid  # добавляем новый столбец в наш dataframe
decomp_avg_add, decomp_season_add, decomp_resid_add = series_decompose_sum(
    training, 30)
decomp_avg_mult, decomp_season_mult, decomp_resid_mult = series_decompose_mul(
    training, 30)

# Рисуем красивые графики
fig = plt.figure(figsize=(20, 10), num='Time Series Decomposition')
ax1 = fig.add_subplot(321, title="Additive model", ylabel="Trend")
ax2 = fig.add_subplot(322, title="Multiplicative model")
ax3 = fig.add_subplot(323, ylabel="Season")
ax4 = fig.add_subplot(324)
ax5 = fig.add_subplot(325, ylabel="Residue")
ax6 = fig.add_subplot(326)
sns.lineplot(data=decomp_avg_add, ax=ax1)
sns.lineplot(data=decomp_season_add, ax=ax3)
sns.lineplot(data=decomp_resid_add, ax=ax5)
sns.lineplot(data=decomp_avg_mult, ax=ax2)
sns.lineplot(data=decomp_season_mult, ax=ax4)
sns.lineplot(data=decomp_resid_mult, ax=ax6)

plt.show()

print("Our test:")
df_test(training['Value'])
print("Library test:")
# Проверяем рабочесть нашего теста Дики-Фуллера на библиотечном
print(sm.tsa.adfuller(training['Value']))

print()

# Поиск порядка интегрируемости
values = training['Value'].to_numpy()
max_k, training_max_k = integral_definer(values)
print("Порядок интегрируемости: ", max_k)

# Ввод необходимый для работы ARIMA
training = pd.read_excel('training.xlsx', header=0, parse_dates=[
                         0], index_col=0, squeeze=True)
testing = pd.read_excel('testing.xlsx', header=0, parse_dates=[
                        0], index_col=0, squeeze=True)

# Тут мы определяем параметры ARMA модели (p,q)
# Для нашей модели надо проверить p = 0,1,2 и q = 0,1,2,3,4
plt.figure(figsize=(8, 8))
plt.subplot(211)
plot_acf(training_max_k, ax=plt.gca())
plt.subplot(212)
plot_pacf(training_max_k, ax=plt.gca())
plt.show()

# Модель обучается и предсказывает
arima_optimizer_AIC(training, testing, 1, 1, 4, 0)
best_train_finder(training, 1, max_k, 4)
