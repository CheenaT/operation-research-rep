from statsmodels.tsa.arima_model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller as ADF
filename = 'training.xlsx'
forrecastnum = 5
data = pd.read_excel(filename, index_col=u'Date')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
data.plot()
plt.title('Time Series')
plt.savefig("timeseries.png")
plot_acf(data)
plt.savefig("1.png")
print(u'原始序列的ADF检验结果为：', ADF(data[u'Value']))
D_data = data.diff(periods=1).dropna()
D_data.columns = [u'Value差分']
D_data.plot()
plt.savefig("2.png")
plot_acf(D_data).show()
plot_pacf(D_data).show()
print(u'1阶差分序列的ADF检验结果为：', ADF(D_data[u'Value差分']))
print(u'差分序列的白噪声检验结果为：', acorr_ljungbox(D_data, lags=1))
data[u'Value'] = data[u'Value'].astype(float)
pmax = int(len(D_data) / 10)
qmax = int(len(D_data) / 10)
bic_matrix = []
for p in range(pmax + 1):
    tmp = []
    for q in range(qmax + 1):
        try:
            tmp.append(ARIMA(data, (p, 1, q)).fit().bic)
        except BaseException:
            tmp.append(None)
    bic_matrix.append(tmp)
bic_matrix = pd.DataFrame(bic_matrix)
print(bic_matrix)
p, q = bic_matrix.stack().idxmin()
print(u'bic最小的P值和q值为：%s、%s' % (p, q))
model = ARIMA(data, (p, 1, q)).fit()
model.summary2()
forecast = model.forecast(5)
print(forecast)
