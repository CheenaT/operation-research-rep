import pandas as pd
import re

path = '~/studies/submissions/task3/'
sell = pd.read_csv(path + 'MS-s5-sell.csv',
                   index_col='date',  parse_dates = True)
inventory = pd.read_csv(path + 'MS-s5-inventory.csv',
                        index_col='date', parse_dates = True)
supply = pd.read_csv(path + 'MS-s5-supply.csv',
                     index_col='date', parse_dates = True)
sell.sku_num = sell.sku_num.str.contains('MS-..-ap', regex = True)
shops = ['s4','s5']
states = ['MS']

sell['pen'] = ~sell.sku_num

df = sell.groupby('date').sum()
df.rename(columns={'sku_num': 'apple'}, inplace = True)
df.apple *= -1
df.pen *= -1

idx = pd.date_range('01-01-2006', '12-31-2015')
supply = supply.reindex(idx, fill_value = 0)
df += supply
n = 0
for i in range(len(df)):
    if(i>0):
        df.apple[i] += df.apple[i-1]
        df.pen[i] += df.pen[i-1]
    if(df.index[i] in inventory.index):
        stolen_apple = df.apple[i] - inventory.apple[n]
        stolen_pen = df.pen[i] - inventory.pen[n]
        df.apple[i] = inventory.apple[n]
        df.pen[i] = inventory.pen[n]
        inventory.apple[n] = stolen_apple
        inventory.pen[n] = stolen_pen
        n += 1
print(df)
print(inventory)

data_inv = pd.DataFrame()
data_sell= pd.DataFrame()
data_sup = pd.DataFrame()

for state in states[:]:
    for shop in shops[:]:
        print(path+state+'-'+shop+'-inventory.csv')
        data_inv  = data_inv.append(pd.read_csv(path+state+'-'+shop+
                                    '-inventory.csv', index_col='date',
                                    parse_dates=True).assign(state=state))
        data_sell = data_sell.append(pd.read_csv(path+state+'-'+shop+
                                     '-sell.csv', index_col='date',
                                     parse_dates=True).assign(state=state))
        data_sup  = data_sup.append(pd.read_csv(path+state+'-'+shop+
                                    '-supply.csv', index_col='date',
                                    parse_dates=True).assign(state=state))


data_inv = data_inv.sort_index().groupby(['date', 'state'])\
                                .sum().reset_index()\
                                .set_index('date')
data_sup = data_sup.sort_index().groupby(['date', 'state'])\
                                .sum().reset_index()\
                                .set_index('date')


data_sell.sku_num = data_sell.sku_num.str\
                             .contains('MS-..ap', regex = True)
data_sell.rename(columns={'sku_num': 'apple'}, inplace = True)
data_sell['pen'] = ~data_sell.apple



df = data_sell\
  .groupby(['date', 'state'])\
  .agg('sum')\
  .reset_index().set_index('date')


data_tmp = data_sup\
          .append(data_inv.iloc[-1])\
          .resample('1d').asfreq(fill_value=0)
data_tmp.iloc[-1,:] = 0
data_tmp.iloc[[0,1,2,-3,-2,-1]]

data_ans = data_tmp.copy()
for i in range(len(data_ans)):
  data_ans.apple += data_tmp.apple.shift(i+1, fill_value = 0)
  data_ans.apple -= df.apple.shift(i, fill_value = 0)

  data_ans.pen += data_tmp.apple.shift(i+1, fill_value = 0)
  data_ans.pen -= df.pen.shift(i, fill_value = 0)

for i in range(1,len(data_ans)):
  if data_ans.iloc[i,0] == 0:
    data_ans.iloc[i,0] = data_ans.iloc[i-1,0]
data_stl = data_ans.resample('1m').asfreq().iloc[:,[1,2]] - data_inv.iloc[:,[1,2]]
data_stl -= data_stl.shift(1, fill_value = 0)

data_ans['apple_stolen'] = data_stl.apple.astype(int)
data_ans['pen_stolen'] = data_stl.pen.astype(int)


data_ans = data_ans.rename({'apple': 'apple_sold', 'pen': 'pen_sold'},axis = 1).resample('y').asfreq()

column_suquence = ['state', 'apple_sold', 'apple_stolen', 'pen_sold', 'pen_stolen']
data_ans = data_ans[column_suquence]

print(data_ans)
