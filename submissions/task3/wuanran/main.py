import pandas as pd

df_supply = pd.read_csv('out_input_MS-b1-supply.csv')
print(df_supply.head())

df_inventory = pd.read_csv('out_input_MS-b1-inventory.csv')
print(df_inventory.head())

df_sell = pd.read_csv('out_input_MS-b1-sell.csv')
print(df_sell.head())

state = df_sell.sku_num[0][:2]

total_apples = 0
total_pens = 0
res_df = pd.DataFrame(columns=['date', 'apple', 'pen'])

from_ind = 0
to_ind = 0
prev_date = '2006-01-01'

for cur_sell in df_sell.values:
    str_d = cur_sell[0]
    code = cur_sell[1][6:8]

    if prev_date != str_d:
        sup = df_supply[df_supply.date == prev_date]

        if sup.shape[0] == 1:
            total_apples += int(sup.apple)
            total_pens += int(sup.pen)

        cur_df = pd.DataFrame(columns=['date', 'apple', 'pen'], data=[[prev_date, total_apples, total_pens]])
        res_df = res_df.append(cur_df)
    prev_date = str_d

    if code == 'ap':
        total_apples -= 1
    else:
        total_pens -= 1

sup = df_supply[df_supply.date == str_d]
if sup.shape[0] == 1:
    total_apples += int(sup.apple)
    total_pens += int(sup.pen)
cur_df = pd.DataFrame(columns=['date', 'apple', 'pen'], data=[[str_d, total_apples, total_pens]])
res_df = res_df.append(cur_df)
res_df.to_csv('daily.csv',index=False)

stolen_df = pd.DataFrame(columns=['date', 'apple', 'pen'])

total_stolen_apples = 0
total_stolen_pens = 0

for x in df_inventory.values:
    cur_d = x[0]
    paper_apples = x[1]
    paper_pens = x[2]
    real_inv = res_df[res_df.date == x[0]]
    real_apples = int(real_inv.apple)
    real_pens = int(real_inv.pen)
    cur_apples = real_apples - paper_apples - total_stolen_apples
    cur_pens = real_pens - paper_pens - total_stolen_pens
    cur_df = pd.DataFrame(columns=['date', 'apple', 'pen'], data=[[cur_d, cur_apples, cur_pens]])
    total_stolen_apples += cur_apples
    total_stolen_pens += cur_pens
    stolen_df = stolen_df.append(cur_df)
stolen_df.to_csv('stolen.csv',index=False)

cols = ['year', 'state', 'apple_sold', 'apple_stolen', 'pen_sold', 'pen_stolen']
year_df = pd.DataFrame(columns=cols)

cur_year = 2006
supply_pens = 0
supply_apples = 0
ind = 0

for x in df_supply.values:
    cur_date = x[0]
    supply_apples += x[1]
    supply_pens += x[2]

    if ind % 24 == 23:
        cur_date = cur_date[:-2] + '31'
        apple_sold = supply_apples - int(res_df[res_df.date == cur_date].apple)
        pen_sold = supply_pens - int(res_df[res_df.date == cur_date].pen)
        pen_stolen = int(res_df[res_df.date == cur_date].pen) - int(df_inventory[df_inventory.date == cur_date].pen)
        apple_stolen = int(res_df[res_df.date == cur_date].apple) - int(
            df_inventory[df_inventory.date == cur_date].apple)
        cur_df = pd.DataFrame(columns=cols, data=[[cur_year, state, apple_sold, apple_stolen, pen_sold, pen_stolen]])
        year_df = year_df.append(cur_df)
        cur_year += 1
    ind += 1
year_df.to_csv('stats.csv',index=False)
