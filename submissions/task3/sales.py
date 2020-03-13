import pandas as pd
import datetime as dt
import numpy as np

def daily(store_name):
    goods_recieved = pd.read_csv(source_path + store_name + supply)
    goods_sold = pd.read_csv(source_path + store_name + sell)
    daily_stats = pd.DataFrame({'date' : [dt.date(2006, 1, 1)], 'apple': [0], 'pen': [0]})
    daily_stats['apple_sold'] = [0]
    daily_stats['pen_sold'] = [0]
    daily_stats.reset_index()
    pens = apples = 0
    index = 0
    index_income = 0
    current_date = dt.date(2006, 1, 1)
    for i in range(len(goods_sold['date']) - 1):
        if (index_income < len(goods_recieved['date'])) and (goods_recieved['date'][index_income] == goods_sold['date'][i]):
            daily_stats.loc[index, 'apple'] += goods_recieved['apple'][index_income]
            daily_stats.loc[index, 'pen'] += goods_recieved['pen'][index_income]
            #print("Recieved apples ",goods_recieved['apple'][index_income])
            #print("Recieved pens ",goods_recieved['pen'][index_income])
            index_income += 1
        if goods_sold['sku_num'][i].find("-ap-") != -1:
            apples += 1
        elif goods_sold['sku_num'][i].find("-ap-") == -1:
            pens += 1
        if goods_sold['date'][i] != goods_sold['date'][i + 1]:
            daily_stats.loc[index, 'date'] = current_date
            daily_stats.loc[index, 'apple'] -= apples
            daily_stats.loc[index, 'pen'] -= pens
            daily_stats.loc[index, 'apple_sold'] = apples
            daily_stats.loc[index, 'pen_sold'] = pens
            index += 1
            current_date = current_date + dt.timedelta(days = 1)
            apples = 0
            pens = 0
            daily_stats.loc[index] = [current_date, daily_stats['apple'][index - 1], daily_stats['pen'][index -1], daily_stats['apple_sold'][index - 1], daily_stats['pen_sold'][index - 1]]
    print("> Daily stats are ready")
    i = len(goods_sold['date']) - 1
    if goods_sold['sku_num'][i].find("-ap-") != -1:
        apples += 1
    elif goods_sold['sku_num'][i].find("-ap-") == -1:
        pens += 1
    daily_stats.loc[index, 'apple'] -= apples
    daily_stats.loc[index, 'pen'] -= pens
    daily_stats_sold = daily_stats.copy()
    daily_stats_sold.drop(['apple', 'pen'], axis = 'columns', inplace = True)
    daily_stats.drop(['apple_sold', 'pen_sold'], axis = 'columns', inplace = True)
    daily_stats.set_index('date', inplace = True)
    print(daily_stats)
    daily_stats.to_csv(out_path + store_name + "-daily.csv")
    print("> Saved daily stats")
    return daily_stats_sold

def stolen(store_name):
    daily_stats = pd.read_csv(out_path + store_name + "-daily.csv")
    monthly_inventory = pd.read_csv(source_path + store_name + inventory)
    monthly_stolen = pd.DataFrame({'date' : [dt.date(2006, 1, 31)], 'apple': [0], 'pen': [0]})
    index = 0
    total_stolen_apple = total_stolen_pen = 0
    for i in range(len(daily_stats['date'])):
        if (index < len(monthly_inventory['date'])) and (daily_stats['date'][i] == monthly_inventory['date'][index]):
            monthly_stolen.loc[index, 'date'] = daily_stats['date'][i]
            monthly_stolen.loc[index, 'apple'] = daily_stats['apple'][i] - monthly_inventory['apple'][index] - total_stolen_apple
            monthly_stolen.loc[index, 'pen'] = daily_stats['pen'][i] - monthly_inventory['pen'][index] - total_stolen_pen
            total_stolen_apple = total_stolen_apple + monthly_stolen['apple'][index]
            total_stolen_pen = total_stolen_pen + monthly_stolen['pen'][index]
            index += 1
            monthly_stolen.loc[index] = [monthly_stolen['date'][index - 1], 0, 0]
    monthly_stolen.drop([index], inplace = True)
    monthly_stolen.set_index('date', inplace = True)
    print("> Stolen stats are ready")
    print(monthly_stolen)
    monthly_stolen.to_csv(out_path + store_name + "-steal.csv")
    print("> Saved stolen stats")

def yearly(store_name, sales_info):
    monthly_stolen = pd.read_csv(out_path + store_name + "-steal.csv")
    index_yearly = index_stolen = 0
    yearly_sold_apples = yearly_sold_pens = yearly_stolen_apples = yearly_stolen_pens = 0
    date_point = sales_info['date'][0]
    yearly_stats = pd.DataFrame({'year' : [dt.date(2006, 1, 1).strftime("%Y")], 'state' : ["MS"], 'apple_sold' : [0], 'apple_stolen' : [0], 'pen_sold' : [0], 'pen_stolen' : [0]})
    for i in range(len(sales_info['date']) - 1):
        yearly_sold_apples += sales_info['apple_sold'][i]
        yearly_sold_pens += sales_info['pen_sold'][i]
        if pd.to_datetime(monthly_stolen['date'][index_stolen]) == sales_info['date'][i]:
            yearly_stolen_apples += monthly_stolen['apple'][index_stolen]
            yearly_stolen_pens += monthly_stolen['pen'][index_stolen]
            index_stolen += 1
        year_check = date_point.year % 4
        if sales_info['date'][i + 1].year != sales_info['date'][i].year:
            yearly_stats.loc[index_yearly] = [sales_info['date'][i].strftime("%Y"), "MS", yearly_sold_apples, yearly_stolen_apples, yearly_sold_pens, yearly_stolen_pens]
            index_yearly += 1
            yearly_sold_apples = yearly_sold_pens = yearly_stolen_apples = yearly_stolen_pens = 0
    i = len(sales_info['date']) - 1
    yearly_sold_apples += sales_info['apple_sold'][i]
    yearly_sold_pens += sales_info['pen_sold'][i]
    if monthly_stolen['date'][index_stolen] == sales_info['date'][i]:
        yearly_stolen_apples += monthly_stolen['apple'][index_stolen]
        yearly_stolen_pens += monthly_stolen['pen'][index_stolen]
        index_stolen += 1
    yearly_stats.loc[index_yearly] = [sales_info['date'][i].strftime("%Y"), "MS", yearly_sold_apples, yearly_stolen_apples, yearly_sold_pens, yearly_stolen_pens]
    yearly_stats.set_index('year', inplace = True)
    print("> Yearly stats are ready")
    print(yearly_stats)
    yearly_stats.to_csv(out_path + store_name + "-yearly.csv")
    print("> Saved yearly stats")

#M A I N
source_path = "ref/out/input/"
out_path = "res/"
names = np.array(["MS-b1", "MS-b2", "MS-m1", "MS-m2", "MS-s1", "MS-s2", "MS-s3", "MS-s4", "MS-s5"])
inventory = "-inventory.csv"
sell = "-sell.csv"
supply = "-supply.csv"

for i in range(len(names)):
    goods_sold = daily(names[i])
    stolen(names[i])
    yearly(names[i], goods_sold)
