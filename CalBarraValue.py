"""
Created on Wed Nov 13 2019
@author: RosemaryHe
"""

import numpy as np
import pandas as pd
import sqlalchemy 
import urllib
import math
from sklearn import datasets, linear_model

def linear_regression_int(X,y):
    reg = linear_model.LinearRegression()
    reg.fit(X,y)
    return reg.intercept_

def cal_RS(t, df):
    RS = 0
    temp_stock = df['sec_return']
    temp_rf = df['rf_return']
    temp_stock[temp_stock <= -1] = 0
    temp_rf[temp_rf <= -1] = 0
    for i in range(1,1009):
        RS = RS + (0.5**(1/504))**(1008-i) * (math.log(1 + temp_stock.iloc[t-i-273]) - math.log(1 + temp_rf.iloc[t-i-273]))
    return RS

#文件名称
DB_CONN = 0         #1-有数据库连接，从数据库取数据；0-无连接，从csv文件取数据
inFilename = 'A.h5'
outFilename = 'Value.h5'

def CalBarraValue(dates):
    ##Value = ['BTOP','ETOP','EPIBS','CETOP','ENMU','LTRSTR','LTHALPHA']
    Value = ['LTHALPHA']
    statemap = {'BTOP':['total_asset','total_liab'],  ##net_asset = total_asset - total_liab
        'ETOP':['net_profit0'],
        'EPIBS':['未来12个月预测归母净利润'], ##先不计算,数据缺失
        'CETOP':['net_profit0'], ##'depreciation_amortization', 先不计算,数据缺失
        'ENMU':['EBITDA','cash','total_liab'], ##企业价值 = 市值 + total_liab - cash
        'LTRSTR':['sec_return'],'LTHALPHA':['sec_return']}

    if DB_CONN == 1:
        #函数以数据库连接
        conn_params = urllib.parse.quote_plus("""DRIVER={SQL Server Native Client 10.0};
                                        SERVER=quant;DATABASE=tbas;UID=*****;PWD=********""")
        conn = sqlalchemy.create_engine("mssql+pyodbc:///?odbc_connect=%s" % conn_params)
        conn_params=urllib.parse.quote_plus("""DRIVER={SQL Server Native Client 10.0};
                                        SERVER=10.130.14.41;DATABASE=fcdb;UID=*****;PWD=********""")
        conn2 = sqlalchemy.create_engine("mssql+pyodbc:///?odbc_connect=%s" % conn_params)

    for factor in Value:
        fcl = ['date'] + statemap[factor]

        #数据读入,市场,财务数据
        st = pd.HDFStore(inFilename)
        if factor == 'LTRSTR':
            state = st.select('mkt', "columns=" + str(fcl))
        elif factor == 'LTHALPHA':
            state = st.select('mkt', "columns=" + str(fcl))
            mkt = st.select('mkt', "columns=['date','close_price','total_share']")
            lists = ['date'] + ['cash']
            tempa = st.select('sheet',"columns="+str(lists))
        else:
            mkt = st.select('mkt', "columns=['close_price','total_share']")
            state = st.select('sheet',"columns="+str(fcl))
        st.close()

        ##set risk_free to 0.1 for now for ALL THE DATES, change if data available
        if factor in ['LTRSTR','LTHALPHA']:
            state['rf_return'] = pd.Series(len(state['sec_return']))
            state['rf_return'] = state['rf_return'].fillna(0.1)

        if factor in ['BTOP','ETOP','EPIBS','CETOP','ENMU']:
            #分红数据，处理市值用.
            if DB_CONN==1:
                dividend = pd.read_sql("""select sec_code,cast(ex_date as varchar) as date,cast(reg_date as varchar) as reg_date,bonus_ratio 
                                          from tbas..tCOM_dividend where div_type = 1""",
                                          con=conn, index_col=['date', 'sec_code'], parse_dates=['date', 'reg_date'])
            else:
                dividend = pd.read_csv(r'data/datadividend.csv', index_col=None, parse_dates=['date', 'reg_date'], encoding='gbk')
                dividend['sec_code'] = [('00000'+str(x))[-6:] for x in dividend['sec_code']]
                dividend = dividend.set_index(['date', 'sec_code'])
            dividend = dividend.reset_index()
            dividend = dividend.drop_duplicates(['date', 'sec_code'])  ###the date and bonus ratio for each divident for each stock
            dividend = dividend.set_index(['reg_date', 'sec_code'])
            dividend['share'] = mkt['total_share']
            dividend = dividend.dropna()
            dividend['dividend'] = dividend['bonus_ratio'] * dividend['share'] / 10.0
            dividend = dividend.reset_index()
            dividend = dividend.set_index(['date', 'sec_code'])

            #因子计算,财务数据对齐
            ##财务数据对齐
            state = state.unstack()
            state = state[(state.index.month.isin([3,6,9,12]))].stack()

            nf = state.reset_index()
            tnf = nf['pdate'].groupby([nf['date'],nf['sec_code']]).max()
            nf = nf.set_index(['date', 'sec_code', 'pdate'])
            tnf = tnf.reset_index()
            tnf = tnf.set_index(['date', 'sec_code', 'pdate'])
            nf = nf[nf.index.isin(tnf.index)]
            nf = nf.reset_index('pdate')
            ##财报数据发布后，剔除分红
            dividend = dividend['dividend'].reindex(nf.index.union(dividend.index)).fillna(0)

            nf = nf.drop(['pdate'],axis=1)
            ##日度对齐
            nf = nf.unstack()
            nf = nf.reindex(nf.index.union(dates)).ffill()
            nf = nf.reindex(dates)
            nf = nf.stack()
            dividend = dividend.unstack()
            dividend = dividend.reindex(dividend.index.union(dates)).ffill()
            dividend = dividend.reindex(dates)

            #估值因子
            mkt = mkt.unstack()
            mkt = mkt.reindex(mkt.index.union(dates)).ffill()
            mkt = mkt.reindex(dates)
            tcap = mkt['close_price'] * mkt['total_share']
            tcap[tcap == 0] = np.nan

        if factor == 'BTOP':
            factorvalue = (nf['total_asset'] - nf['total_liab']) / (tcap + dividend)
        elif factor == 'ETOP':
            factorvalue = (nf['net_profit0'].rolling(252).sum() / 63) / (tcap + dividend)
        elif factor == 'EPIBS':
            factorvalue = nf['未来12月预测'] / (tcap + dividend)
        elif factor == 'CETOP':
            factorvalue = ((nf['net_profit0'] + nf['depreciation_amortization']).rolling(252).sum() / 63) / (tcap + dividend)
        elif factor == 'ENMU':
            factorvalue = (nf['EBITDA'].rolling(252).sum() / 63) / (tcap + dividend + nf['total_liab'] - nf['cash'])
        elif factor == 'LTRSTR':
            temp_arr = [0] * len(state['sec_return'])
            factorvalue = pd.Series(temp_arr)
            for i in range(len(state['sec_return'])-1, 1008+272, -1):
                factorvalue.iloc[i] = cal_RS(i,state)

        elif factor == 'LTHALPHA':
            dates = state.index.get_level_values('date')
            stocks = state.index.get_level_values('sec_code')
            state['secID'] = stocks
            state['dates'] = dates
            dates = dates.drop_duplicates()
            stocks = stocks.drop_duplicates()

            # 分红数据，处理市值用.
            if DB_CONN == 1:
                dividend = pd.read_sql("""select sec_code,cast(ex_date as varchar) as date,cast(reg_date as varchar) as reg_date,bonus_ratio 
                                                                  from tbas..tCOM_dividend where div_type = 1""",
                                       con=conn, index_col=['date', 'sec_code'], parse_dates=['date', 'reg_date'])
            else:
                dividend = pd.read_csv(r'data/datadividend.csv', index_col=None, parse_dates=['date', 'reg_date'],
                                       encoding='gbk')
                dividend['sec_code'] = [('00000' + str(x))[-6:] for x in dividend['sec_code']]
                dividend = dividend.set_index(['date', 'sec_code'])
            dividend = dividend.reset_index()
            dividend = dividend.drop_duplicates(
                ['date', 'sec_code'])  ###the date and bonus ratio for each divident for each stock
            dividend = dividend.set_index(['reg_date', 'sec_code'])
            dividend['share'] = mkt['total_share']
            dividend = dividend.dropna()
            dividend['dividend'] = dividend['bonus_ratio'] * dividend['share'] / 10.0
            dividend = dividend.reset_index()
            dividend = dividend.set_index(['date', 'sec_code'])

            # 4. 因子计算
            # 财务数据对齐
            tempa = tempa.unstack()
            tempa = tempa[(tempa.index.month.isin([3, 6, 9, 12]))].stack()

            nf = tempa.reset_index()
            tnf = nf['pdate'].groupby([nf['date'], nf['sec_code']]).max()
            nf = nf.set_index(['date', 'sec_code', 'pdate'])
            tnf = tnf.reset_index()
            tnf = tnf.set_index(['date', 'sec_code', 'pdate'])
            nf = nf[nf.index.isin(tnf.index)]
            nf = nf.reset_index('pdate')

            ##财报数据发布后，剔除分红
            dividend = dividend['dividend'].reindex(nf.index.union(dividend.index)).fillna(0)

            ##日度对齐
            nf = nf.unstack()
            nf = nf.reindex(nf.index.union(dates)).ffill()
            nf = nf.reindex(dates)
            dividend = dividend.unstack()
            dividend = dividend.reindex(dividend.index.union(dates)).ffill()
            dividend = dividend.reindex(dates)

            # 估值因子
            mkt = mkt.unstack()
            mkt = mkt.reindex(mkt.index.union(dates)).ffill()
            mkt = mkt.reindex(dates)
            tcap = mkt['close_price'] * mkt['total_share']
            tcap[tcap == 0] = np.nan

            total_weight = pd.DataFrame(dates, columns=['date'])
            total_weight['Rt'] = pd.Series()
            for i in range(len(dates)):
                temp_return = state[state.dates == dates[i]]
                temp_tcap = pd.DataFrame(tcap.iloc[i, :])
                temp_tcap['secID'] = temp_tcap.index.values
                weights = temp_tcap[temp_tcap['secID'].isin(temp_return.secID.values)]
                temp_weight = weights.iloc[:, 0]
                temp_weight = temp_weight / temp_weight.sum()
                total_weight.iloc[i, 1] = (temp_weight * temp_return['sec_return']).sum()
            total_weight = total_weight.dropna()
            state = state[state.dates.isin(total_weight.date.values)]
            new_state = pd.merge(state, total_weight, left_on='dates', right_on='date', how='outer').fillna(0)

            for s in range(len(stocks)):
                each_stock = new_state[new_state.secID == stocks[s]]
                each_stock = each_stock.set_index(['dates', 'secID'])
                each_fact = abs(each_stock.iloc[:, 0] * 0)
                factor_arr = []
                for l in range(len(each_stock) - 1, 1280, -1):
                    short_y = each_stock['sec_return'].iloc[l - 273 - 1008:l] - each_stock['rf_return'].iloc[l - 273 - 1008:l].values  ##向前推13个月, 再用48个月
                    short_x = each_stock['Rt'].iloc[l - 273 - 1008:l].values
                    short_x = short_x * ((0.5 ** (1 / 63)) ** (252 - l))  ##将数据进行指数加权
                    short_y = short_y * ((0.5 ** (1 / 63)) ** (252 - l))
                    x = np.asmatrix(short_x).transpose()
                    y = np.asmatrix(short_y).transpose()
                    b = linear_regression_int(x, y)
                    each_fact.iloc[l] = b
                factor_arr.append(each_fact)
            factorvalue = pd.concat(factor_arr)

        st = pd.HDFStore(outFilename)
        if factor in [x[1:] for x in st.keys()]:
            existday = st.select_column(factor, 'index')
            st.append(factor, factorvalue.loc[factorvalue.index.difference(existday)], format='t')
        else:
            st.append(factor, factorvalue, format='t')
        st.close()
    
        print(factor)





