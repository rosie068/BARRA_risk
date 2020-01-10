"""
Created on Wed Nov 13 2019
@author: heyuan
"""

import pandas as pd
import numpy as np
import math
import datetime
import sqlalchemy
import urllib
from sklearn import datasets, linear_model

def linear_regression_int(X,y):
    reg = linear_model.LinearRegression()
    reg.fit(X,y)
    return reg.intercept_

##calculate RS according to BARRA Handbook
def cal_RS(t, sec_return, rf_return, length, halflife):
    RS = 0
    sec_return[sec_return <= -1] = 0
    rf_return[rf_return <= -1] = 0
    for i in range(1,length):
        RS = RS + (0.5**(1/halflife))**(length-i) * (math.log(1 + sec_return.iloc[t-i]) - math.log(1 + rf_return.iloc[t-i]))
    return RS

DB_CONN = 0         #1-有数据库连接，从数据库取数据；0-无连接，从csv文件取数据
inFilename = 'A.h5'
outFilename = 'BarraMomentum.h5'

def CalBarraMomentum(dates):
    Momentum = ['STREV','Seasonality','INDMOM','RSTR','HALPHA']
    statemap = {'STREV':['sec_return'],'Seasonality':['sec_return'], 'INDMOM':['sec_return'],'RSTR':['sec_return'],
        'HALPHA':['sec_return']}

    if DB_CONN == 1:
        #函数以数据库连接
        conn_params = urllib.parse.quote_plus("""DRIVER={SQL Server Native Client 10.0};
                                        SERVER=quant;DATABASE=tbas;UID=quant;PWD=quant007""")
        conn = sqlalchemy.create_engine("mssql+pyodbc:///?odbc_connect=%s" % conn_params)
        conn_params=urllib.parse.quote_plus("""DRIVER={SQL Server Native Client 10.0};
                                        SERVER=10.130.14.41;DATABASE=fcdb;UID=ch_data;PWD=68880980""")
        conn2 = sqlalchemy.create_engine("mssql+pyodbc:///?odbc_connect=%s" % conn_params)

    for factor in Momentum:
        fcl = ['date'] + statemap[factor]

        #3. 数据读入
        #市场、财务数据
        st = pd.HDFStore(inFilename)
        state = st.select('mkt', "columns="+str(fcl))
        if factor in ['HALPHA','INDMOM']:
            lists = ['date'] + ['cash']
            tempa = st.select('sheet', "columns=" + str(lists))
            mkt = st.select('mkt', "columns=['close_price', 'total_share']")
        st.close()

        if factor in ['STREV', 'INDMOM', 'RSTR', 'HALPHA']:
            ##set the risk_free return to be 0.1 for everyday, can change if data available
            state['rf_return'] = pd.Series()
            state['rf_return'] = state['rf_return'].fillna(0.1)

        if factor in ['INDMOM','HALPHA']:
            dates = state.index.get_level_values('date')
            stocks = state.index.get_level_values('sec_code')
            state['secID'] = stocks
            state['dates'] = dates
            dates = dates.drop_duplicates()
            stocks = stocks.drop_duplicates()

        if factor == 'STREV':
            sec_return = state['sec_return']
            rf_return = state['rf_return']
            RS = abs(sec_return.copy(deep=True) * 0.0)
            factorvalue = abs(sec_return.copy(deep=True) * 0.0)

            for i in range(len(sec_return.iloc[0,:])):
                each_sec = sec_return.iloc[:,i]
                each_df = rf_return.iloc[:,i]
                for k in range(len(each_sec)-1, 62, -1):
                    RS.iloc[k,i] = cal_RS(k, each_sec, each_df, 63, 10)
                factorvalue.iloc[:,i] = RS.rolling(3).mean()

        elif factor == 'Seasonality':
            factorvalue = abs(state.copy(deep=True) * 0.0)

            for i in range(len(state.iloc[0, :])):
                each_stock = abs(state.iloc[:,i].copy(deep=True) * 0.0)
                mean_on_day = abs(each_stock.copy(deep=True) * 0.0)
                for j in range(len(each_stock) - 23, 0, -1):
                    mean_on_day.iloc[j] = state.iloc[j+1:j+22,i].mean()
                for k in range(len(mean_on_day)-1, 251, -1):
                    t = k
                    y = 1
                    temp_sum = 0
                    while t > 0 and y < 6:
                        temp_sum += mean_on_day.iloc[t]
                        t = t - y * 252
                        y = y + 1
                    each_stock.iloc[k] = temp_sum / y
                factorvalue.iloc[:,i] = each_stock

        elif factor == 'INDMOM':
            # 分红数据
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

            # 行业信息
            if DB_CONN == 1:
                sector = pd.read_sql("""select t1.sec_code,cast(t1.enter_date as varchar) as date,t1.sector_code,t2.group_code
                                                      from tbas..tSECTOR_config t1, tbas..tSECTOR_group_config t2
                                                      where t1.class_code=%d and t2.group_class=%d and t2.sector_code=t1.sector_code
                                                      order by t1.sec_code"""
                                         % (class_code, group_class),
                                         con=conn, index_col=['date', 'sec_code'], parse_dates=['date'])
            else:
                sector = pd.read_csv(r'data/datasector.csv', index_col=None, parse_dates=[0],encoding='gbk')  ##行业分类
                sector['sec_code'] = [('00000' + str(x))[-6:] for x in sector['sec_code']]  ##get their sec_ID code, end up with 000001,000003,...
                sector = sector.set_index(['date', 'sec_code'])

            new_state = pd.merge(state, sector, left_index=True, right_index=True, how='outer')
            new_state[['sector_code', 'group_code']] = new_state[['sector_code', 'group_code']].fillna(method='ffill')
            new_state = new_state.dropna()

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

            industry = new_state.group_code.drop_duplicates()

            ##计算RS
            temp_array = []
            for i in range(len(stocks)):
                each_stock = new_state[new_state.secID==stocks[i]]
                sec_ret = each_stock.sec_return
                rf_ret = each_stock.rf_return
                RS = abs(sec_ret.copy(deep=True) * 0)

                for j in range(len(each_stock)-1, 125, -1):
                    RS.iloc[j] = cal_RS(j, sec_ret, rf_ret, 126, 21)
                each_stock['RS'] = RS

                each_stock['tcap'] = tcap.loc[:,stocks[i]].values
                temp_array.append(each_stock)
            all_data = pd.concat(temp_array)

            final = []
            for j in range(len(industry)):
                each_ind = all_data[all_data.group_code==industry[j]]
                for k in range(len(dates)):
                    each_day = each_ind[each_ind.dates==dates[k]]
                    each_day['weight'] = each_day['tcap'] / each_day['tcap'].sum()
                    iRS = (each_day['RS'] * each_day['weight']).sum()
                    each_day['ind_RS'] = [iRS] * len(each_day)
                    final.append(each_day)
            final_data = pd.concat(final)
            factorvalue = final_data['RS'] - final_data['weight'] * final_data['ind_RS']

        elif factor == 'RSTR':
            sec_return = state['sec_return']
            rf_return = state['rf_return']
            factorvalue = abs(sec_return.copy(deep=True) * 0.0)

            RS = abs(sec_return.copy(deep=True) * 0.0)
            for i in range(len(sec_return.iloc[0, :])):
                each_sec = sec_return.iloc[:, i]
                each_rf = rf_return.iloc[:, i]

                for k in range(len(each_sec) - 1, 503, -1):
                    RS.iloc[k, i] = cal_RS(k, each_sec, each_rf, 504, 126)
                factorvalue.iloc[:, i] = RS.shift(10).rolling(11).mean()

        elif factor == 'HALPHA':
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
            dividend = dividend.unstack()
            dividend = dividend.reindex(dividend.index.union(dates)).ffill()
            dividend = dividend.reindex(dates).fillna(0)

            # 估值因子
            mkt = mkt.unstack()
            mkt = mkt.reindex(mkt.index.union(dates)).ffill()
            mkt = mkt.reindex(dates)
            tcap = mkt['close_price'] * mkt['total_share'] + dividend
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

            ##回归得alpha
            for s in range(len(stocks)):
                each_stock = new_state[new_state.secID == stocks[s]]
                each_stock = each_stock.set_index(['dates', 'secID'])
                each_fact = abs(each_stock.iloc[:, 0] * 0)
                each_fact_res = abs(each_stock.iloc[:, 0] * 0)
                factor_arr = []
                res_arr = []
                for l in range(len(each_stock) - 1, 251, -1):
                    short_y = each_stock['sec_return'].iloc[l - 252:l] - each_stock['rf_return'].iloc[l - 252:l].values
                    short_x = each_stock['Rt'].iloc[l - 252:l].values
                    short_x = short_x * ((0.5 ** (1 / 63)) ** (252 - l))
                    short_y = short_y * ((0.5 ** (1 / 63)) ** (252 - l))
                    x = np.asmatrix(short_x).transpose()
                    y = np.asmatrix(short_y).transpose()
                    a = linear_regression_int(x, y)
                    each_fact.iloc[l] = a
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

