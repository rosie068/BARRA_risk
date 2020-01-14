"""
Created on Wed Nov 13 2019
@author: RosemaryHe
"""

import pandas as pd
import numpy as np
import math
from sklearn import datasets, linear_model
import sqlalchemy
import urllib

def linear_regression(X,y):
    reg = linear_model.LinearRegression()
    reg.fit(X,y)
    return reg.coef_, reg.intercept_

def cal_Z(T, stock_rate, rf_rate):
    z = 0
    for i in range(T):
        z += math.log(stock_rate.iloc[i]) - math.log(rf_rate.iloc[i])
    return z

DB_CONN = 0         #1-有数据库连接，从数据库取数据；0-无连接，从csv文件取数据
inFilename = 'A.h5'
outFilename = 'BarraMomentum.h5'

def CalBarraVolatility(dates):
    Volatility = ['Beta','HSIGMA','DASTD','CMRA']
    statemap = {'Beta':['sec_return'],'HSIGMA':['sec_return'],'DASTD':['sec_return'],'CMRA':['sec_return']}

    if DB_CONN == 1:
        #函数以数据库连接
        conn_params = urllib.parse.quote_plus("""DRIVER={SQL Server Native Client 10.0};
                                        SERVER=quant;DATABASE=tbas;UID=*****;PWD=********""")
        conn = sqlalchemy.create_engine("mssql+pyodbc:///?odbc_connect=%s" % conn_params)
        conn_params=urllib.parse.quote_plus("""DRIVER={SQL Server Native Client 10.0};
                                        SERVER=10.130.14.41;DATABASE=fcdb;UID=*****;PWD=********""")
        conn2 = sqlalchemy.create_engine("mssql+pyodbc:///?odbc_connect=%s" % conn_params)

    ##to save time, we will use the regression results directly from Beta
    Beta_res_error = pd.DataFrame()

    for factor in Volatility:
        fcl = ['date'] + statemap[factor]

        #数据读入,市场,财务数据
        st = pd.HDFStore(inFilename)
        state = st.select('mkt',"columns="+str(fcl))
        if factor in ['Beta','HSIGMA']:
            lists = ['date','cash']
            tempa = st.select('sheet', "columns=" + str(lists))
            mkt = st.select('mkt', "columns=['close_price', 'total_share']")
        st.close()

        dates = state.index.get_level_values('date')
        stocks = state.index.get_level_values('sec_code')
        state['secID'] = stocks
        state['dates'] = dates
        dates = dates.drop_duplicates()
        stocks = stocks.drop_duplicates()

        ##setting risk_free to 0.1 for now, will change later
        if factor in ['Beta','HSIGMA','CMRA']:
            temp_arr = [0.1] * len(state['sec_return'])
            state['rf_return'] = temp_arr

        if factor == 'Beta':
        ##if factor in ['Beta','HSIGMA']:
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
                each_fact_res = abs(each_stock.iloc[:, 0] * 0)
                factor_arr = []
                res_arr = []
                for l in range(len(each_stock) - 1, 251, -1):
                    short_y = each_stock['sec_return'].iloc[l - 252:l] - each_stock['rf_return'].iloc[l - 252:l].values
                    short_x = each_stock['Rt'].iloc[l - 252:l].values
                    short_x = short_x * ((0.5 ** (1 / 63)) ** (252 - l))  ##指数加权
                    short_y = short_y * ((0.5 ** (1 / 63)) ** (252 - l))
                    x = np.asmatrix(short_x).transpose()
                    y = np.asmatrix(short_y).transpose()
                    b,a = linear_regression(x, y)
                    each_fact.iloc[l] = b

                    intercept_arr = [a] * len(x)
                    array = y - np.multiply(b,x) - intercept_arr
                    each_fact_res.iloc[l] = np.std(array)
                factor_arr.append(each_fact)
                res_arr.append(each_fact_res)
            factorvalue = pd.concat(factor_arr)
            Beta_res_error = pd.concat(res_arr)

        elif factor == 'HSIGMA':  ##must run Beta so only regress once,
            factorvalue = Beta_res_error

        elif factor == 'DASTD':
            weights_arr = [0.0] * 252
            for w in range(len(weights_arr)):
                weights_arr[w] = (0.5**(1/42))**(252-w)

            factor_arr = []
            for s in stocks:
                each_return = state[state.secID==s]
                each_return = each_return.set_index(['dates', 'secID'])
                each_factor = abs(each_return.iloc[:, 0] * 0)
                for i in range(len(each_return)-1, 251, -1):
                    temp_slice = each_return.iloc[i-252:i,0]
                    #temp_slice['weights'] = weights
                    temp_r = temp_slice * weights_arr
                    each_factor.iloc[i] = temp_r.std()
                factor_arr.append(each_factor)
            factorvalue = pd.concat(factor_arr)

        elif factor == 'CMRA':
            fact_array = []
            for s in stocks:
                each_stock = state[state.secID==s]
                fact = abs(each_stock.iloc[:,0] * 0)
                for k in range(len(each_stock)-1, 251, -1):
                    monthly_stock = pd.Series([0.0] * 12)
                    monthly_rf = pd.Series([0.0] * 12)
                    ##find the return of each month
                    for m in range(12):
                        each_month = each_stock.iloc[k - 252:k, :]
                        monthly_compound_stock = 1
                        monthly_compound_rf = 1
                        for i in range(21):
                            monthly_compound_stock = monthly_compound_stock * (1 + each_month.iloc[21*m+i,0] / 100)
                            monthly_compound_rf = monthly_compound_rf * (1 + each_month.iloc[21*m+i,3] / 100)
                        monthly_stock.iloc[m] = monthly_compound_stock - 1
                        monthly_rf.iloc[m] = monthly_compound_rf - 1
                    monthly_stock[monthly_stock <= -1] = 0
                    monthly_rf[monthly_rf <= -1] = 0
                    each_arr = [math.log(1 + monthly_stock.iloc[x]) - math.log(1 + monthly_rf.iloc[x]) for x in np.arange(12)]
                    each_Z = pd.Series(each_arr)
                    Z = []
                    for j in range(len(each_Z)):
                        if j == 0:
                            Z.append(each_Z[j])
                        else:
                            Z.append(Z[j-1] + each_Z[j])
                    fact.iloc[k] = np.max(Z) - np.min(Z)
                fact_array.append(fact)
            factorvalue = pd.concat(fact_array)


        st = pd.HDFStore(outFilename)
        if factor in [x[1:] for x in st.keys()]:
            existday = st.select_column(factor, 'index')
            st.append(factor, factorvalue.loc[factorvalue.index.difference(existday)], format='t')
        else:
            st.append(factor, factorvalue, format='t')
        st.close()

        print(factor)


