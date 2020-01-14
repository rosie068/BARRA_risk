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

##find the coefficient of linear regression
def linear_regression_coef(X,y):
    reg = linear_model.LinearRegression()
    reg.fit(X,y)
    return reg.coef_

DB_CONN = 0         #1-有数据库连接，从数据库取数据；0-无连接，从csv文件取数据
inFilename = 'A.h5'
outFilename = 'BarraGrowth.h5'

def CalBarraGrowth(dates):
    Growth = ['EGRLF','EGRSF','EGRO','SGRO']
    statemap = {'EGRLF':['对未来三年预期净利润','net_profit0'], ##数据缺失
        'EGRSF':['对未来一年预期净利润','net_profit0'], ##数据缺失
        'EGRO':['net_profit0'], ##年报每股收益 = 当期净利润/当期在外发行普通股
        'SGRO':['operate_profit','operate_expense']}

    if DB_CONN == 1:
        #函数以数据库连接
        conn_params = urllib.parse.quote_plus("""DRIVER={SQL Server Native Client 10.0};
                                        SERVER=quant;DATABASE=tbas;UID=*****;PWD=********""")
        conn = sqlalchemy.create_engine("mssql+pyodbc:///?odbc_connect=%s" % conn_params)
        conn_params=urllib.parse.quote_plus("""DRIVER={SQL Server Native Client 10.0};
                                        SERVER=10.130.14.41;DATABASE=fcdb;UID=*****;PWD=********""")
        conn2 = sqlalchemy.create_engine("mssql+pyodbc:///?odbc_connect=%s" % conn_params)

    for factor in Growth:
        fcl = ['date'] + statemap[factor]

        #数据读入,市场,财务数据
        st = pd.HDFStore(inFilename)
        state = st.select('sheet',"columns="+str(fcl))
        if factor in ['EGRO','SGRO']:
            mkt = st.select('mkt', "columns=['total_share']")
        st.close()

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

        nf = nf.drop(['pdate'], axis=1)

        ##日度对齐
        nf = nf.unstack()
        if factor in ['EGRO','SGRO']:
            nf = nf.reindex(nf.index.union(dates))
            mkt = mkt.unstack()
            mkt = mkt.reindex(mkt.index.union(dates)).ffill()
            mkt = mkt.reindex(dates)
        else:
            nf = nf.reindex(nf.index.union(dates)).ffill()
        nf = nf.reindex(dates)

        if factor in ['EGRLF','EGRSF']:
            factorvalue = nf[statemap[factor][0]] / (math.abs(nf[statemap[factor][1]]) - 1)

        elif factor in ['EGRO','SGRO']:
            ##temp5是要回归的数据
            if factor == 'EGRO':
                temp5 = nf['net_profit0'] / mkt['total_share']
            else:
                temp5 = (nf['operate_profit'] + nf['operate_expense']) / mkt['total_share']

            factorvalue = abs(temp5.copy(deep=True) * 0)
            for i in range(len(temp5.iloc[0, :])):
                stock_val = temp5.iloc[:, i]
                stock_val = stock_val.dropna(how='all')
                stock_fval = abs(stock_val.copy(deep=True) * 0)
                for j in range(len(stock_val)-1, 19, -1):    ##回归过去5年,就是20个季度
                    temp_y = stock_val.iloc[j-20:j].fillna(0)
                    temp_x = np.arange(20)
                    x = np.asmatrix(temp_x).transpose()
                    y = np.asmatrix(temp_y).transpose()
                    B = linear_regression_coef(x, y)
                    mean = y.mean()
                    if mean == 0:
                        stock_fval.iloc[j] = B
                    else:
                        stock_fval.iloc[j] = B / mean
                stock_fval = stock_fval.reindex(stock_fval.index.union(dates)).ffill()
                factorvalue.iloc[:,i] = stock_fval

        st = pd.HDFStore(outFilename)
        if factor in [x[1:] for x in st.keys()]:
            existday = st.select_column(factor, 'index')
            st.append(factor, factorvalue.loc[factorvalue.index.difference(existday)], format='t')
        else:
            st.append(factor, factorvalue, format='t')
        st.close()
    
        print(factor)




