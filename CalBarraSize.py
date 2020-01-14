"""
Created on Wed Nov 13 2019
@author: RosemaryHe
"""

import pandas as pd
import numpy as np
import math
import sqlalchemy
import urllib
from sklearn import datasets, linear_model

def linear_regression_coef(X,y):
    reg = linear_model.LinearRegression()
    reg.fit(X,y)
    return reg.coef_

DB_CONN = 0         #1-有数据库连接，从数据库取数据；0-无连接，从csv文件取数据
inFilename = 'A.h5'
outFilename = 'BarraSize.h5'

def CalBarraSize(dates):
    Size = ['Size','MidCap']
    statemap = {'Size':['close_price','total_share'],'MidCap':['close_price','total_share']}

    if DB_CONN == 1:
        #函数以数据库连接
        conn_params = urllib.parse.quote_plus("""DRIVER={SQL Server Native Client 10.0};
                                        SERVER=quant;DATABASE=tbas;UID=*****;PWD=********""")
        conn = sqlalchemy.create_engine("mssql+pyodbc:///?odbc_connect=%s" % conn_params)
        conn_params=urllib.parse.quote_plus("""DRIVER={SQL Server Native Client 10.0};
                                        SERVER=10.130.14.41;DATABASE=fcdb;UID=*****;PWD=********""")
        conn2 = sqlalchemy.create_engine("mssql+pyodbc:///?odbc_connect=%s" % conn_params)

    st = pd.HDFStore(inFilename)
    state = st.select('mkt', "columns=['close_price', 'total_share']")
    st.close()

    #分红数据
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
    dividend['share'] = state['total_share']
    dividend = dividend.dropna()
    dividend['dividend'] = dividend['bonus_ratio'] * dividend['share'] / 10.0
    dividend = dividend.reset_index()
    dividend = dividend.set_index(['date', 'sec_code'])

    div_only = dividend['dividend']
    state = state.merge(div_only, how='left',left_index=True, right_index=True).fillna(0)

    #Calc Size factor
    factorvalue1 = (state['close_price'] * state['total_share'] + state['dividend']).fillna(1)
    factorvalue1[factorvalue1.values <= 0] = 1
    factorvalue1 = factorvalue1.apply(lambda x: math.log(x))

    ##Calc MidCap factor
    factorvalue2 = factorvalue1.copy(deep=True)
    size = factorvalue2.values

    ##standardize
    mean = size.mean()
    std = size.std()
    size = (size-mean) / std

    #winsorize
    s_plus = max(0, min(1, 0.5 / (max(size) - 3)))
    s_minus = max(0, min(1, 0.5 / (-3 - min(size))))
    for i in range(len(size)):
        if size[i] > 3:
            size[i] = 3 * (1 - s_plus) + size[i] * s_plus
        elif size[i] < -3:
            size[i] = -3 * (1 - s_minus) + size[i] * s_minus

    #regress size^3 = b * size + e
    size_cubed = size**3
    x = np.asmatrix(size).transpose()
    y = np.asmatrix(size_cubed).transpose()
    slope = linear_regression_coef(x, y)
    mid_cap_val = size_cubed - slope * size
    for i in range(len(mid_cap_val[0])):
        factorvalue2.iloc[i] = mid_cap_val[0][i]

    st = pd.HDFStore(outFilename)
    st.append('Size', factorvalue1, format='t')
    st.append('MidCap', factorvalue2, format='t')
    st.close()
