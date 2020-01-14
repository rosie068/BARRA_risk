"""
Created on Wed Nov 13 2019
@author: RosemaryHe
"""

import pandas as pd
import numpy as np
import sqlalchemy
import urllib
import math

DB_CONN = 0         #1-有数据库连接，从数据库取数据；0-无连接，从csv文件取数据
inFilename = 'A.h5'
outFilename = 'BarraDividendYield.h5'

def CalBarraDividendYield(dates):
    DividendYield = ['DividendYield','DPIBS']
    statemap = {'DividendYield':['cash_dividend'], ##现金分红
                'DPIBS':['未来12个月预测每股股息']} ##暂无数据

    if DB_CONN == 1:
        #函数以数据库连接
        conn_params = urllib.parse.quote_plus("""DRIVER={SQL Server Native Client 10.0};
                                        SERVER=quant;DATABASE=tbas;UID=*****;PWD=********""")
        conn = sqlalchemy.create_engine("mssql+pyodbc:///?odbc_connect=%s" % conn_params)
        conn_params=urllib.parse.quote_plus("""DRIVER={SQL Server Native Client 10.0};
                                        SERVER=10.130.14.41;DATABASE=fcdb;UID=*****;PWD=********""")
        conn2 = sqlalchemy.create_engine("mssql+pyodbc:///?odbc_connect=%s" % conn_params)

    for factor in DividendYield:
        fcl = ['date'] + statemap[factor]

        #数据读入,市场,财务数据
        st = pd.HDFStore(inFilename)
        mkt = st.select('mkt',"columns=['close_price','total_share']")
        state = st.select('sheet',"columns="+str(fcl))
        st.close()

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

        ##日度对齐
        nf = nf.unstack()
        nf = nf.reindex(nf.index.union(dates)).ffill()
        nf = nf.reindex(dates)
        dividend = dividend.unstack()
        dividend = dividend.reindex(dividend.index.union(dates)).ffill()
        dividend = dividend.reindex(dates)

        #估值因子
        mkt = mkt.unstack()
        mkt = mkt.reindex(mkt.index.union(dates)).ffill()
        mkt = mkt.reindex(dates)  
        tcap = mkt['close_price'] * mkt['total_share']
        tcap[tcap == 0] = np.nan

        if factor == 'DividendYield':
            factorvalue = ((nf['cash_dividend'].rolling(252).sum() / 63) / (tcap + dividend)) * 100
        elif factor == 'DPIBS':
            factorvalue = nf['未来12个月预测每股股息'] / mkt['close_price']

        ##数据输出
        st = pd.HDFStore(outFilename)
        if factor in [x[1:] for x in st.keys()]:
            existday = st.select_column(factor, 'index')
            st.append(factor, factorvalue.loc[factorvalue.index.difference(existday)], format='t')
        else:
            st.append(factor, factorvalue, format='t')
        st.close()

        print(factor)



