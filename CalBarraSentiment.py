"""
Created on Wed Nov 13 2019
@author: hey
"""

import pandas as pd
import numpy as np
import math
import sqlalchemy
import urllib

DB_CONN = 0         #1-有数据库连接，从数据库取数据；0-无连接，从csv文件取数据
inFilename = 'A.h5'
outFilename = 'BarraSentiment.h5'

def CalBarraSentiment(dates):
    Sentiment = ['EPIBSC','EARNC']  ##Sentiment里还有一项'RRIBS',目前由于数据缺失不计算
    statemap = {'EPIBSC':['EPIBS'],'EARNC':['EARN']} ##EPIBS = 预测EP, EARN = 预测每股收益

    if DB_CONN == 1:
        #函数以数据库连接
        conn_params = urllib.parse.quote_plus("""DRIVER={SQL Server Native Client 10.0};
                                        SERVER=quant;DATABASE=tbas;UID=quant;PWD=quant007""")
        conn = sqlalchemy.create_engine("mssql+pyodbc:///?odbc_connect=%s" % conn_params)
        conn_params=urllib.parse.quote_plus("""DRIVER={SQL Server Native Client 10.0};
                                        SERVER=10.130.14.41;DATABASE=fcdb;UID=ch_data;PWD=68880980""")
        conn2 = sqlalchemy.create_engine("mssql+pyodbc:///?odbc_connect=%s" % conn_params)

    for factor in Sentiment:
        fcl = ['date'] + statemap[factor]

        #数据读入,市场,财务数据
        st = pd.HDFStore(inFilename)
        state = st.select('sheet',"columns="+str(fcl))
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

        ##日度对齐
        nf = nf.unstack()
        nf = nf.reindex(nf.index.union(dates)).ffill()
        nf = nf.reindex(dates)

        if factor in ['EPIBSC','EARNC']:
            temp = nf[statemap[factor][0]]
            factorvalue = (temp - temp.shift(63)) / math.abs(temp.shift(63))

        st = pd.HDFStore(outFilename)
        if factor in [x[1:] for x in st.keys()]:
            existday = st.select_column(factor, 'index')
            st.append(factor, factorvalue.loc[factorvalue.index.difference(existday)], format='t')
        else:
            st.append(factor, factorvalue, format='t')
        st.close()
    
        print(factor)

