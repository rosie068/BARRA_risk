"""
Created on Wed Nov 13 2019
@author: RosemaryHe
"""

import pandas as pd
import numpy as np
import math
import sqlalchemy
import urllib

DB_CONN = 0         #1-有数据库连接，从数据库取数据；0-无连接，从csv文件取数据
inFilename = 'A.h5'
outFilename = 'BarraLiquidity.h5'

def CalBarraLiquidity(dates):
    Liquidity = ['STOM','STOQ','STOA','ATR']
    statemap = {'STOM':['trade_volumn','total_share'],'STOQ':['trade_volumn','total_share'],'STOA':['trade_volumn','total_share'],
        'ATR':['trade_volumn','total_share','close_price']}

    if DB_CONN == 1:
        #函数以数据库连接
        conn_params = urllib.parse.quote_plus("""DRIVER={SQL Server Native Client 10.0};
                                        SERVER=quant;DATABASE=tbas;UID=*****;PWD=********""")
        conn = sqlalchemy.create_engine("mssql+pyodbc:///?odbc_connect=%s" % conn_params)
        conn_params=urllib.parse.quote_plus("""DRIVER={SQL Server Native Client 10.0};
                                        SERVER=10.130.14.41;DATABASE=fcdb;UID=*****;PWD=********""")
        conn2 = sqlalchemy.create_engine("mssql+pyodbc:///?odbc_connect=%s" % conn_params)

    for factor in Liquidity:
        fcl = ['date'] + statemap[factor]
        #数据读入,市场,财务数据
        st = pd.HDFStore(inFilename)
        state = st.select('mkt',"columns="+str(fcl))
        mkt = st.select('mkt', "columns=['close_price','total_share']")
        st.close()

        if factor == 'STOM':
            tempvalue = state['trade_volumn'] / state['total_share']
            factorvalue = tempvalue.rolling(21).sum()
            factorvalue[factorvalue <= 0] = 1
            factorvalue = factorvalue.apply(lambda x: math.log(x))
        elif factor == 'STOQ':
            tempvalue = state['trade_volumn'] / state['total_share']
            factorvalue = tempvalue.rolling(3*21).sum() / 3
            factorvalue[factorvalue <= 0] = 1
            factorvalue = factorvalue.apply(lambda x: math.log(x))
        elif factor == 'STOA':
            tempvalue = state['trade_volumn'] / state['total_share']
            factorvalue = tempvalue.rolling(12*21).sum() / 12
            factorvalue[factorvalue <= 0] = 1
            factorvalue = factorvalue.apply(lambda x: math.log(x))
        elif factor == 'ATR':
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
            dividend = dividend.drop_duplicates(['date', 'sec_code'])  ###the date and bonus ratio for each divident for each stock
            dividend = dividend.set_index(['reg_date', 'sec_code'])
            dividend['share'] = state['total_share']
            dividend = dividend.dropna()
            dividend['dividend'] = dividend['bonus_ratio'] * dividend['share'] / 10.0
            dividend = dividend.reset_index()
            dividend = dividend.set_index(['date', 'sec_code'])

            div_only = dividend['dividend']
            mkt = mkt.merge(div_only, how='left', left_index=True, right_index=True).fillna(0)
            mkt['final'] = mkt['close_price'] * mkt['total_share'] + mkt['dividend']   ##最终的市值
            mkt = mkt.drop(columns=['close_price','total_share','dividend'])

            ##找到每个月的月底
            mkt['isMonthEnd'] = False
            dates = mkt.index.values
            mkt['date'] = [x[0] for x in dates]
            month_end_dates = []
            for i in range(len(mkt) - 1):
                if dates[i][0].month != dates[i+1][0].month:
                    mkt.iloc[i,1] = True
                    month_end_dates.append(dates[i][0])
            mkt.iloc[len(mkt)-1,1] = True
            month_end_dates.append(dates[len(mkt)-1][0])

            mkt['total_mkt'] = pd.Series()
            for i in range(len(month_end_dates)):
                temp = mkt[mkt.date == month_end_dates[i]]['final']
                total_sum = sum(temp.values)
                for j in range(len(mkt)):
                    if mkt.iloc[j,2] == month_end_dates[i]:
                        mkt.iloc[j,3] = total_sum

            ##把每一天的权重设成改月月底占市值权重
            mkt['weights'] = (mkt['final'] / mkt['total_mkt']).fillna(method='bfill')
            daily_value = state['trade_volumn'] / state['total_share']
            factorvalue = (daily_value * mkt['weights']).rolling(21*12).sum() / 12

        st = pd.HDFStore(outFilename)
        if factor in [x[1:] for x in st.keys()]:
            existday = st.select_column(factor, 'index')
            st.append(factor, factorvalue.loc[factorvalue.index.difference(existday)], format='t')
        else:
            st.append(factor, factorvalue, format='t')
        st.close()

        print(factor)
