"""
Created on Wed Nov 13 2019
@author: HeYuan
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
outFilename = 'BarraQuality.h5'

def CalBarraQuality(dates):
    Quality = ['MLEV','BLEV','DTOA','VSAL','VERN','VFLO','SPIBS','CETOE','ACBS','ACCF','ATO','GP','GM','ROA','AGRO','IGRO','CXGRO']
    statemap = {'MLEV':['total_long_liab'],'BLEV':['total_asset','total_liab'],'DTOA':['total_long_liab','total_equity'],
        'VSAL':['operate_in_total'],'VERN':['net_profit0','operate_in_total'],'VFLO':['operate_net_cash','operate_in_total'],
        'SPIBS':['未来12个月每股收益预测值标准差'],   ##暂无数据
        'CETOE':['depreciation_amortization'],    ##暂无数据
        'ACBS':['total_cur_asset','cash','total_cur_liab','short_borrow','total_asset'],   ##'depreciation_amortization'暂无数据
        'ACCF':['inventory','total_cur_liab','total_asset'],   ##'account_rec','depreciation_amortization'暂无数据
        'ATO':['operate_profit','operate_expense','total_asset'], #operate_income = operate_profit + operate_exp
        'GP':['operate_profit','total_asset'],
        'GM':['operate_profit','operate_expense'],'ROA':['ROA'],
        'AGRO':['total_asset'],'IGRO':['total_equity'],
        'CXGRO':['capital_expenditure']} ##capital expenditure = 购置各种长期资产支出 - 无息长期负债  暂无数据

    if DB_CONN == 1:
        #函数以数据库连接
        conn_params = urllib.parse.quote_plus("""DRIVER={SQL Server Native Client 10.0};
                                        SERVER=quant;DATABASE=tbas;UID=quant;PWD=quant007""")
        conn = sqlalchemy.create_engine("mssql+pyodbc:///?odbc_connect=%s" % conn_params)
        conn_params=urllib.parse.quote_plus("""DRIVER={SQL Server Native Client 10.0};
                                        SERVER=10.130.14.41;DATABASE=fcdb;UID=ch_data;PWD=68880980""")
        conn2 = sqlalchemy.create_engine("mssql+pyodbc:///?odbc_connect=%s" % conn_params)


    for factor in Quality:
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

        for x in statemap[factor]:
            if x in ['total_cur_asset','cash','total_cur_liab','short_borrow','inventory']: ##'depreciation_amortization', 先不算
                dat = state[x].unstack()
                cdat = dat - dat.shift(4)
                state['delta'+x] = cdat.stack()

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
        nf = nf.drop(['pdate'], axis=1)
        nf = nf.unstack()
        if factor in ['AGRO', 'IGRO', 'CXGRO']:
            nf = nf.reindex(nf.index.union(dates))
        else:
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

        if factor == 'MLEV':
            factorvalue = (nf['total_long_liab'] + tcap + dividend) / (tcap + dividend)
        elif factor == 'BLEV':
            factorvalue = nf['total_liab'] / nf['total_asset']
        elif factor == 'DTOA':
            factorvalue = (nf['total_long_liab'] + nf['total_equity']) / nf['total_equity']
        elif factor == 'VSAL':
            factorvalue = nf['operate_in_total'].rolling(756).std() / nf['operate_in_total'].rolling(756).mean()
        elif factor == 'VERN':
            factorvalue = (nf['net_profit0'] / nf['operate_in_total']).rolling(756).std()
        elif factor == 'VFLO':
            factorvalue = (nf['operate_net_cash'] / nf['operate_in_total']).rolling(756).std()
        elif factor == 'SPIBS':
            factorvalue = nf['未来12个月每股收益预测值标准差'] / nf['stock_price']
        elif factor == 'CETOE':
            factorvalue = (nf['depreciation_amortization'].rolling(252).sum() / 63) / mkt['close_price']
        elif factor == 'ACBS':
            factorvalue = (nf['delta'+'total_cur_asset'] - nf['delta'+'cash'] - nf['delta'+'total_cur_liab'] +
                nf['delta'+'short_borrow']) / nf['total_asset']
            ##factorvalue = (nf['delta'+'total_cur_asset'] - nf['delta'+'cash'] - nf['delta'+'total_cur_liab'] + nf['delta'+'short_borrow'] - nf['depreciation_amortization']) / nf['total_asset']
        elif factor == 'ACCF':
            factorvalue = (nf['delta' + 'inventory'] - nf['delta' + 'total_cur_liab']) / nf['total_asset']
            ##factorvalue = (nf['delta'+'account_rec'] + nf['delta'+'inventory'] - nf['delta'+'account_pay'] - nf['delta'+'tax_pay'] - nf['delta'+'other_current_liab_asset'] - nf['depreciation_amortization']) / nf['total_asset']
        elif factor == 'ATO':
            factorvalue = ((nf['operate_profit'] + nf['operate_expense']).rolling(252).sum() / 63) / nf['total_asset']
        elif factor == 'GP':
            factorvalue = (nf['operate_profit'].rolling(252).sum() / 63) / nf['total_asset']
        elif factor == 'GM':
            factorvalue = (nf['operate_profit'].rolling(252).sum() / 63) / ((nf['operate_profit'] + nf['operate_expense']).rolling(252).sum() / 63)
        elif factor == 'ROA':
            factorvalue = nf['ROA']

        elif factor in ['AGRO','IGRO','CXGRO']:
            temp = nf[statemap[factor][0]]
            factorvalue = abs(temp.copy(deep=True) * 0)

            for i in range(len(temp.iloc[0, :])):
                stock_val = temp.iloc[:, i]
                stock_val = stock_val.dropna(how='all')
                stock_fval = abs(stock_val.copy(deep=True) * 0)
                for j in range(len(stock_val) - 1, 19, -1):
                    temp_y = stock_val.iloc[j - 20:j].fillna(0)
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
                factorvalue.iloc[:, i] = stock_fval

        st = pd.HDFStore(outFilename)
        if factor in [x[1:] for x in st.keys()]:
            existday = st.select_column(factor, 'index')
            st.append(factor, factorvalue.loc[factorvalue.index.difference(existday)], format='t')
        else:
            st.append(factor, factorvalue, format='t')
        st.close()

        print(factor)


