"""
Created on Wed Dec 18 2019
@author: RosemaryHe
"""

import pandas as pd
import numpy as np
import math
from sklearn import datasets, linear_model

##combine values into 9 style factors, fill in Nan values with fitted values

def linear_regression(X,y):
    reg = linear_model.LinearRegression()
    reg.fit(X,y)
    return reg.coef_, reg.intercept_

##for missing values
def fill_in_missing(df, des):
    X = np.asmatrix(df[des]).transpose()
    temp_y = [math.log(x) for x in df['mktVal']]
    y = np.asmatrix(temp_y).transpose()
    coef, intercept = linear_regression(X, y)

    temp_df = df[df.des.isnull()]
    temp_df['ln_mktVal'] = [math.log(x) for x in temp_df['mktVal']]
    temp_df[des] = temp_df['ln_mktVal'] * intercept + coef
    df[df.index.isin(temp_df.index)][des] = temp_df[des]
    return df

DB_CONN = 0
def Factor_exposure(daaaates):
    '''
    factors = ['Liquidity','Quality','Value','Growth','Sentiment','Momentum','Size','Volatility','DividendYield']
    style_arr = []
    for fac in factors:
        filename = 'Barra'+fac+'.h5'
        st = pd.HDFStore(filename)
        ##get the entire sheet
        state = st.select(st.keys()[0])
        ##average of each row, 这里我们把所有三级因子等权加权取得一级因子
        value = state.mean(1)
        style_arr = style_arr.push_back(value)
        st.close()
    styleFactors = pd.concat(style_arr)   ##这里有九个一级因子

    #calc marketvalue
    st2 = pd.HDFStore('A.h5')
    mkt = st2.select('mkt', "columns=['close_price','total_share','sec_return']")
    temp = st2.select('sheet',"columns=cash")
    st2.close()
    '''

    ##for testing
    factors = ['date', 'cash']  ##for testing
    st = pd.HDFStore('A.h5')
    styleFactors = st.select('sheet', "columns=" + str(factors))
    mkt = st.select('mkt', "columns=['close_price','total_share','sec_return']")
    st.close()
    ##test end

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
    ##财务数据对齐
    temp = styleFactors.unstack()
    temp = temp[(temp.index.month.isin([3,6,9,12]))].stack()
    nf = temp.reset_index()
    tnf = nf['pdate'].groupby([nf['date'],nf['sec_code']]).max()
    nf = nf.set_index(['date', 'sec_code', 'pdate'])
    tnf = tnf.reset_index()
    tnf = tnf.set_index(['date', 'sec_code', 'pdate'])
    nf = nf[nf.index.isin(tnf.index)]
    nf = nf.reset_index('pdate')
    ##财报数据发布后，剔除分红
    dividend = dividend['dividend'].reindex(nf.index.union(dividend.index)).fillna(0)
    ##日度对齐
    dividend = dividend.unstack()
    dividend = dividend.reindex(dividend.index.union(daaaates)).ffill()
    dividend = dividend.reindex(daaaates)
    mkt = mkt.unstack()
    mkt = mkt.reindex(mkt.index.union(daaaates)).ffill()
    mkt = mkt.reindex(daaaates)
    tcap = mkt['close_price'] * mkt['total_share']
    tcap[tcap == 0] = np.nan

    tcap = tcap + dividend
    tcap = tcap.stack()
    temp_tcap = pd.DataFrame(tcap, columns=['mktVal'])

    tindex = temp_tcap.index.values
    temp_tcap['date'] = [x[0] for x in tindex]
    temp_tcap['sec_ID'] = [x[1] for x in tindex]

    sindex = styleFactors.index.values
    styleFactors['date'] = [x[0] for x in sindex]
    styleFactors['sec_ID'] = [x[1] for x in sindex]
    styleFactors = styleFactors[styleFactors.date.isin(daaaates)]
    mkt = mkt.stack()
    temp_tcap['sec_return'] = mkt['sec_return']

    styleFactors = styleFactors.merge(temp_tcap, left_on=['date','sec_ID'], right_on=['date','sec_ID'], how='outer')
    styleFactors = styleFactors.sort_values(by=['sec_ID','date'])

    styleFactors = styleFactors.fillna(method='bfill')

    long_date = styleFactors.date
    dates = long_date.drop_duplicates()
    styleFactors['ref_date'] = styleFactors['date']
    styleFactors = styleFactors.set_index(['date','sec_ID'])

    styleFactors['weight'] = pd.Series()
    style_fac_arr = []
    for d in dates:
        tempw = styleFactors[styleFactors.ref_date==d]
        tempw['weight'] = tempw['mktVal'] / tempw['mktVal'].sum()
        ##将因子标准化去极值
        for fac in ['cash']:
        ##for fac in factors:
            weighted_mean = (tempw['weight'] * tempw[fac]).sum()
            tempw[fac] = (tempw[fac] - weighted_mean) / tempw[fac].std()

            s_plus = max(0, min(1, 0.5 / (max(tempw[fac]) - 3)))
            s_minus = max(0, min(1, 0.5 / (-3 - min(tempw[fac]))))

            tempw = tempw.iloc[0:10,:]  ##for testing

            for x in tempw.index.values:
                if tempw.loc[x,fac] > 3:
                    tempw.loc[x,fac] = 3 * (1 - s_plus) + tempw.loc[x,fac] * s_plus
                elif tempw.loc[x,fac] < -3:
                    tempw.loc[x,fac] = -3 * (1 - s_minus) + tempw.loc[x,fac] * s_minus
        style_fac_arr.append(tempw)
    new_styleFactors = pd.concat(style_fac_arr)

    #行业信息
    if DB_CONN == 1:
        sector = pd.read_sql("""select t1.sec_code,cast(t1.enter_date as varchar) as date,t1.sector_code,t2.group_code
                                              from tbas..tSECTOR_config t1, tbas..tSECTOR_group_config t2
                                              where t1.class_code=%d and t2.group_class=%d and t2.sector_code=t1.sector_code
                                              order by t1.sec_code""" % (class_code, group_class),
                             con=conn, index_col=['date', 'sec_code'], parse_dates=['date'])
    else:
        sector = pd.read_csv(r'data/datasector.csv', index_col=None, parse_dates=[0], encoding='gbk')  ##行业分类
        sector['sec_code'] = [('00000' + str(x))[-6:] for x in
                              sector['sec_code']]  ##get their sec_ID code, end up with 000001,000003,...
        sector = sector.set_index(['date', 'sec_code'])

    Factors = pd.merge(new_styleFactors, sector, left_index=True, right_index=True, how='left')
    Factors[['sector_code', 'group_code']] = Factors[['sector_code', 'group_code']].fillna(method='ffill')
    Factors = Factors.dropna(how='all')
    Industry = Factors.group_code.unique()

    ##fill in missing values,填充缺失数据
    filled_arr = []
    for i in Industry:
        temp_df = Factors[Factors.group_code==i]
        #for j in factors:
        for j in ['cash']:
            temp_df = fill_in_missing(temp_df,j)
        filled_arr.append(temp_df)
    filled_factors = pd.concat(filled_arr)

    ##再次标准化
    for fact in factors:
        weight_mean = (filled_factors['weight'] * filled_factors[fact]).sum()
        filled_factors[fact] = (filled_factors[fact] - weight_mean) / filled_factors[fact].std()

    st1 = pd.HDFStore('filled_factors.h5')
    st1.put('factor', filled_factors)
    st1.close()
