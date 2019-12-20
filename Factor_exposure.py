import pandas as pd
import numpy as np
import math
from sklearn import datasets, linear_model

##combine values into 9 style factors, fill in Nan values with fitted values

##standardize the factor, des is the descriptor to be standardized
def standardize(df, des):
	weighted_mean = (df['weight']*df[des]).sum()
	std = df[des].std()
	df[des] = (df[des]-weighted_mean)/std
	return df

##winsorize the factor, des is the descriptor to be winsorized
def winsorize(df, des):
	s_plus = max(0,min(1, 0.5/(max(df[des])-3)))
	s_minus = max(0,min(1, 0.5/(-3-min(df[des]))))
	for i in range(len(df)):
		if df[des].iloc[i]>3:
			df[des].iloc[i] = 3*(1-s_plus)+df[des].iloc[i]*s_plus
		elif df[des].iloc[i]<-3:
			df[des].iloc[i] = -3*(1-s_minus)+df[des].iloc[i]*s_minus
	return df

def linear_regression(X,y):
    reg = linear_model.LinearRegression()
    reg.fit(X,y)
    return reg.coef_, reg.intercept_

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
def Factor_exposure(dates):
    factors = ['Liquidity','Quality','Value','Growth','Sentiment','Momentum','Size','Volatility','DividendYield']
    styleFactors = pd.DataFrame()
    for fac in factors:
        filename = 'Barra'+fac+'.h5'
        st = pd.HDFStore(filename)
        ##get the entire sheet
        state = st.select(st.keys()[0])
        ##average of each row
        value = state.mean(1)
        styleFactors[fac] = value
        st.close()

    #calc marketvalue
    st2 = pd.HDFStore('A.h5')
    mkt = st2.select('mkt',"columns=['close_price','total_share','sec_return']")
    temp = st2.select('sheet',"columns=cash")
    st2.close()
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
    temp = temp.unstack()
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
    dividend = dividend.reindex(dividend.index.union(dates)).ffill()
    dividend = dividend.reindex(dates)
    mkt = mkt.unstack()
    mkt = mkt.reindex(mkt.index.union(dates)).ffill()
    mkt = mkt.reindex(dates)
    tcap = mkt['close_price'] * mkt['total_share']
    tcap[tcap == 0] = np.nan

    styleFactors['mktVal'] = (tcap + dividend).stack()
    styleFactors['sec_return'] = mkt['sec_return']

    dates = styleFactors.index.get_level_values('date')  ##returns an index with only one level 'date'
    styleFactors['date'] = dates
    dates = dates.drop_duplicates()

    for d in dates:
        temp = styleFactors[styleFactors.date==d]
        temp['weight'] = temp['mktVal'] / temp['mktVal'].sum()
        styleFactors[styleFactors.index.isin(temp.index)]['weight'] = temp['weight']
    for fac in factors:
        styleFactors = standardize(styleFactors, fac)
        styleFactors = winsorize(styleFactors, fac)

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

    Industry = sector.group_code.unique()
    Factors = pd.merge(styleFactors, sector, left_index=True, right_index=True, how='outer')
    Factors[['sector_code', 'group_code']] = Factors[['sector_code', 'group_code']].fillna(method='ffill')
    Factors = Factors.dropna(how='all')

    filled_arr = []
    for i in Industry:
        temp_df = Factors[Factors.group_code==i]
        for j in factors:
            temp_df = fill_in_missing(temp_df,j)
        filled_arr.append(temp_df)
    filled_factors = pd.concat(filled_arr)
    Factors.to_csv('Factors.csv', index=False)

    st1 = pd.HDFStore('filled_factors.h5')
    st1.put('factor', filled_factors)
    st1.close()