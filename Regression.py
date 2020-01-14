"""
Created on Wed Dec 18 2019
@author: RosemaryHe
"""

import pandas as pd
import numpy as np
import math
from sklearn import linear_model
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

##regress factors values to get style factor excess returns
DB_CONN = 0
def Regression(daaaates):
    '''
    fcl = ['DividendYield','Growth','Liquidity','Quality','Sentiment','Size','Value','Volatility','Momentum']
    st = pd.HDFStore('filled_factors.h5')
    state = st.select('factor',"columns="+str(fcl))
    mkt = st.select('mkt', "columns=['sec_return','close_price','total_share']")
    st.close()
    '''
    fcl = ['date','cash','operate_profit','operate_expense']      ##for testing
    st = pd.HDFStore('A.h5')
    state = st.select('sheet',"columns="+str(fcl))
    mkt = st.select('mkt', "columns=['sec_return','close_price','total_share']")
    st.close()
    ##test end

    state['mktVal'] = mkt['close_price'] * mkt['total_share']
    state['sec_return'] = mkt['sec_return']
    #行业信息
    if DB_CONN == 1:
        sector = pd.read_sql("""select t1.sec_code,cast(t1.enter_date as varchar) as date,t1.sector_code,t2.group_code
                                              from tbas..tSECTOR_config t1, tbas..tSECTOR_group_config t2
                                              where t1.class_code=%d and t2.group_class=%d and t2.sector_code=t1.sector_code
                                              order by t1.sec_code"""
                             % (class_code, group_class),
                             con=conn, index_col=['date', 'sec_code'], parse_dates=['date'])
    else:
        sector = pd.read_csv(r'data/datasector.csv', index_col=None, parse_dates=[0], encoding='gbk')  ##行业分类
        sector['sec_code'] = [('00000' + str(x))[-6:] for x in sector['sec_code']]  ##get their sec_ID code, end up with 000001,000003,...
        sector = sector.set_index(['date', 'sec_code'])
    state['sec_code'] = [x[1] for x in state.index.values]
    state = state.set_index(['date','sec_code'])

    Factors = pd.merge(state, sector, left_index=True, right_index=True, how='outer')
    Factors[['sector_code', 'group_code']] = Factors[['sector_code', 'group_code']].fillna(method='ffill')
    Factors = Factors.drop('group_code', axis=1)

    index = Factors.index.values
    Factors['date'] = [x[0] for x in index]
    Factors['sec_ID'] = [x[1] for x in index]
    Factors = Factors[Factors.date.isin(daaaates)]

    Factors = Factors.drop_duplicates(subset=['date','sec_ID'], keep='first')
    Factors = Factors.dropna()
    days = Factors['date'].unique()

    ###test
    for fac in ['cash','operate_profit','operate_expense']:
        ##standardize factors
        Factors[fac] = (Factors[fac] - Factors[fac].mean()) / Factors[fac].std()
        ##winsorize
        s_plus = max(0,min(1, 0.5/(max(Factors[fac])-3)))
        s_minus = max(0,min(1, 0.5/(-3-min(Factors[fac]))))
        for i in range(len(Factors)):
            if Factors[fac].iloc[i]>3:
                Factors[fac].iloc[i] = 3*(1-s_plus)+Factors[fac].iloc[i]*s_plus
            elif Factors[fac].iloc[i]<-3:
                Factors[fac].iloc[i] = -3*(1-s_minus)+Factors[fac].iloc[i]*s_minus
    ##test end

    ##get list for industry names
    ind_list = Factors.iloc[:, 5].drop_duplicates().values
    style_Factors = pd.DataFrame(index=days, columns=['cash','operate_profit','operate_expense'])
    ##style_Factors = pd.DataFrame(index=days, columns=['DividendYield','Growth','Liquidity','Quality','Sentiment','Size','Value','Volatility','Momentum'])
    industry_Factors = pd.DataFrame(index=days, columns=ind_list)
    special_Factors = pd.DataFrame(index=Factors.index, columns=['special_inc'])

    last_day = days[0]
    last_exposure = pd.DataFrame()
    for day in days:
        each_Factor = Factors[Factors.date==day]
        ##test if data is long enough
        if each_Factor.shape[0] > 5:
            last_day = day
            style = each_Factor.iloc[:, 0:3]
            ind = each_Factor.iloc[:, 5].drop_duplicates().values

            indus = pd.DataFrame(columns=ind, index=each_Factor.index)
            for i in ind:
                temp = each_Factor[each_Factor.sector_code == i]
                indus.loc[indus.index.isin(temp.index), i] = 1
            indus = indus.fillna(0)

            each_Factor['weight'] = [math.sqrt(x) for x in each_Factor['mktVal']]
            cutoff = np.percentile(each_Factor['weight'], 95)
            each_Factor['weight'] = [min(x, cutoff) for x in each_Factor['weight']]
            ##each_Factor['weight'] = each_Factor['mktVal'] / each_Factor['mktVal'].sum()

            big_matrix = pd.merge(style, indus, left_index=True, right_index=True)

            X = big_matrix.as_matrix(columns=None)
            y = np.array(each_Factor.loc[:, 'sec_return'])
            w = np.array(each_Factor.loc[:, 'weight'])
            ##第一次横截面回归
            wls_model = sm.WLS(y, X, weights=w)
            result = wls_model.fit()
            ##style factors first, then industry factors
            ##factor_return = result.params
            residual = result.resid

            ##检测异常收益,修正
            sig_res = 1.4826 * np.median(abs(residual - np.median(residual)))
            error = pd.DataFrame(index=each_Factor.index)
            error['res'] = residual
            error['error'] = pd.Series()
            error = error.fillna(0)

            temp = error[abs(error.res)>4*sig_res]
            temp['error'] = (abs(temp['res']) - sig_res) * (temp['res']/abs(temp['res']))
            error['error'][error.index.isin(temp.index)] = temp['error']
            ##第二次修正后横截面回归
            new_y = np.array(each_Factor.loc[:, 'sec_return'] - error['error'])
            new_wls_model = sm.WLS(new_y, X, weights=w)
            new_result = new_wls_model.fit()
            ##style factors first, then industry factors
            ##最终因资收益率
            new_factor_return = new_result.params
            ##特质收益率
            new_residual = new_result.resid + error['error']

            style_Factors.loc[day,:] = new_factor_return[0:3]
            for k in range(len(ind)):
                industry_Factors.loc[day,ind[k]] = new_factor_return[3+k]
            special_Factors['special_inc'][special_Factors.index.isin(new_residual.index)] = new_residual

    last_exposure = Factors[Factors.date==last_day]

    st1 = pd.HDFStore('style_Factors.h5')
    st1.put('style_factor',style_Factors)
    st1.put('industry_factor',industry_Factors)
    st1.put('special_factor',special_Factors)
    st1.put('last_exposure',last_exposure)
    st1.close()
