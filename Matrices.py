"""
Created on Wed Dec 18 2019
@author: RosemaryHe
"""

import pandas as pd
import numpy as np
import math
import statsmodels.api as sm
from numpy import linalg as la

##calculate risk covariance matrix F and specific risk matrix Delta

def half_life(time, length):
    sum = 0
    for i in range(length):
        sum += 2**(i/length)
    return (2**(time/length))/sum

def cal_covar(k1_val, k2_val, halflife, k1_start, phi):
    sum = 0
    k1_mean = np.mean(k1_val)
    k2_mean = np.mean(k2_val)
    for t in range(abs(phi), len(k1_val)-1-abs(phi)):
        sum += half_life(t,halflife) * (k1_val[t+k1_start] - k1_mean) * (k2_val[t+phi] - k2_mean)
    return sum

def Matrices():
    st = pd.HDFStore('style_Factors.h5')
    style = st.select('style_factor')
    industry = st.select('industry_factor')
    special = st.select('special_factor')
    last_expo = st.select('last_exposure')
    st.close()

    ###因子收益率协方差
    factors = pd.merge(style,industry, left_index=True, right_index=True, how='outer')
    factors = factors.fillna(0)
    factor_names = factors.columns.values.tolist()

    Fn = pd.DataFrame(index=factor_names,columns=factor_names)
    ###standardize, winsorize
    for d in range(len(factor_names)):
        Fn.iloc[d,d] = 1
        factors[factor_names[d]] = (factors[factor_names[d]] - factors[factor_names[d]].mean()) / factors[factor_names[d]].std()
        s_plus = max(0, min(1, 0.5 / (max(factors[factor_names[d]]) - 3)))
        s_minus = max(0, min(1, 0.5 / (-3 - min(factors[factor_names[d]]))))
        for i in range(len(factors)):
            if factors[factor_names[d]].iloc[i] > 3:
                factors[factor_names[d]].iloc[i] = 3 * (1 - s_plus) + factors[factor_names[d]].iloc[i] * s_plus
            elif factors[factor_names[d]].iloc[i] < -3:
                factors[factor_names[d]].iloc[i] = -3 * (1 - s_minus) + factors[factor_names[d]].iloc[i] * s_minus
        factors = factors.fillna(0)

    ##factors = factors.iloc[0:200]  ##testing first 200
    ##计算各因子之间协方差
    for i in range(len(factor_names)):
        for j in range(i+1, len(factor_names)):
            k1_values = factors[factor_names[i]].values
            k2_values = factors[factor_names[j]].values

            F_k1_k2 = 0
            ##we are setting N = 10
            for phi in range(-10, 11):
                k1_k2_corr = cal_covar(k1_values, k2_values, 480, 0, phi) / (math.sqrt(cal_covar(k1_values, k1_values, 480, 0, 0) * cal_covar(k2_values, k2_values, 480, phi, phi)))
                covariance = k1_k2_corr * math.sqrt(cal_covar(k1_values, k1_values, 90, 0, 0)) * math.sqrt(cal_covar(k2_values, k2_values, 90, phi, phi))
                F_k1_k2 += (11 - abs(phi)) * covariance
            Fn.loc[factor_names[i], factor_names[j]] = F_k1_k2
            Fn.loc[factor_names[j], factor_names[i]] = F_k1_k2

    ##H = 21, N = 10
    Fh = Fn * 21.0 / 11.0
    Fh = Fh.fillna(0)

    Fh_mat = Fh.as_matrix(columns=None)
    U,D,V = la.svd(Fh_mat)

    for n in range(len(D)):
        if D[n] <= 0:
            D[n] = 0.001
    D_mat = np.eye(len(D))
    for d in range(len(D)):
        D_mat[d][d] = D[d]

    F = np.matmul(np.matmul(U,D_mat), V)

    ##特质风险
    index = special.index.values
    index = special.index.values
    special['sec_ID'] = [x[1] for x in index]
    special['date'] = [x[0] for x in index]
    special = special[special.sec_ID.isin(last_expo.sec_ID)]
    special = special.sort_values('sec_ID')
    sec_IDs = special['sec_ID'].unique()
    special = special.fillna(0)

    ##特质风险计算
    ##gamma, 这里因测试数据不够长无法测试未完成. 导致部分股票历史数据不够长,按照计算方法可能无法准确估计该股票特质风险
    ##此段为计算gamma 并且修正那些gamma<1的股票.因为数据不足而无法测试,如需要请调试
    '''
    special['gamma'] = pd.Series()
    for s in sec_IDs:
        ##计算鲁邦差
        each_special = special[special.sec_ID==s]
        robust_sd = (np.percentile(each_special['special_inc'], 75) - np.percentile(each_special['special_inc'], 25)) / 1.35
        each_special['special_inc'][each_special.special_inc > 10*robust_sd] = 10 * robust_sd
        each_special['special_inc'][each_special.special_inc < -10 * robust_sd] = -10 * robust_sd
    
        normal_sd = each_special.special_inc.std()
        Z = abs((normal_sd - robust_sd) / robust_sd)
        ##计算blending coefficient
        gamma = min(1, max(0, (len(each_special)-60)/120)) * min(1, max(0, math.exp(1-Z)))
        special['gamma'][special.sec_ID==s] = gamma
    special = special.fillna(0)
    
    ###get the parameters from gamma=1 stocks
    last_day = last_expo.iloc[0,6]     ##找到最后一天的回归值,计算gamma<1的股票特质因子值
    last_day_specials = special[special.date==last_day]
    reference_gamma = last_day_specials[last_day_specials.gamma==1]  ##gamma==1 我们用来做借助的数据
    reset_gamma = last_day_specials[last_day_specials.gamma<1]  ##gamma<1 我们要从新计算特质值
    
    temp_mktVal = last_expo.mktVal
    temp_mktVal = [math.sqrt(x) for x in temp_mktVal]     ##weight=市值的平方根
    cutoff = np.percentile(temp_mktVal, 95)     ##去极值处理
    temp_mktVal = [min(x, cutoff) for x in temp_mktVal]
    w = np.array(temp_mktVal)    ##w = weights
    
    temp_y = last_day_specials[last_day_specials.sec_ID.isin(reference_gamma.sec_ID)]
    y = np.array(temp_y.loc[:,'special_inc'])
    
    temp_x = last_expo[last_expo.sec_ID.isin(reference_gamma.sec_ID)]
    ind = temp_x.loc[:, 'sector_code'].drop_duplicates().values
    x_ind = pd.DataFrame(columns=ind, index=temp_x.index)
    for i in ind:
        temp = temp_x[temp_x.sector_code == i]
        x_ind.loc[x_ind.index.isin(temp.index), i] = 1
    x_ind = x_ind.fillna(0)
    x_style = temp_x.iloc[:,0:3]  ##for testing
    ##x_style = temp_x.loc[:,['volatility','liquidity','momentum']]
    x_all = pd.merge(x_style, x_ind, left_index=True, right_index=True)    ###把风格因子和行业因子合并
    X = x_all.as_matrix(columns=None)
    
    wls_model = sm.WLS(y, X, weights=w)
    result = wls_model.fit()
    ##style factors first, then industry factors
    factor_return = result.params    ##回归的系数,用来计算gamma<1的股票特质因子值
    
    ####计算gamma<1的股票的特质因子值
    new_temp_x = last_expo[last_expo.sec_ID.isin(reset_gamma.sec_ID)]
    new_ind = new_temp_x.loc[:, 'sector_code'].drop_duplicates().values
    new_x_ind = pd.DataFrame(columns=ind, index=new_temp_x.index)
    for i in new_ind:
        new_temp = new_temp_x[new_temp_x.sector_code == i]
        new_x_ind.loc[new_x_ind.index.isin(new_temp.index), i] = 1
    new_x_ind = new_x_ind.fillna(0)
    new_x_style = new_temp_x.iloc[:,0:3]  ##for testing
    ##x_style = temp_x.loc[:,['volatility','liquidity','momentum']]
    new_x_all = pd.merge(new_x_style, new_x_ind, left_index=True, right_index=True)    ###把风格因子和行业因子合并
    new_X = new_x_all.as_matrix(columns=None)
    new_special = np.multiply(factor_return,new_X)
    reset_gamma['special_inc'] = new_special
    
    special = pd.merge(reset_gamma,reference_gamma, left_index=True, right_index=True) ##把修正过的特质因子值并到一个表格
    '''

    ###计算特质风险收益协方差
    delta = pd.DataFrame(index=sec_IDs,columns=sec_IDs)
    for d in range(len(sec_IDs)):
        delta.iloc[d,d] = 1

    for i in range(len(sec_IDs)):
        for j in range(i+1, len(sec_IDs)):
            s1_val = special['special_inc'][special.sec_ID==sec_IDs[i]].values
            s2_val = special['special_inc'][special.sec_ID == sec_IDs[j]].values

            if len(s1_val) >= 10 and len(s2_val) >= 10:
                D_k1_k2 = 0
                for phi in range(-10, 10):
                    s1_s2_corr = cal_covar(s1_val, s2_val, 480, 0, phi) / (math.sqrt(cal_covar(s1_val, s1_val, 480, 0, 0) * cal_covar(s1_val, s1_val, 480, phi, phi)))
                    covariance = s1_s2_corr * math.sqrt(cal_covar(s1_val, s1_val, 90, 0, 0)) * math.sqrt(cal_covar(s2_val, s2_val, 90, phi, phi))
                    D_k1_k2 += (11 - abs(phi)) * covariance
                delta.loc[sec_IDs[i], sec_IDs[j]] = D_k1_k2
                delta.loc[sec_IDs[j], sec_IDs[i]] = D_k1_k2
    delta = delta.fillna(0)

    ##tranfrom the dataframe to the matrix with all factor exposures
    temp = last_expo[last_expo.sec_ID.isin(sec_IDs)]
    temp = temp.set_index('sec_ID')
    temp['sec_ID'] = temp.index.values
    exposure = pd.DataFrame(index=temp.sec_ID, columns=factor_names)
    for s in style.columns.values:
        exposure[s] = temp[s]
    exist_industries = np.unique(temp['sector_code'].values)
    for ind in exist_industries:
        list = temp['sec_ID'][temp.sector_code==ind]
        exposure[ind][exposure.index.isin(list.values)] = 1

    exposure = exposure.fillna(0)
    X = exposure.values
    Delta = delta.values
    risk = np.matmul(np.matmul(X,F), np.transpose(X)) + Delta  ##股票超额收益率的协方差

    risk_df = pd.DataFrame(risk, index=delta.index, columns=delta.columns.values)

    st1 = pd.HDFStore('Risk.h5')
    st1.put('risk',risk_df)
    st1.close()
