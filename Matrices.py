import pandas as pd
import numpy as np
import math
from numpy import linalg as la

##calculate risk covariance matrix

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

st = pd.HDFStore('style_Factors.h5')
style = st.select('style_factor')
industry = st.select('industry_factor')
special = st.select('special_factor')
last_expo = st.select('last_exposure')
st.close()

###因子收益率协方差
factors = pd.merge(style,industry, left_index=True, right_index=True, how='outer')
##just for testing
factors = factors.dropna(how='all')
factors = factors.iloc[0:200, 0:20]
##end of testing
factors = factors.fillna(0)
factor_names = factors.columns.values.tolist()

Fn = pd.DataFrame(index=factor_names,columns=factor_names)
for d in range(len(factor_names)):
    Fn.iloc[d,d] = 1
    factors[factor_names[d]] = (factors[factor_names[d]] - factors[factor_names[d]].mean()) / factors[factor_names[d]].std()
    factors = factors.fillna(0)

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
special['sec_ID'] = [x[1] for x in index]
special['date'] = [x[0] for x in index]
special = special[special.sec_ID.isin(last_expo.sec_ID)]
special = special.sort_values('sec_ID')
sec_IDs = special['sec_ID'].unique()
special = special.fillna(0)

##特质风险计算
'''
##计算gamma
special['gamma'] = pd.Series()
for s in sec_IDs:
    each_special = special[special.sec_ID==s]
    robust_sd = (np.percentile(each_special['special_inc'], 75) - np.percentile(each_special['special_inc'], 25)) / 1.35
    each_special['special_inc'][each_special.special_inc > 10*robust_sd] = 10 * robust_sd
    each_special['special_inc'][each_special.special_inc < -10 * robust_sd] = -10 * robust_sd

    normal_sd = each_special.special_inc.std()
    Z = abs((normal_sd - robust_sd) / robust_sd)
    gamma = min(1, max(0, (len(each_special)-60)/120)) * min(1, max(0, math.exp(1-Z)))
    special['gamma'][special.sec_ID==s] = gamma
    #if gamma < 1:
'''
###for testing
sec_IDs = sec_IDs[0:20]
###end testing

delta = pd.DataFrame(index=sec_IDs,columns=sec_IDs)
for d in range(len(sec_IDs)):
    delta.iloc[d,d] = 1

for i in range(len(sec_IDs)):
    for j in range(i+1, len(sec_IDs)):
        s1_val = special['special_inc'][special.sec_ID==sec_IDs[i]].values
        s2_val = special['special_inc'][special.sec_ID == sec_IDs[j]].values

        if len(s1_val) >= 10 and len(s2_val) >= 10:
            D_k1_k2 = 0
            ##we are setting N = 2, because we do NOT have enough samples
            for phi in range(-2, 2):
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
risk = np.matmul(np.matmul(X,F), np.transpose(X)) + Delta

risk_df = pd.DataFrame(risk, index=delta.index, columns=delta.columns.values)

st1 = pd.HDFStore('Risk.h5')
st1.put('risk',risk_df)
st1.close()