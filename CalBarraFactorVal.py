"""
Created on Wed Nov 13 2019
@author: HeYuan
"""
import pandas as pd
import numpy as np
from CalBarraLiquidity import CalBarraLiquidity
from CalBarraQuality import CalBarraQuality
from CalBarraValue import CalBarraValue
from CalBarraGrowth import CalBarraGrowth
from CalBarraSentiment import CalBarraSentiment
from CalBarraMomentum import CalBarraMomentum
from CalBarraSize import CalBarraSize
from CalBarraVolatility import CalBarraVolatility
from CalBarraDividendYield import CalBarraDividendYield
from Factor_exposure import Factor_exposure
from Regression import Regression
from Matrices import Matrices

dates = pd.date_range('20140101', '20191231')

CalBarraVolatility(dates) ##tested for first 300 entries
CalBarraLiquidity(dates)  ##tested for first 300 entries
CalBarraSize(dates)   ##tested for first 300 entries
CalBarraMomentum(dates)  ##tested for first 300 entries

CalBarraQuality(dates) ##tested for first 300 entries, some data missing
CalBarraValue(dates)  ##tested for first 300 entries, some data missing
CalBarraGrowth(dates) ##tested for all entries of first 2 stocks, some data missing

CalBarraDividendYield(dates)  ##not tested, missing all data
CalBarraSentiment(dates)  ##not tested, missing all data

Factor_exposure(dates)   
##tested for small segment of data, combines all factors into nine major factors and fill in missing data
##may have trouble with running time and some logic on bigger dataset
##fill_in functions NOT tested!

Regression(dates)
##regress to find risk factors' exposures
##tested for sample data, may have trouble with running time on bigger dataset

Matrices()
##estimate covariance matrices F and specific risk Delta, return V, the covariance matrix for stock returns
##tested for sample data, may have trouble with running time on bigger dataset