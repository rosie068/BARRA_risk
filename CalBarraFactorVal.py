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

dates = pd.date_range('20090101', '20191101')

#CalBarraVolatility(dates) ##tested for first 300 entries
#CalBarraLiquidity(dates)  ##tested for first 300 entries
#CalBarraSize(dates)   ##tested for first 300 entries
#CalBarraMomentum(dates)  ##tested for first 300 entries

#CalBarraQuality(dates) ##tested for first 300 entries, some data missing
#CalBarraValue(dates)  ##tested for first 300 entries, some data missing
#CalBarraGrowth(dates) ##tested for all entries of first 2 stocks, some data missing

#CalBarraDividendYield(dates)  ##not tested, missing all data
#CalBarraSentiment(dates)  ##not tested, missing all data