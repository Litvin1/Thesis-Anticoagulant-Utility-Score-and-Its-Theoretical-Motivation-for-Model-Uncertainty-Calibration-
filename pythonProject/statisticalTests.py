# Vadim Litvinov
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
from scipy.stats import skew
from scipy.stats import chi2
import visualization


def statisticalAnalysis(yv_proba, yb_proba, yv, yb):
    # TODO check normality of ACU of v and b
    maskv = yv == 1
    maskb = yb == 1
    mask0 = (yv == 0) & (yb == 0)
    #yv_proba = yv_proba[maskv]
    #yb_proba = yb_proba[maskb]
    acuv = yv_proba[maskv] - yb_proba[maskv]
    acub = yv_proba[maskb] - yb_proba[maskb]
    acu0 = yv_proba[mask0] - yb_proba[mask0]
    #visualization.visRandomVariable(acub)
    #visualization.visRandomVariable(acu0)
    #visualization.visRandomVariable(acuv)
    print('pearsons skewness acub:', skew(acub))
    print('pearsons skewness acu0:', skew(acu0))
    print('pearsons skewness acuv:', skew(acuv))
    #plt.show()
    #plt.show()
    CIb, CI0, CIv = confidenceInterval(acub, acu0, acuv)
    #unpairedTTest(acub, acu0, acuv)
    #print(shapiroWilkNormality(acuv))
    #print(shapiroWilkNormality(acub))
    #print(kolmogorovSmirnov(acuv))
    #print(kolmogorovSmirnov(acub))
    return CIb, CI0, CIv


def confidenceInterval(acub, acu0, acuv):
    meanb = np.average(acub)
    mean0 = np.average(acu0)
    meanv = np.average(acuv)
    seb = (np.std(acub))/np.sqrt(len(acub))
    se0 = (np.std(acu0))/np.sqrt(len(acu0))
    sev = (np.std(acuv))/np.sqrt(len(acuv))
    CIb = stats.norm.interval(0.95, loc=meanb, scale=seb)
    CI0 = stats.norm.interval(0.95, loc=mean0, scale=se0)
    CIv = stats.norm.interval(0.95, loc=meanv, scale=sev)
    print(CIb)
    print(CI0)
    print(CIv)
    return CIb, CI0, CIv


def shapiroWilkNormality(x):
    shapiro_test = stats.shapiro(x)
    return shapiro_test.pvalue


def kolmogorovSmirnov(x):
    return stats.kstest(x, 'norm').pvalue


def pairedTTest(a, b):
    a_df = pd.DataFrame(a)
    b_df = pd.DataFrame(b)
    a_df.to_excel(r'/data/old_data/vadim/model ba.xlsx', index=False)
    b_df.to_excel(r'/data/old_data/vadim/padua ba.xlsx', index=False)
    d_df = a_df - b_df
    d_lst = d_df[0].values.tolist()
    #visualization.visRandomVariable(d_lst)
    #print('d std:', np.std(d_lst))
    #print('d avg:', np.average(d_df))
    print('avg model:', np.average(a))
    print('avg padua:', np.average(b))
    print(stats.ttest_rel(a, b, alternative='greater'))


def unpairedTTest(b, z, v):
    print('avg acub:', np.average(b))
    print('avg acu0:', np.average(z))
    print('avg acuv:', np.average(v))
    print(stats.ttest_ind(v, b))
