# Vadim Litvinov
import numpy as np
import pandas as pd
import scipy.stats
import statsmodels.sandbox.distributions.gof_new
from sklearn.impute import KNNImputer
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import NearestNeighbors
from statsmodels.stats.weightstats import ttost_ind
import featureSelection
import visualization


def completePaduaScoreCol(beilinson_new):
    # replace nans with zeros
    nanToZero(beilinson_new)
    activeCaner = 3 * beilinson_new['malignant']
    vtePrior = 3 * beilinson_new['vteprior']
    correctSurgery30DaysAndreducedMobility(beilinson_new)
    reducedMobility = 3 * beilinson_new['function 1-ind 2- unind']
    thrombophilia = 3 * beilinson_new['yhrombophilia']
    traumaSurgery30Days = 2 * beilinson_new['surgerylast30d']
    olderThen70 = 1 * beilinson_new['age>70']
    heartRespiratoryFailure= 1 * beilinson_new['CHF']
    strokeOrMi = 1 * beilinson_new['MI']
    infectionOrRheumatic = 1 * beilinson_new['CTD']
    obesity = 1 * beilinson_new['obesity']
    hormonal = 1 * beilinson_new['hormonal']
    sum = pd.DataFrame(activeCaner + vtePrior \
          + reducedMobility + thrombophilia + traumaSurgery30Days + olderThen70\
          + heartRespiratoryFailure + strokeOrMi + infectionOrRheumatic + obesity + hormonal)
    # insert SUM only to where there is no padua score
    sum[:6341] = 0
    beilinson_new['pauda score'].fillna(0, inplace=True)
    beilinson_new['pauda score'] = beilinson_new['pauda score'] + sum[0]


def nanToZero(beilinson_new):
    #beilinson_new.loc[:, beilinson_new.columns != 'pauda score'].fillna(0, inplace=True)
    f = [c for c in beilinson_new.columns if c not in ['pauda score']]
    beilinson_new[f] = beilinson_new[f].fillna(0)
    #beilinson_new.fillna(0, inplace=True)


def correctSurgery30DaysAndreducedMobility(beilinson_new):
    #beilinson_new['surgerylast30d'][~beilinson_new['surgerylast30d'].isnull()] = 1
    beilinson_new['function 1-ind 2- unind'][beilinson_new['function 1-ind 2- unind'] != 2] = 1
    beilinson_new['function 1-ind 2- unind'] = beilinson_new['function 1-ind 2- unind'] - 1


def createPaduaScoreByVars(beilinson_new):
    # delete patients with no weight or hight
    deleteMissingWeightHeight(beilinson_new)
    # string to numeric
    strToFloat(beilinson_new)
    # create BMI var
    beilinson_new['BMI'] = calculateBMI(beilinson_new['HIGHT'], beilinson_new['WEIGHT'])
    beilinson_new[beilinson_new['BMI'] == np.inf] = np.NaN
    # create obesity var
    beilinson_new['obesity'] = np.where(beilinson_new['BMI'] > 30, 1, 0)
    #beilinson_new['pauda score'] +=
    beilinson_new['age>70'] = np.where(beilinson_new['AGE'] > 70, 1, 0)


def deleteMissingWeightHeight(df):
    #pd.set_option('display.max_rows', 5000)
    #print(df['WEIGHT'].value_counts())
    #print(df['HIGHT'].value_counts())
    df = df.loc[~((df['WEIGHT'] == '.') & (df['pauda score'].isna())), :]
    df = df.loc[~((df['HIGHT'] == '.') & (df['pauda score'].isna())), :]
     #df.drop(df[(df['WEIGHT'] == '.') & (df.pauda score.isnull())].index, inplace=True)
    #df = df.drop(df[(df['HIGHT'] == '.') & (df['pauda score'].isnull())].index)
    df.reset_index(drop=True, inplace=True)
    print('sas')


def calculateBMI(height, weight):
    return weight/(pow(height/100, 2))


def strToFloat(df):
    df['HIGHT'].replace('.', 0, inplace=True)
    df['WEIGHT'].replace('.', 0, inplace=True)

    df['HIGHT'] = df['HIGHT'].astype(float)
    df['WEIGHT'] = df['WEIGHT'].astype(float)


def Nmaxelements(list1, N):
    final_list = []
    lrgst_ind = 0
    for i in range(0, N):
        max1 = 0

        for j in range(len(list1)):
            if abs(list1[j][1]) > abs(max1):
                lrgst_ind = j
                max1 = list1[j][1]

        final_list.append(list1[lrgst_ind])
        del list1[lrgst_ind]
        #list1.remove(lrgst_ind)
    print(final_list)


def imputeMean(df):
    frstloc = df.columns.get_loc("min_BUN")
    lastloc = df.columns.get_loc("max_WBC") + 1
    # and now meanImpute only on numeric charevents features
    cols_mean = df.iloc[:, frstloc:lastloc].mean()
    df.iloc[:, frstloc:lastloc] = df.iloc[:, frstloc:lastloc].fillna(cols_mean)


def imputeMeanBeilinson(df):
    #df.replace(['.', '-'], np.NaN, inplace=True)
    df.fillna(df.mean(), inplace=True)


def imputeMedianBeilinson(df):
    df.replace(['.', '-'], np.NaN, inplace=True)
    df.fillna(df.median(), inplace=True)


def imputeMedian(df):
    frstloc = df.columns.get_loc("min_BUN")
    lastloc = df.columns.get_loc("mean_WBC") + 1
    # and now meanImpute only on numeric charevents features
    cols_median = df.iloc[:, frstloc:lastloc].median()
    df.iloc[:, frstloc:lastloc] = df.iloc[:, frstloc:lastloc].fillna(cols_median)
    # fill 0 in nan's in binary features
    df.iloc[:, :frstloc] = df.iloc[:, :frstloc].fillna(0)


def imputeKnn(df):
    k = round(np.sqrt(len(df.index)))
    imputer = KNNImputer(n_neighbors=k, weights='uniform')
    return imputer.fit_transform(df)


def negaToPos(df):
    df['min_Creatinine'] = df['min_Creatinine'].abs()
    df['mean_Creatinine'] = df['mean_Creatinine'].abs()
    df['min_Heart Rate'] = df['min_Heart Rate'].abs()


def drop999999(df):
    # DROP THE 9999999 TEMPORARY
    df.drop(111964, inplace=True)
    # next one is a bleeder
    #df.drop(130520, inplace=True)


def dropAnomalies(df):
    df.drop([39934, 8405, 45874, 8953, 40864], inplace=True)


def avgProba(y_proba, y):
    summa = 0
    for i in range(len(y)):
        if y[i] == 1:
            summa += y_proba[i][1]
    total = np.count_nonzero(y == 1)
    #print('total:', total)
    return summa / total


def paduaPredExportAndDelete(X):
    paduaPred = X['padua prediction']
    paduaScore = X['pauda score']
    X.drop(columns=['pauda score', 'padua prediction'], inplace=True)
    return paduaScore, paduaPred


def dataToNumpy(y, padua_score, padua_pred):
    return y.to_numpy(), padua_score.to_numpy(), padua_pred.to_numpy()


def albuminFirstAlbuminAvg(vte_old, vte_new):
    albuminFirst = vte_old['albuminFirst'][~np.isnan(vte_old['albuminFirst'])]
    albumin = vte_new['albumin'][~np.isnan(vte_new['albumin'])]
    print(ttost_ind(albuminFirst, albumin, -0.212, 0.212, usevar='unequal'))
    visualization.visRandomVariable(albuminFirst)
    visualization.visRandomVariable(albumin)
    print('albuminFirst avg: ', albuminFirst.mean(), 'albumin avg: ', albumin.mean())
    print('albuminFirst std: ', np.std(albuminFirst), 'albumin std: ', np.std(albumin))
    print('sas')


def creatMaxCreatinineAvg(vte_old, vte_new):
    creatMax = vte_old['creatMax'][~np.isnan(vte_old['creatMax'])]
    creatinine = vte_new['creatinine'][~np.isnan(vte_new['creatinine'])]
    print(ttost_ind(creatMax, creatinine, -0.045, 0.045, usevar='unequal'))
    #visualization.visRandomVariable(creatMax)
    #visualization.visRandomVariable(creatinine)
    print('creatMax avg: ', creatMax.mean(), 'creatinine avg: ', creatinine.mean())
    print('creatMax std: ', np.std(creatMax), 'creatinine std: ', np.std(creatinine))
    print('sas')


def createTwoVarPred(combined):
    combined = combined.astype('float64')
    combined.loc[combined['malignant'] == 1, 'two var prediction'] = 1
    combined.loc[combined['vteprior'] == 1, 'two var prediction'] = 1
    combined['two var prediction'].fillna(0, inplace=True)
    return combined


def exploreSubsets(df):
    one_to_zero = df[(df['padua prediction'] == 1) & (df['two var prediction'] == 0)]
    zero_to_one = df[(df['padua prediction'] == 0) & (df['two var prediction'] == 1)]
    print('number of v=1 in zeroToOne:', zero_to_one['vt'].value_counts())
    print('number of v=1 in oneToZero:', one_to_zero['vt'].value_counts())
    print('number of b=1 in zeroToOne:', zero_to_one['bleed'].value_counts())
    print('number of b=1 in o neToZero:', one_to_zero['bleed'].value_counts())


def checkLabelSimilarities(vte_old, vte_new):
    X_old = featureSelection.select5vSFS(vte_old)
    y_old = vte_old['vt']
    y_old = y_old.to_numpy()
    X_new = featureSelection.select5vSFS(vte_new)
    y_new = vte_new['vt']
    # create knn
    neigh = NearestNeighbors(n_neighbors=1)
    # fit it to old data, only 5 relevant features
    neigh.fit(X_old)
    closest_from_old_data = neigh.kneighbors(X_new)[1]
    closest_vt_label = y_old[closest_from_old_data]
    print('confusion matrix that sums tha diffrent'
          ' labels distributions between the data sets', confusion_matrix(y_new, closest_vt_label))
    return None
