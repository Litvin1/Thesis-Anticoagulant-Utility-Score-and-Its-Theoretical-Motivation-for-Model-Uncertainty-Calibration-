# Vadim Litvinov
# from numpy import genfromtxt
import numpy as np
import pandas as pd
import antiCoagulantsUtility
import beilinsonLoadPreprocess
import helperFunctions
import loadAndPreprocess
import normalizationImputation
import preprocessData
import visualization
import classifier
import featureSelection

LABEL = 'vt'
# LABEL = 'bleed'
PADUA_FLAG = 1

if __name__ == '__main__':
    '''
    # load create
    no_normalization, min_max, standartized = loadAndPreprocess.loadProcessedData()
    helperFunctions.negaToPos(no_normalization)
    
    # drop v=1, b=1 samples and check
    no_normalization = preprocessData.delvAndb(no_normalization)
    min_max = preprocessData.delvAndb(min_max)
    standartized = preprocessData.delvAndb(standartized)
    print('v and b count:', np.sum(no_normalization['v and b'] == 2))
    print('v and b count:', np.sum(min_max['v and b'] == 2))
    print('v and b count:', np.sum(standartized['v and b'] == 2))

    print('shape of created data', no_normalization.shape, '\n')
    print('shape of created data', min_max.shape, '\n')
    print('shape of created data', standartized.shape, '\n')

    # vizualize
    #visualization.visualizeCharteventsVars(no_normalization, LABEL)

    #featureSelection.featureSelectionANOVA(no_normalization, LABEL)
    #featureSelection.featureSelectionANOVA(min_max, LABEL)
    #featureSelection.featureSelectionANOVA(standardized, LABEL)

    # try 3 different, no normalization, minmax and standartize

    #classifier.logisticRegression(no_normalization, LABEL)
    #classifier.logisticRegression(min_max, LABEL)
    #classifier.logisticRegression(standartized, LABEL)
    
    antiCoagulantsUtility.logisticRegression(no_normalization)
    antiCoagulantsUtility.logisticRegression(min_max)
    #antiCoagulantsUtility.logisticRegression(standartized)
    '''

    '''
    # work with beilinson old+new
    vte_old, vte_new = beilinsonLoadPreprocess.loadPreprocess(PADUA_FLAG)
    #print(vte_new['vt'].value_counts())
    #print(vte_new['bleed'].value_counts())
    #beilinsonLoadPreprocess.checkPaduaPerformance(vte_new)
    #visualization.visualizeCharteventsVars(beilinson, LABEL)
    #beilinson = beilinson[['bicarbonate', 'd dimer', 'vt']]
    #helperFunctions.orderFeatures(beilinson)
    #visualization.visualizeCharteventsVars(beilinson, LABEL)
    preprocessData.renameCols(vte_old)
    #print(vte_old.isnull().sum()*100/len(vte_old))
    #print(vte_new.isnull().sum()*100/len(vte_new))
    combined = beilinsonLoadPreprocess.combineAndImpute(vte_old, vte_new, PADUA_FLAG)
    # find where the problematic '-' is
    #print(np.where(combined.values == '-'))  # [(3, 'C')]
    print(combined['vt'].value_counts())
    print(combined['bleed'].value_counts())
    print('v and b count:', np.sum(combined['v and b'] == 2))
    # delete v and b samples only when doing the ACU model!
    #combined = preprocessData.delvAndb(combined)
    print(combined['vt'].value_counts())
    print('v and b count:', np.sum(combined['v and b'] == 2))
    #combined = normalizationImputation.normalizeMinMax(combined)
    #combined = normalizationImputation.standartize(combined)
    #beilinsonLoadPreprocess.checkPaduaPerformance(combined)
    beilinsonLoadPreprocess.createPaudaPred(combined)
    #classifier.twoVarPadua(combined)
    #classifier.twoVarPadua(combined)
    classifier.logisticRegression(combined, LABEL)
    #classifier.logisticRegression(vte_old, LABEL)
    #classifier.logisticRegression(vte_new, LABEL)
    #antiCoagulantsUtility.logisticRegression(combined)
    '''

    '''
    # work with beilinson old
    # padua related flag is 1 only when combing the two datasets and comparing to padua
    vte_old, vte_new = beilinsonLoadPreprocess.loadPreprocess(padua_related_flag=0)
    # check performance and create prediction vector
    beilinsonLoadPreprocess.createPaudaPred(vte_old)
    #classifier.twoVarPadua(vte_old)
    classifier.logisticRegression(vte_old, LABEL)
    '''
    # only the subset of the ones who had padua score from the new set
    vte_old, vte_new = beilinsonLoadPreprocess.loadPreprocess(padua_related_flag=0)
    # print(vte_new['vt'].value_counts())
    beilinsonLoadPreprocess.createPaudaPred(vte_new)
    classifier.trainValidationTestLR(vte_old, vte_new)
    # classifier.logisticRegression(vte_new, LABEL)

    '''
    # work with beilinson old + FULL PADUA VARIABLES beilinson new only vte prediction
    combined = beilinsonLoadPreprocess.fullPaduaVariablesLoadPreprocessCombine()
    print(combined['vt'].value_counts())
    classifier.logisticRegression(combined, LABEL)
    #classifier.twoVarPadua(combined)
    '''
