# Vadim Litvinov
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import featureSelection
import helperFunctions
import statisticalTests
import visualization
import calibration


def stratifiedKfold(Xv, Xb, yv, yb):
    # stratify by two labels
    comb_lbls = yv.astype(str) + '_' + yb.astype(str)
    yv = yv.to_numpy()
    yb = yb.to_numpy()
    v_clf = LogisticRegression(random_state=0, class_weight='balanced',
                               max_iter=10000)
    b_clf = LogisticRegression(random_state=0, class_weight='balanced',
                               max_iter=10000)

    print('before creating folds')
    lstv_accu_stratified = []
    lstb_accu_stratified = []
    CI_lst = []
    k = 5
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=1)
    for train_index, test_index in skf.split(Xv, comb_lbls):
        #features for both v and b
        xv_train_fold, xv_test_fold = Xv[train_index], Xv[test_index]
        xb_train_fold, xb_test_fold = Xb[train_index], Xb[test_index]
        # different y for v and for b
        yv_train_fold, yv_test_fold = yv[train_index], yv[test_index]
        # check stratification is working
        #print(np.unique(yv_train_fold, return_counts=True), np.unique(yv_test_fold, return_counts=True))
        yb_train_fold, yb_test_fold = yb[train_index], yb[test_index]
        #print(np.unique(yb_train_fold, return_counts=True), np.unique(yb_test_fold, return_counts=True))
        print('starts fitting the models')
        v_clf.fit(xv_train_fold, yv_train_fold)
        print('finished training first')
        b_clf.fit(xb_train_fold, yb_train_fold)
        print('finished fitting the models and before creating proba')
        # create predictions for checking balanced accuracy
        yv_hat = v_clf.predict(xv_test_fold)
        yb_hat = b_clf.predict(xb_test_fold)
        # create proba for ACU
        yv_proba = v_clf.predict_proba(xv_test_fold)
        yb_proba = b_clf.predict_proba(xb_test_fold)
        print('avg p(v|x) for set V: ', helperFunctions.avgProba(yv_proba, yv_test_fold))
        print('avg p(v|x) for set B: ', helperFunctions.avgProba(yv_proba, yb_test_fold))
        print('avg p(b|x) for set B: ', helperFunctions.avgProba(yb_proba, yb_test_fold))
        print('avg p(b|x) for set V: ', helperFunctions.avgProba(yb_proba, yv_test_fold))
        print('v ba:', balanced_accuracy_score(yv_test_fold, yv_hat))
        #print('auc v:', roc_auc_score(yv_test_fold, yv_proba))
        print('b ba:', balanced_accuracy_score(yb_test_fold, yb_hat))
        #print('auc b:', roc_auc_score(yb_test_fold, yb_proba))
        #print(clf.score(x_test_fold, y_test_fold))
        lstv_accu_stratified.append(balanced_accuracy_score(yv_test_fold, yv_hat))
        lstb_accu_stratified.append(balanced_accuracy_score(yb_test_fold, yb_hat))

        # CALIBRATION CHECK AND PLOT
        #y_proba = clf.predict_proba(x_test_fold)[:, 1]
        # hesmer lemeshaw test
        #print(statisticalTests.HosmerLemeshow(yv_proba, yv_test_fold))
        # show reliability curve before calibration
        #calibration.visCalibration(yv_test_fold, yv_proba[:, 1])
        # calibrate clasiffiers
        #cal_yv_proba = calibration.calibrate(yv_train_fold, v_clf.predict_proba(xv_train_fold)[:, 1], yv_proba[:, 1])
        # sow curve after calibration
        #calibration.visCalibration(yv_test_fold, cal_yv_proba)

        #cal_yb_proba = calibration.calibrate(yb_train_fold, b_clf.predict_proba(xb_train_fold)[:, 1], yb_proba[:, 1])

        #calibration.visCalibration(yv_test_fold, cal_yv_proba[:, 1])
        calibration.visCalibration(yv_test_fold, yv_proba[:, 1])
        calibration.visCalibration(yb_test_fold, yb_proba[:, 1])
        #calibration.visCalibration(yb_test_fold, cal_yb_proba[:, 1])
        #print('avg p(v|x) for set V: ', helperFunctions.avgProba(cal_yv_proba, yv_test_fold))
        #print('avg p(v|x) for set B: ', helperFunctions.avgProba(cal_yv_proba, yb_test_fold))
        #print('avg p(b|x) for set B: ', helperFunctions.avgProba(cal_yb_proba, yb_test_fold))
        #print('avg p(b|x) for set V: ', helperFunctions.avgProba(cal_yb_proba, yv_test_fold))
        # STATISTICAL TESTS FOR ACU DIFFRENCES
        #CIb, CI0, CIv = statisticalTests.statisticalAnalysis(yv_proba[:, 1], yb_proba[:, 1], yv_test_fold, yb_test_fold)
        #CI_lst.append(statisticalTests.statisticalAnalysis(corrected_proba_v, corrected_proba_b, yv_test_fold, yb_test_fold))
        # TODO FOR THESIS
        CI_lst.append(statisticalTests.statisticalAnalysis(yv_proba[:, 1], yb_proba[:, 1], yv_test_fold, yb_test_fold))
        # visualize proba
        #visualization.visProba(yv_test_fold, yb_test_fold, yv_proba[:, 1], yb_proba[:, 1])
        #visualization.visProba(yv_test_fold, yb_test_fold, cal_yv_proba[:, 1], cal_yb_proba[:, 1])
    visualization.visConfidenceIntervals(CI_lst)
    '''
    print('List of possible v accuracy:', lstv_accu_stratified)
    print('List of possible  b accuracy:', lstb_accu_stratified)
    print('Overall Mean v Accuracy:',
          np.mean(lstv_accu_stratified) * 100, '%')
    print('Overall Mean b Accuracy:',
          np.mean(lstb_accu_stratified) * 100, '%')
    print('Standard Deviation is:', np.std(lstv_accu_stratified))
    print('Standard Deviation is:', np.std(lstb_accu_stratified), '\n')
    '''


def logisticRegression(df):
    print('before mean/median impute')
    # features and labels
    #X = df.drop(['vt', 'bleed', 'v and b'], axis=1)
    #y = df[LABEL]
    #kfold_clf = LogisticRegression(random_state=0, class_weight='balanced', max_iter=10000)
    print('after creating clf and before kfold')
    # categorial feature selection for vt
    yv = df['vt']
    yb = df['bleed']
    del df['vt'], df['bleed'], df['v and b']
    Xv = featureSelection.select5vSFS(df)
    print(Xv.shape)
    #Xb = featureSelection.sequentialSelection(df, yb)
    Xb = featureSelection.select7bSFS(df)
    print(Xb.shape)
    stratifiedKfold(Xv, Xb, yv, yb)
    # Create lists
    #print(clf.coef_, clf.intercept_)
    #print(X.columns)
    #print('score on train:', clf.score(X, y))
    #mi_score = MIC(X, y)
    