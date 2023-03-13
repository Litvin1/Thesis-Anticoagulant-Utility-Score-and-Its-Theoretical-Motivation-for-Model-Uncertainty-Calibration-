# Vadim Litvinov
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_classif as MIC, SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split as tts
from sklearn.neural_network import MLPClassifier
from deployml.sklearn import LogisticRegressionBase
from sklearn.utils import class_weight
from sklearn.model_selection import ParameterGrid
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss
import parfit.parfit as pf
import calibration
import featureSelection
import helperFunctions
from scipy import stats
import normalizationImputation
import statisticalTests
import visualization
from main import LABEL


def stratifiedKfold(clf, X, y, padua_pred, padua_score):
    #clf_features = X.columns
    print('before creating folds')
    lst_accu_stratified = []
    padua_lst_accu_stratified = []
    lst_auc_stratified = []
    padua_lst_auc_stratified = []
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    for train_index, test_index in skf.split(X, y):
        #train_index.reshape(-1)
        # cut the padua performance only on the test set
        padua_test_pred = padua_pred[test_index]
        padua_test_score = padua_score[test_index]
        x_train_fold, x_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]
        print('starts fitting the model')
        clf.fit(x_train_fold, y_train_fold)
        print('finished fitting the model and before creating predictions')
        y_hat = clf.predict(x_test_fold)
        print('ba:', balanced_accuracy_score(y_test_fold, y_hat))
        #print(clf.score(x_test_fold, y_test_fold))
        lst_accu_stratified.append(balanced_accuracy_score(y_test_fold, y_hat))
        padua_lst_accu_stratified.append(balanced_accuracy_score(y_test_fold, padua_test_pred))
        # for calibration
        y_proba = clf.predict_proba(x_test_fold)[:, 1]
        print('auc:', roc_auc_score(y_test_fold, y_proba))
        lst_auc_stratified.append(roc_auc_score(y_test_fold, y_proba))
        padua_lst_auc_stratified.append(roc_auc_score(y_test_fold, padua_test_score))
        print('precision:', precision_score(y_test_fold, y_hat))
        print('recall:', recall_score(y_test_fold, y_hat))
        print(confusion_matrix(y_hat, y_test_fold))
        # CALIBRATION CHECK AND PLOT
        #calibration.calibrate(y_test_fold, y_proba)
    # fit to all data, to see the coefficients
    clf.fit(X, y)
    print('classification function learned on all DATA: ', clf.intercept_, clf.coef_)
    print('ba test:')
    statisticalTests.pairedTTest(lst_accu_stratified, padua_lst_accu_stratified)
    print('auc test:')
    statisticalTests.pairedTTest(lst_auc_stratified, padua_lst_auc_stratified)
    #visualization.visRandomVariable(lst_accu_stratified)
    #visualization.visRandomVariable(lst_auc_stratified)
    print('List of possible accuracy:', lst_accu_stratified)
    print('List of possible auc:', lst_auc_stratified)
    print('Overall Mean Accuracy:',
          np.mean(lst_accu_stratified))
    print('Overall Mean AUC:',
          np.mean(lst_auc_stratified))
    print('Standard Deviation is:', np.std(lst_accu_stratified), '\n')


def logisticRegression(df, LABEL):
    # reset indexes from admission_num, for the cross validation
    #df.reset_index(inplace=True)
    # drop admission_id columnklk
    #df.drop(columns=['HADM_ID'], inplace=True)
    # MEDIAN IMPUTATION
    helperFunctions.imputeMedianBeilinson(df)
    print('after impute and before features labels split')
    # features and labels
    #X = df.drop(['vt', 'bleed', 'v and b'], axis=1)
    #y = df[LABEL]
    kfold_clf = LogisticRegression(random_state=0, class_weight='balanced', max_iter=10000)
    #kfold_clf = SGDClassifier(penalty='l1', alpha=0.0001, random_state=0, loss='log', class_weight='balanced', early_stopping=False, validation_fraction=0.1, n_iter_no_change=100, verbose=False)
    #kfold_clf = LogisticRegressionBase(penalty='l1')
    print('after creating clf and before kfold')
    #X = df.drop(columns=['vt'])
    X = df.drop(columns=['bleed', 'vt', 'v and b'])
    y = df[LABEL]
    #X = featureSelection.anovaBeilinson(X, y, LABEL)
    padua_score, padua_pred = helperFunctions.paduaPredExportAndDelete(X)
    #X = featureSelection.sequentialSelection(X, y)
    X = featureSelection.paduaVariables(X)
    #X = featureSelection.select5vSFS(X)
    #X = featureSelection.selectOnlyprevVtCancer(X)
    #X = featureSelection.select7bSFS(X)

    y, padua_score, padua_pred = helperFunctions.dataToNumpy(y, padua_score, padua_pred)
    stratifiedKfold(kfold_clf, X, y, padua_pred, padua_score)
    # Create lists
    #print(clf.coef_, clf.intercept_)
    #print(X.columns)
    #print('score on train:', clf.score(X, y))


def twoVarPadua(df):
    helperFunctions.imputeMedianBeilinson(df)
    df = helperFunctions.createTwoVarPred(df)
    y = df['vt']
    paduaPred = df['padua prediction']
    print('ba PADUA:', balanced_accuracy_score(y, paduaPred))
    twoVarPred = df['two var prediction']
    print('ba 2Var:', balanced_accuracy_score(y, twoVarPred))
    helperFunctions.exploreSubsets(df)


def trainValidationTestLR(vte_old, vte_new):
    # impute median to both. no data leakage
    helperFunctions.imputeMedianBeilinson(vte_old)
    helperFunctions.imputeMedianBeilinson(vte_new)
    # normalize for the KNN in next stage
    vte_old = normalizationImputation.minMaxScalar(vte_old)
    vte_new = normalizationImputation.minMaxScalar(vte_new)
    # check if the label distributions between old and new data is similar
    helperFunctions.checkLabelSimilarities(vte_old, vte_new)

    clf = LogisticRegression(random_state=0, class_weight='balanced', max_iter=10000)
    # lebel vector for train
    y_train = vte_old['vt']
    # drop labels and 'pauda score'
    X_train = vte_old.drop(columns=['bleed', 'vt', 'v and b', 'pauda score'])
    X_train = featureSelection.sequentialSelection(X_train, y_train)

    X_test = vte_new.drop(columns=['bleed', 'vt', 'v and b'])
    X_test = featureSelection.test(X_test)
    y_test = vte_new['vt']
    #X_test = featureSelection.sequentialSelection(X_test, y_test)
    print(y_test.value_counts())
    #check padua performance on the test set from new data
    print('ba PADUA:', balanced_accuracy_score(y_test, vte_new['padua prediction']))
    print('auc PADUA:', balanced_accuracy_score(y_test, vte_new['pauda score']))

    clf.fit(X_train, y_train)
    #clf.fit(X_test, y_test)
    y_hat = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)
    print('ba:', balanced_accuracy_score(y_test, y_hat))
    print('auc:', roc_auc_score(y_test, y_proba[:, 1]))
