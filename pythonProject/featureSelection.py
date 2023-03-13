# Vadim Litvinov
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_selection import mutual_info_classif as MIC, SelectKBest, f_classif
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import helperFunctions
from sklearn.feature_selection import chi2
from scipy.stats import chisquare, chi2_contingency
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
from sklearn.metrics import matthews_corrcoef
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from sklearn.feature_selection import SequentialFeatureSelector

FEATURE_NUM = 20

# the number of p.v<0.05 features
#TOP_VT_SCORE_FEATURES = 819
# the number of p.v<0.01 features
#TOP_VT_SCORE_FEATURES = 497
# the number of p.v<0.001 features
#TOP_VT_SCORE_FEATURES = 329
# num of categorial features for vt
TOP_VT_CATEGORIAL = 13

# the number of p.v<0.05 features
#TOP_B_SCORE_FEATURES = 779
# the number of p.v<0.01 features
#TOP_B_SCORE_FEATURES = 16
# num of categorial features for b
TOP_B_CATEGORIAL = 20

ILLIGAL_DIAG_FEATURES = [ # (Venous thrombosis due to central venous access device)
                        'Comp-oth vasc dev/graft',
                        # thrombus)
                         'Vasc comp med care NEC', 'Abn react-procedure NEC', 'Ac embl suprfcl up ext', 'Embl suprfcl ves low ext', 'Abn react-surg proc NEC',
                        'Comp-ren dialys dev/grft', 'Surg comp-peri vasc syst', 'Vascular comp vessel NEC',
                        'min_PTT', 'max_PTT', 'mean_PTT',
                        # bleeding
                        'Ac posthemorrhag anemia', 'Chr blood loss anemia', 'Gastroduodenal dis NEC', 'Surg compl-heart',
                        'ADMISSION_ELECTIVE']


def combChiSquareAnova(df):
    removeIllegalFeatures(df)
    labels = ['vt', 'bleed']
    # only to categorial features
    frstloc = df.columns.get_loc("min_BUN")
    lastloc = df.columns.get_loc("max_WBC") + 1
    cat_Xv = df.iloc[:, :frstloc]
    cat_Xb = df.iloc[:, :frstloc]
    num_Xv = df.iloc[:, frstloc:lastloc]
    num_Xb = df.iloc[:, frstloc:lastloc]
    del frstloc, lastloc
    yv = df['vt']
    yb = df['bleed']
    selector_chi_v = SelectKBest(chi2, k=TOP_VT_CATEGORIAL)
    selector_chi_b = SelectKBest(chi2, k=TOP_B_CATEGORIAL)
    selector_anova_v = SelectKBest(f_classif, k=FEATURE_NUM-TOP_VT_CATEGORIAL)
    selector_anova_b = SelectKBest(f_classif, k=FEATURE_NUM-TOP_B_CATEGORIAL)
    selector_chi_v.fit(cat_Xv, yv)
    selector_chi_b.fit(cat_Xb, yb)
    selector_anova_v.fit(num_Xv, yv)
    selector_anova_b.fit(num_Xb, yb)

    # sort p values for debugging by viewing
    selector_chi_v.pvalues_.sort()
    selector_anova_v.pvalues_.sort()
    cols_chi_v = selector_chi_v.get_support(indices=True)
    cols_chi_b = selector_chi_b.get_support(indices=True)
    cols_anova_v = selector_anova_v.get_support(indices=True)
    cols_anova_b = selector_anova_b.get_support(indices=True)

    X_new_v = cat_Xv.iloc[:, cols_chi_v]
    X_new_b = cat_Xb.iloc[:, cols_chi_b]
    X_new_v = X_new_v.join(num_Xv.iloc[:, cols_anova_v])
    X_new_b = X_new_b.join(num_Xb.iloc[:, cols_anova_b])
    return X_new_v, X_new_b, yv, yb


def chiSquare(df, LABEL):
    removeIllegalFeatures(df)
    if LABEL == 'vt':
        top = TOP_VT_CATEGORIAL
    else:
        top = TOP_B_CATEGORIAL
    # only to categorial features
    frstloc = df.columns.get_loc("min_BUN")
    lastloc = df.columns.get_loc("mean_WBC") + 1
    X = df.iloc[:, :frstloc]
    y = df[LABEL]
    '''
    chi_scores = chi2(X, y)
    p_values = pd.Series(chi_scores[1], index=X.columns)
    p_values.sort_values(inplace=True)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(p_values)
    '''
    # select k best
    selector = SelectKBest(chi2, k=top)
    selector.fit(X, y)
    # sort p values for debugging by viewing
    selector.pvalues_.sort()
    cols = selector.get_support(indices=True)
    X_new = X.iloc[:, cols]
    # WITH NUMERIC FEATURES
    #checkMulticollinearity(X_new)
    X_new = X_new.join(df.iloc[:, frstloc:lastloc])
    return X_new, y


def anovaBeilinson(X, y, LABEL):
    if LABEL == 'vt':
        top = TOP_VT_CATEGORIAL
    else:
        top = TOP_B_CATEGORIAL
    selector = SelectKBest(f_classif, k=top)
    selector.fit(X, y)
    # sort p values for debugging by viewing
    selector.pvalues_.sort()
    cols = selector.get_support(indices=True)
    X_new = X.iloc[:, cols]
    return X_new


def anova(X, y, LABEL):
    if LABEL == 'vt':
        top = TOP_VT_CATEGORIAL
    else:
        top = TOP_B_CATEGORIAL
    if len(X) < 60000:
        #  only to categorial features
        frstloc = X.columns.get_loc("min_BUN")
        lastloc = X.columns.get_loc("mean_WBC") + 1
        X_new = X.iloc[:, frstloc:lastloc]
        #y = df[LABEL]
    else:
        X_new = X
        del X
    '''
    chi_scores = chi2(X, y)
    p_values = pd.Series(chi_scores[1], index=X.columns)
    p_values.sort_values(inplace=True)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(p_values)
    '''
    # select k best
    selector = SelectKBest(f_classif, k=top)
    selector.fit(X_new, y)
    # sort p values for debugging by viewing
    #selector.pvalues_.sort()
    cols = selector.get_support(indices=True)
    X_new = X_new.iloc[:, cols]

    # WITH NUMERIC FEATURES
    #checkMulticollinearity(X_new)
    if len(y) < 60000:
        X_new = X_new.join(X.iloc[:, :frstloc])
    return X_new


def checkMulticollinearity(X):
    # set figure size
    plt.figure(figsize=(10, 7))
    # Generate a mask to onlyshow the bottom triangle
    mask = np.triu(np.ones_like(X.corr(), dtype=bool))
    # generate heatmap
    sns.heatmap(X.corr(), annot=True, mask=mask, vmin=-0.1, vmax=0.35)
    plt.title('Correlation Coefficient Of Predictors')
    plt.show()

    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    # calculating VIF for each feature
    vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                       for i in range(len(X.columns))]
    print(vif_data)


def removeIllegalFeatures(X):
    X.drop(ILLIGAL_DIAG_FEATURES, axis=1, inplace=True)


def featureSelectionMI(X, y, clf, LABEL):
    X = X.to_numpy()
    y = y.to_numpy()
    # with all features
    X_train, X_test, y_train, y_test = tts(
        X, y, test_size=0.33
        , random_state=0
        , stratify=y
    )
    clf.fit(X_train, y_train)
    y_hat = clf.predict(X_test)
    print('ba of calssif with all features:', balanced_accuracy_score(y_test, y_hat))

    # with selected top mutual information features
    mi_scores = MIC(X, y, n_neighbors=1, random_state=0)
    # TODO need to choose top 50% mi features
    mi_score_selected_index = np.where(mi_scores > 0.005)[0]
    X_2 = X[:, mi_score_selected_index]
    X_train, X_test, y_train, y_test = tts(X_2, y, random_state=0, stratify=y)
    clf.fit(X_train, y_train)
    y_hat = clf.predict(X_test)
    print('ba of calssif with top mi features:', balanced_accuracy_score(y_test, y_hat))

    # MUTUAL INFORMATION
    column_labels = X.columns.tolist()
    mi = MIC(X, y, n_neighbors=1, random_state=0).tolist()
    labels_mi = list(zip(column_labels, mi))
    print('MI:', labels_mi)
    helperFunctions.Nmaxelements(labels_mi, TOP_VT_CATEGORIAL)


def allCombSelection(X, y):
    # reduce feature number for complexity reasons
    #X = anova(X, y, LABEL)
    logistic_regression = LogisticRegression(random_state=0, class_weight='balanced', max_iter=1000)
    efs1 = EFS(logistic_regression,
               min_features=9,
               max_features=10,
               scoring='balanced_accuracy',
               print_progress=True,
               cv=4, n_jobs=-1)
    efs1 = efs1.fit(X, y)
    print('Best score: %.2f' % efs1.best_score_)
    print('Best subset (indices):', efs1.best_idx_)
    print('Best subset (corresponding names):', efs1.best_feature_names_)


def sequentialSelection(X, y):
    print('in seq selection')
    n_features = 5
    lr = LogisticRegression(class_weight='balanced', random_state=0, max_iter=10000)
    sfs = SequentialFeatureSelector(lr, direction='forward', cv=5,
                                    scoring='balanced_accuracy', n_jobs=-1,
                                    n_features_to_select=n_features)
    sfs.fit(X, y)
    print("Top {} features by sfs : {} ".format(n_features, list(X.columns[sfs.get_support()])))
    return sfs.transform(X)


def select5vSFS(df):
    return df[['SEX 1-M 2-W', 'malignant', 'vteprior',
               'albumin',
               #'albuminFirst',
               'aids']].to_numpy()


def selectOnlyprevVtCancer(df):
    return df[['malignant', 'vteprior']].to_numpy()


def paduaVariables(df):
    return df[['yhrombophilia', 'malignant', 'vteprior', 'CPPS_ReducedMobility', 'surgerylast30d', 'CPPS_Over70', 'CPPS_HeartRespiratoryFailure',
               'CPPS_MIorCVA', 'CPPS_InfectionRheumatologicalDisorder', 'CPPS_Obesity', 'CPPS_OngoingHormonalTreatment']].to_numpy()


def select7bSFS(df):
    return df[['CHF', 'SEVERE', 'malignant', 'aids', 'creatinine', 'albumin', 'function 1-ind 2- unind']].to_numpy()


def test(df):
    return df[['DM2', 'aids', 'vteprior', 'albumin', 'malignant']]
