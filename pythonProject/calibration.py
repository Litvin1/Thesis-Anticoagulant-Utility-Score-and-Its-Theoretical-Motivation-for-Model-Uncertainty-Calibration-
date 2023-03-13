# Vadim Litvinov
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression as IR
import visualization
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression


def lossCorrectedConfidenceScore(prob_pred, beta_1):
    nom = (1-beta_1) * prob_pred
    denom = beta_1 + (1-2*beta_1)*prob_pred
    return nom / denom


def expectedCalibrationError(prob_true, prob_pred):
    diff = abs(prob_true - prob_pred)
    summa = sum(diff)
    ece = summa * (1/len(prob_true))
    print('ECE:', ece)


def visCalibration(y_true, y_proba_pred):
    num_positives = np.count_nonzero(y_true)
    num_negetives = len(y_true) - num_positives
    weight1 = len(y_true) / (2 * num_positives)
    weight0 = len(y_true) / (2 * num_negetives)
    beta1 = weight1 / (weight1 + weight0)
    nBins = 10
    correctedProba = lossCorrectedConfidenceScore(y_proba_pred, beta1)

    prob_true, prob_pred = calibration_curve(y_true, y_proba_pred, n_bins=nBins, strategy='quantile')
    #correctedProba = lossCorrectedConfidenceScore(prob_pred, 0.01)
    prob_true_cor, prob_pred_cor = calibration_curve(y_true, correctedProba, n_bins=nBins, strategy='quantile')

    expectedCalibrationError(prob_true, prob_pred)
    expectedCalibrationError(prob_true_cor, prob_pred_cor)
    visualization.visCalibration(prob_true, prob_pred)
    visualization.visCalibration(prob_true_cor, prob_pred_cor)
    return correctedProba


def calibrate(y_true, y_train_proba_pred, y_test_proba):
    #y_train_proba_pred = y_train_proba_pred.reshape(-1, 1)
    y_test_proba = y_test_proba.reshape(-1, 1)
    #calibrated = CalibratedClassifierCV(clf, method='isotonic', cv='prefit').fit(X, y_true)
    #lr = LogisticRegression(random_state=0, class_weight='balanced', penalty='none').fit(y_train_proba_pred, y_true)
    ir = IR().fit(y_train_proba_pred, y_true)
    #return lr.predict_proba(y_test_proba)
    return ir.transform(y_test_proba)
    #return calibrated.predict_proba(X)
