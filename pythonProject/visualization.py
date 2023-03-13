# Vadim Litvinov
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
import matplotlib.lines as mlines
import helperFunctions

#LABEL = 'na'


def createRlvntSamples(yv, yb, yv_proba, yb_proba):
    mask = (yv == 1) | (yb == 1)
    y = yv[mask]
    #maskb = (yb == 1) | ()
    yv_proba2 = yv_proba[mask]
    yb_proba2 = yb_proba[mask]
    return yv_proba2, yb_proba2, y


def prepareForPlot3Groups(yv, yb):
    yb[yb == 1] = 2
    y = yv + yb
    return y


def visProba(yv, yb, yv_proba, yb_proba):
    # create one vector of 1=v, 2=b
    yv_rlvnt, yb_rlvnt, y = createRlvntSamples(yv, yb, yv_proba, yb_proba)
    yv_proba = yv_rlvnt.flatten()
    yb_proba = yb_rlvnt.flatten()
    colors = ['red', 'blue']
    # visualize all
    #y = prepareForPlot3Groups(yv, yb)
    #colors = ['grey', 'blue', 'red']
    plt.scatter(yv_proba, yb_proba, c=y, cmap=matplotlib.colors.ListedColormap(colors), s=7)
    plt.show()
    ax = sns.jointplot(x=yv_proba, y=yb_proba,
                       color='royalblue', kind='kde', hue=y, fill=True, joint_kws={'alpha': 0.8})
    ax.set_axis_labels('P(v|X)', 'P(b|X)')
    ax.fig.suptitle('Probability space', fontsize=10, verticalalignment='bottom')
    plt.show()


def visCalibration(prob_true, prob_pred):
    fig1, ax = plt.subplots()
    # only these two lines are calibration curves
    plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label='Classifier')
    # reference line, legends, and axis labels
    line = mlines.Line2D([0, 1], [0, 1], color='black')
    #transform = ax.transAxes
    #line.set_transform(transform)
    ax.add_line(line)
    fig1.suptitle('Calibration plot')
    #ax.set_yscale('log')
    #ax.set_xscale('log')
    ax.set_xlabel('Predicted probability')
    ax.set_ylabel('True probability')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.legend()
    plt.show()


def countLabel1(df, LABEL):
    total = len(df)
    df = df[df[LABEL] == 1]
    minority = len(df)
    majority = total - minority
    return round(majority / minority)


def replicate(df, rep_times, LABEL):
    #[df['vt'] == 1] * rep_times
    minority_repeated = pd.concat([df[df[LABEL] == 1]] * (rep_times - 1), ignore_index=True)
    df = pd.concat([df, minority_repeated], ignore_index=True)
    return df


def visualizeCharteventsVars(df, LABEL):
    # drop one problematic all 999999 values admission
    #helperFunctions.drop999999(df)
    # for max_Sodium (serum)
    helperFunctions.dropAnomalies(df)
    '''
    df.drop([188721], inplace=True)
    df.drop([109597], inplace=True)
    df.drop([169880], inplace=True)
    df.drop([151021], inplace=True)
    '''

    '''
    df.drop([175676], inplace=True)
    df.drop([169757], inplace=True)
    df.drop([158767], inplace=True)
    df.drop([108153], inplace=True)
    df.drop([179544], inplace=True)
    df.drop([152254], inplace=True)
    df.drop([167812], inplace=True)

    df.drop([120455], inplace=True)
    '''
    # drop nas in target variable
    df_nona = df[df['WEIGHT'].notna()]
    df_nona = df_nona.sort_values(by='WEIGHT')
    # check what factor we should replicate minority class
    replicate_factor = countLabel1(df_nona, LABEL)
    df_nona = replicate(df_nona, replicate_factor, LABEL)
    rep_df = df_nona[[LABEL, 'WEIGHT', 'AGE']]
    # sort so minority label will be on top
    rep_df = rep_df.sort_values(by=[LABEL])
    '''
    for col1 in reversed(df.columns):
        for col2 in reversed(df.columns):
            fusion = pd.DataFrame()
            fusion['vt'] = df['vt']
            fusion[col1] = df[col1]
            fusion[col2] = df[col2]
    '''
    sns.pairplot(rep_df, hue=LABEL, kind='scatter', diag_kind="kde"
                 )
    # if we want bin sizes on diagonal:  diag_kws={'alpha': 0.55, 'bins': 50}
    plt.show()


def visRandomVariable(samples):
    # convert to series for visualization
    samples = pd.Series(samples)
    samples.plot.hist(bins=6)
    #plt.bar(samples, height=len(samples))
    plt.show()


def visConfidenceIntervals(CI_lst):
    for fold, y in zip(CI_lst, range(1, len(CI_lst)+1)):
        data_dict = {}
        #data_dict['category'] = ['bleeding', 'healthy', 'vte']
        data_dict['lower'] = [fold[0][0], fold[1][0], fold[2][0]]
        data_dict['upper'] = [fold[0][1], fold[1][1], fold[2][1]]
        c = ['red', 'green', 'blue']
        dataset = pd.DataFrame(data_dict)
        for lower, upper, color in zip(dataset['lower'], dataset['upper'], c):
            plt.plot((lower, upper), (y, y), '|-', color=color, markeredgewidth=1.5)
            plt.plot((lower + upper)/2, y, '.', color='black', markersize=4.5)
        #matplotlib.axes.Axes.set_ylabel('fold number')
        #matplotlib.axes.Axes.set_xlabel('Anti Coagulant Utility')
        plt.yticks(range(1, len(CI_lst)+1))
    plt.show()
