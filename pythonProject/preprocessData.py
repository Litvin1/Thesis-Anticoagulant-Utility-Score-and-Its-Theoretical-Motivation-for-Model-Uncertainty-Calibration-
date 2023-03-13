# Vadim Litvinov
import pandas as pd
import numpy as np
import time

PERCENT_MISS = 0.35


def countRelevantDiagnoses(data, diagnoses, len_all_diagnoses, diagnosis_flag):
    count = 0
    count_percent = 0
    all_relevant_patients = pd.DataFrame()
    for d in diagnoses:
        if diagnosis_flag:
            relevant_patients = data[data.ICD9_CODE == d]
        else:
            relevant_patients = data[data.DRUG == d]
        all_relevant_patients = all_relevant_patients.append(relevant_patients)
        # show percent, 2 digits after the point
        count_percent += (len(relevant_patients)/len_all_diagnoses)*100
        print('diagnosis:', d, 'count:', len(relevant_patients), 'percent from total:', format((len(relevant_patients)/len_all_diagnoses)*100, '.2f'), '%')
        count += len(relevant_patients)
        print(count)
        print(count_percent)
    print('relevant diagnoses number:', count, len(all_relevant_patients))
    # number of admissions with one or more of the diagnoses
    print('relevant admissions number:', len(all_relevant_patients['HADM_ID'].unique()))
    #print(all_relevant_patients)


def dropUnneededColumns(data):
    #data.drop(data.columns.difference(['HADM_ID', 'vt', 'bleed', 'proph']), 1, inplace=True, errors='ignore')
    data.drop(columns=['ROW_ID', 'SUBJECT_ID', 'ADMITTIME', 'DISCHTIME', 'DEATHTIME', 'ADMISSION_LOCATION', 'DISCHARGE_LOCATION', 'LANGUAGE', 'EDREGTIME', 'EDOUTTIME', 'DIAGNOSIS', 'HOSPITAL_EXPIRE_FLAG', 'HAS_CHARTEVENTS_DATA',
                       'ICUSTAY_ID', 'CHARTTIME', 'STORETIME', 'CGID', 'VALUE', 'ERROR', 'RESULTSTATUS', 'STOPPED',
                       'ABBREVIATION', 'DBSOURCE', 'LINKSTO', 'CATEGORY', 'UNITNAME', 'PARAM_TYPE', 'CONCEPTID',
                       'SEQ_NUM'], inplace=True, errors='ignore')


def addBinaryLabel(data, diagnoses_list, flag):
    #data['vt'] = np.where(data['ICD9_CODE'] in diagnoses_list, '1', '0')
    # add vte label
    if flag == 1:
        data['vt'] = data['ICD9_CODE'].isin(diagnoses_list)
        #print(data[data['vt'] == True])
        data['vt'] = data['vt'].astype(int)
        # add bleed label
    elif flag == 2:
        data['bleed'] = data['ICD9_CODE'].isin(diagnoses_list)
        #print(data[data['bleed'] == True])
        data['bleed'] = data['bleed'].astype(int)
        # add proph label
    else:
        data['proph'] = data['DRUG'].isin(diagnoses_list)
        #print(data[data['anticoagulants'] == True])
        data['proph'] = data['proph'].astype(int)


def createDiagnosisFeatures(diagnoses, d_diagnoses, vt_lst, bleed_lst):
    # first merge, so SHORT_TITLE will be next to ICD9_CODE
    # drop all vt and bleed diagnoses
    diagnoses_no_lbl = diagnoses[~diagnoses['ICD9_CODE'].isin(vt_lst)]
    diagnoses_no_lbl = diagnoses_no_lbl[~diagnoses_no_lbl['ICD9_CODE'].isin(bleed_lst)]
    # 16,338 diagnoses dropped because not in d_diagnoses
    diagnoses_no_lbl = pd.merge(diagnoses_no_lbl, d_diagnoses, on=['ICD9_CODE'])
    # create features out of 'SHORT_TITLE', and aggregate by 'HADM_ID'
    diag_pvt_tble = diagnoses_no_lbl.pivot_table(values='ICD9_CODE', index='HADM_ID', columns='SHORT_TITLE',
                                        aggfunc='count')
    # pvt_tble has 575,784 less rows than diagnosis because of multiple diagnoses per admission
    diag_pvt_tble = diag_pvt_tble.fillna(0)
    # convers all to binary dummy features
    diag_pvt_tble = diag_pvt_tble.astype(bool)
    diag_pvt_tble = diag_pvt_tble.astype(int)
    diag_pvt_tble.reset_index(inplace=True)
    return diag_pvt_tble


def createVandBLabel(df):
    df['vt'] = df['vt'].astype(bool)
    df['bleed'] = df['bleed'].astype(bool)
    #df['proph'] = df['proph'].astype(bool)
    df['vt'] = df['vt'].astype(int)
    df['bleed'] = df['bleed'].astype(int)
    #df['proph'] = df['proph'].astype(int)
    sum_column = df['vt'] + df['bleed']
    df['v and b'] = sum_column
    #sum_column = df['bleed'] + df['proph']
    #df['b and p'] = sum_column
    #sum_column = df['vt'] + df['proph']
    #df['v and p'] = sum_column
    return df


def createCharteventsFeatures(chartevents):
    # NEW VERSION
    print(chartevents.shape)
    print('before pvt tble')
    pvt_tble = chartevents.pivot_table(
        values='VALUENUM', index='HADM_ID', columns='LABEL', aggfunc=['min', 'max', np.mean])
    pvt_tble.columns = pvt_tble.columns.map(('_'.join))
    pvt_tble.reset_index(inplace=True)
    print('after pvt tble')
    print(pvt_tble.shape)
    #chartevents = chartevents.astype(str)
    return pvt_tble


def dropColumnsToAll(admissions, chartevents, d_items, diagnoses_icd, d_diagnoses):
    dropUnneededColumns(admissions)
    dropUnneededColumns(chartevents)
    dropUnneededColumns(d_items)
    dropUnneededColumns(diagnoses_icd)
    dropUnneededColumns(d_diagnoses)


def removeNewborn(data):
    return data[data.ADMISSION_TYPE != 'NEWBORN']
    #return data.reset_index()


def removeColsNansLabel1(df):
    df_vt1 = df[df['vt'] == 1]
    min_count = np.round((1-PERCENT_MISS) * df_vt1.shape[0])
    suff_col = df_vt1.dropna(axis=1, thresh=min_count).columns
    df_suff_col = df[suff_col]
    del df_vt1, min_count, suff_col
    return df_suff_col


def removeColsNans(df):
    df['pauda score'].fillna(df['pauda score'].median(), inplace=True)
    min_count = np.round((1-PERCENT_MISS) * df.shape[0])
    # df.set_index('HADM_ID', inplace=True)
    return df.dropna(axis=1, thresh=min_count)


def categoricalToOneHot(df):
    df = pd.concat([df, pd.get_dummies(df.ADMISSION_TYPE, prefix='ADMISSION')], axis=1)
    df.drop(columns=['ADMISSION_TYPE'], inplace=True)
    df = pd.concat([df, pd.get_dummies(df.INSURANCE, prefix='INSURANCE')], axis=1)
    df.drop(columns=['INSURANCE'], inplace=True)
    # make all not specific religions NA
    df['RELIGION'].replace({'UNOBTAINABLE': pd.NA, 'NOT SPECIFIED': pd.NA, 'OTHER': pd.NA}, inplace=True)
    df = pd.concat([df, pd.get_dummies(df.RELIGION, prefix='RELIGION')], axis=1)
    df.drop(columns=['RELIGION'], inplace=True)
    # make marital status unknown na
    df['MARITAL_STATUS'].replace({'UNKNOWN (DEFAULT)': pd.NA}, inplace=True)
    df = pd.concat([df, pd.get_dummies(df.MARITAL_STATUS, prefix='MARITAL_STATUS')], axis=1)
    df.drop(columns=['MARITAL_STATUS'], inplace=True)
    # make na
    df['ETHNICITY'].replace({'OTHER': pd.NA, 'PATIENT DECLINED TO ANSWER': pd.NA, 'UNABLE TO OBTAIN': pd.NA,
                             'UNKNOWN/NOT SPECIFIED': pd.NA}, inplace=True)
    df = pd.concat([df, pd.get_dummies(df.ETHNICITY, prefix='ETHNICITY')], axis=1)
    df.drop(columns=['ETHNICITY'], inplace=True)
    return df


def intToBinaryFeature(df, LABEL):
    df[LABEL] = df[LABEL].astype(bool)
    df[LABEL] = df[LABEL].astype(int)
    return df


def delvAndb(df):
    return df[df['v and b'] != 2]


def renameCols(vte_old):
    vte_old.rename(columns={'albuminFirst': 'albumin', 'creatMax': 'creatinine'}, inplace=True)
