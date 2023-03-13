# Vadim Litvinov
import helperFunctions
import preprocessData
import pandas as pd
import loadingRawData
import normalizationImputation

LABEL_TYPE_VT = loadingRawData.BOOK_VT_WUPPER_WOCHRON
LABEL_TYPE_B = loadingRawData.PAPER_MJR_BLD

PATH_NO_NORMALIZ = '/data/old_data/vadim/no_normalization_dataset' + str(preprocessData.PERCENT_MISS) + 'miss.csv'
PATH_MINMAX_NORMALIZ = '/data/old_data/vadim/minmax_normalization_dataset' + str(preprocessData.PERCENT_MISS) + 'miss.csv'
PATH_STANDARTIZED = '/data/old_data/vadim/standartized_dataset' + str(preprocessData.PERCENT_MISS) + 'miss.csv'


def loadPreprocess():
    #dictionary_icd = csvToPd('D_ICD_DIAGNOSES.csv')
    # load all raw data and remove newborn population
    admissions, chartevents, d_items, diagnoses, d_diagnoses = loadingRawData.loadAll()
    admissions = preprocessData.removeNewborn(admissions)
    #prescriptions = csvToPd('PRESCRIPTIONS.csv')
    #count icd9_code values
    #val_counter = diagnosis_icd['ICD9_CODE'].value_counts().rename_axis('ICD9_CODE').reset_index(name='counts')
    # merge with the long title and make it a csv file
    #toCsv = pd.merge(val_counter, dictionary_icd, on=['ICD9_CODE'])
    #toCsv.to_csv('counted_diagnosis.csv', index=False)
    # count relevant diagnosis labels
    #preprocessData.countRelevantDiagnoses(diagnoses, LABEL_TYPE_VT, loadingRawData.LEN_ALL_THROMB, 1)
    #preprocessData.countRelevantDiagnoses(diagnoses, LABEL_TYPE_B, loadingRawData.LEN_ALL_BLEED, 1)
    #preprocessData.countRelevantDiagnoses(loadingRawData.prescriptions, loadingRawData.RELEVANT_ANTICOAGULANTS, loadingRawData.LEN_ALL_PROPH, 0)
    # drop unrelevant columns
    preprocessData.dropColumnsToAll(admissions, chartevents, d_items, diagnoses, d_diagnoses)
    # categorial features to one hot
    admissions = preprocessData.categoricalToOneHot(admissions)
    chartevents = pd.merge(chartevents, d_items, on=['ITEMID'])
    del d_items
    chartevents = preprocessData.createCharteventsFeatures(chartevents)

    # create diagnoses binary features (except label diagnoses
    diag_pvt_tble = preprocessData.createDiagnosisFeatures(diagnoses, d_diagnoses, LABEL_TYPE_VT, LABEL_TYPE_B)
    del d_diagnoses
    preprocessData.addBinaryLabel(diagnoses, LABEL_TYPE_VT, 1)
    preprocessData.addBinaryLabel(diagnoses, LABEL_TYPE_B, 2)
    diagnoses.drop(columns=['ICD9_CODE'], inplace=True)
    # collapse, aggregate and create one HADM_ID per row
    diagnoses = diagnoses.groupby(['HADM_ID'], as_index=False).aggregate('sum')
    # int labels to binary labels
    diagnoses = preprocessData.intToBinaryFeature(diagnoses, 'vt')
    diagnoses = preprocessData.intToBinaryFeature(diagnoses, 'bleed')
    #addBinaryLabel(prescriptions, RELEVANT_ANTICOAGULANTS, 3)
    # merge diagnoses features with labels
    diagnoses = pd.merge(diag_pvt_tble, diagnoses, on='HADM_ID', how='outer')
    del diag_pvt_tble
    #merged = pd.merge(admissions, chartevents, on=['HADM_ID'])
    #dropUnneededColumns(diagnosis_icd)
    #dropUnneededColumns(prescriptions)

    admissions = pd.merge(admissions, diagnoses, on=['HADM_ID'], how='left')
    del diagnoses
    # aggregate to create every sample as admission
    adm_lebels = preprocessData.createVandBLabel(admissions)
    del admissions
    # merge with charevents features
    features_labels = pd.merge(adm_lebels, chartevents, on=['HADM_ID'], how='left')
    del adm_lebels, chartevents
    # make HADM_ID as index so it wont be feature
    features_labels.set_index('HADM_ID', inplace=True)
    print('num features with above', preprocessData.PERCENT_MISS, 'miss', len(features_labels.columns))
    features_labels = preprocessData.removeColsNans(features_labels)
    #features_labels = preprocessData.removeColsNans(features_labels)
    print('num features WITHOUT above', preprocessData.PERCENT_MISS, 'miss', len(features_labels.columns))
    features_labels = labelsToTheBack(features_labels)

    # drop 999999 admission, it ruins the normalizations
    helperFunctions.drop999999(features_labels)

    #features_labels = normalizationImputation.dropConstCol(features_labels)
    no_normalization, min_max, standartized = normalizationImputation.dropNaAndConstCols(features_labels), normalizationImputation.normalizeMinMax(features_labels), normalizationImputation.standartize(features_labels)


    processedDataToCsv(no_normalization, min_max, standartized)
    processedDataToPkl(no_normalization, min_max, standartized)

    '''
    print(merged_df['vt'].value_counts(), '\n\n',
          merged_df['bleed'].value_counts(), '\n\n',
          merged_df['v and b'].value_counts())
    print(merged_df['proph'].value_counts(), '\n\n',
          merged_df['b and p'].value_counts(), '\n\n',
          merged_df['v and p'].value_counts(), '\n\n')
    merged_df.to_csv('merged_df.csv', index=False)
    '''
    return normalizationImputation.dropNaAndConstCols(features_labels), normalizationImputation.normalizeMinMax(features_labels), normalizationImputation.standartize(features_labels)
    #diagnosis_icd.to_csv('with_labels.csv', index=False)
    #print(diagnosis_icd[diagnosis_icd['vt']]==True and diagnosis_icd[diagnosis_icd['bleed']]==True)
    #return features_labels


def processedDataToCsv(no_normalization, min_max, standartized):
    no_normalization.to_csv(PATH_NO_NORMALIZ, index=True)
    min_max.to_csv(PATH_MINMAX_NORMALIZ, index=True)
    standartized.to_csv(PATH_STANDARTIZED, index=True)


def processedDataToPkl(no_normalization, min_max, standartized):
    no_normalization.to_pickle('/data/old_data/vadim/nonormalization.pkl', compression='infer', protocol=5, storage_options=None)
    min_max.to_pickle('/data/old_data/vadim/minmax.pkl', compression='infer', protocol=5, storage_options=None)
    standartized.to_pickle('/data/old_data/vadim/standartized.pkl', compression='infer', protocol=5, storage_options=None)


def labelsToTheBack(df):
    cols = list(df.columns.values)
    cols.pop(cols.index('vt'))
    cols.pop(cols.index('bleed'))
    cols.pop(cols.index('v and b'))
    return df[cols + ['vt', 'bleed', 'v and b']]


def loadProcessedData():
    return pd.read_pickle('/data/old_data/vadim/nonormalization.pkl'), pd.read_pickle('/data/old_data/vadim/minmax.pkl'), pd.read_pickle('/data/old_data/vadim/standartized.pkl')
    #return pd.read_csv(PATH_NO_NORMALIZ, index_col='HADM_ID'), pd.read_csv(PATH_MINMAX_NORMALIZ, index_col='HADM_ID'), pd.read_csv(PATH_STANDARTIZED, index_col='HADM_ID')
