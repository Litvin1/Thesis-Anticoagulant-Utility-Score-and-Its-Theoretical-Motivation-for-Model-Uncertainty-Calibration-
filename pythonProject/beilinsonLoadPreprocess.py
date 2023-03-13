# Vadim Litvinov
import numpy as np
import pandas as pd
import sklearn.metrics
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, auc, recall_score, average_precision_score, \
    confusion_matrix
import helperFunctions
import normalizationImputation
import preprocessData


def renameCol(vte_new):
    vte_new.rename(columns={'thrombophilia': 'yhrombophilia'}, inplace=True)


def fullPaduaVariablesLoadPreprocessCombine():
    vte_old = pd.read_pickle('/data/old_data/vadim/vte_old_with_coloer.pkl')
    #vte_new = pd.read_excel('/data/old_data/vadim/full padua variables-part1 patients dvt (version 1) (version 1).xlsx')
    #vte_new.to_pickle('/data/old_data/vadim/full padua variables-beilinson.pkl', compression='infer', protocol=5, storage_options=None)
    vte_new = pd.read_pickle('/data/old_data/vadim/full padua variables-beilinson.pkl')

    add90DaysVt(vte_old, vte_new)
    vte_new = dropUnwantedCols(vte_new)
    renameCol(vte_new)
    correctMistakes(vte_new)
    helperFunctions.createPaduaScoreByVars(vte_new)
    vte_new = featuresProcess(vte_new)
    helperFunctions.completePaduaScoreCol(vte_new)
    createPaudaPred(vte_old)
    createPaudaPred(vte_new)
    preprocessOldData(vte_old)
    vLabelProcess(vte_new)
    bLabelProcess(vte_new)
    preprocessData.createVandBLabel(vte_new)
    # change padua variables to the same name for the union
    vte_new = changeNamesPaduaVars(vte_new)
    #rename albumin col to combine them, before merging
    preprocessData.renameCols(vte_old)
    # combined by all columns. 24 identical columns in both
    combined = pd.concat([vte_old, vte_new])
    # DELETE VARS WITH OVER 33% MISS

    #checkVarsAreOnlyBinary(combined)
    # FIND THE problematic inf (BMI)
    pd.options.display.max_columns = 50
    print(combined[combined.eq(np.inf).any(1)])
    return combined


def checkVarsAreOnlyBinary(combined):
    list = ['malignant',
            'vteprior',
            'CPPS_ReducedMobility',
            'surgerylast30d',
            'CPPS_Over70',
            'CPPS_HeartRespiratoryFailure',
            'CPPS_MIorCVA',
            'CPPS_InfectionRheumatologicalDisorder',
            'CPPS_Obesity',
            'CPPS_OngoingHormonalTreatment',
            'yhrombophilia']
    for var in list:
        print(combined[var].value_counts())


def changeNamesPaduaVars(vte_new):
    vte_new.rename(columns={'function 1-ind 2- unind': 'CPPS_ReducedMobility',
                            'age>70': 'CPPS_Over70',
                            'CHF': 'CPPS_HeartRespiratoryFailure',
                            'MI': 'CPPS_MIorCVA',
                            'CTD': 'CPPS_InfectionRheumatologicalDisorder',
                            'obesity': 'CPPS_Obesity',
                            'hormonal': 'CPPS_OngoingHormonalTreatment'}, inplace=True)
    return vte_new


def combineAndImpute(vte_old, vte_new, padua_flag):
    combined = pd.concat([vte_old, vte_new])
    if padua_flag:
        combined = combined[vte_old.columns]
    else:
        combined = combined[vte_new.columns]
    # check missing values percent
    print(combined.isnull().sum()*100/len(combined))
    # remove variables with more than 33% missing
    combined = preprocessData.removeColsNans(combined)
    # impute median
    helperFunctions.imputeMedianBeilinson(combined)
    #helperFunctions.imputeMeanBeilinson(combined)
    return combined


def makeDummyCols(data):
    data['albuminFirst'] = np.nan
    data['prevHosp'] = np.nan


def genderFeature(data):
    data['SEX 1-M 2-W'] = data['SEX 1-M 2-W'] + 1


def changeLabelName(vte_old):
    vte_old.rename(columns={'BLEEDING': 'bleed'}, inplace=True)
    vte_old.rename(columns={'pe/dvt': 'vt'}, inplace=True)


def preprocessOldData(vte_old):
    dropUnwantedColsOld(vte_old)
    genderFeature(vte_old)
    changeLabelName(vte_old)
    preprocessData.createVandBLabel(vte_old)
    #change 3 to binary variable 1
    vte_old['yhrombophilia'].replace(3, 1, inplace=True)
    vte_old['function 1-ind 2- unind'].replace('Independant', 1, inplace=True)
    vte_old['function 1-ind 2- unind'].replace(['Weak', 'Dependant', 'complex dependant', 'שיקומי'], 2, inplace=True)
    vte_old['smoking'].replace('no', 0, inplace=True)
    vte_old['smoking'].replace('yes', 1, inplace=True)
    return None


def createPaudaPred(combined):
    combined.loc[combined['pauda score'] >= 4, 'padua prediction'] = 1
    combined['padua prediction'].fillna(0, inplace=True)


def checkPaduaPerformance(vte_new):
    print(vte_new['vt'].value_counts())
    vte_new = vte_new.loc[:, ['pauda score', 'vt']]
    #vte_old = vte_old.loc[:, ['pauda score', 'pe/dvt']]
    vte_new.loc[vte_new['pauda score'] >= 4, 'prediction'] = 1
    #vte_old.loc[vte_old['pauda score'] >= 4, 'prediction'] = 1
    vte_new['prediction'].fillna(0, inplace=True)
    #vte_old['prediction'].fillna(0, inplace=True)
    #vte_old['proba'] = vte_old['pauda score'] / 20
    print(confusion_matrix(vte_new['vt'], vte_new['prediction']))
    #print(confusion_matrix(vte_old['pe/dvt'], vte_old['prediction']))
    print(balanced_accuracy_score(vte_new['vt'], vte_new['prediction']))
    #print(balanced_accuracy_score(vte_old['pe/dvt'], vte_old['prediction']))
    #print(auc(vte_old['pe/dvt'], vte_old['proba']))
    print(roc_auc_score(vte_new['vt'], vte_new['pauda score']))
    #print(roc_auc_score(vte_old['pe/dvt'], vte_old['proba']))
    print(recall_score(vte_new['vt'], vte_new['prediction']))
    #print(recall_score(vte_old['pe/dvt'], vte_old['prediction']))
    print(average_precision_score(vte_new['vt'], vte_new['prediction']))
    #print(average_precision_score(vte_old['pe/dvt'], vte_old['prediction']))
    return None


def loadPreprocess(padua_related_flag):
    pd.options.mode.chained_assignment = None  # default='warn'
    #vte_new = pd.read_excel('/data/old_data/vadim/part1 patients dvt (version 1) (version 1).xlsx')
    #vte_new.to_pickle('/data/old_data/vadim/beilinson.pkl', compression='infer', protocol=5, storage_options=None)
    #vte_old = pd.read_excel('/data/old_data/vadim/vte_old_with_coloer.xlsx')
    #vte_old.to_pickle('/data/old_data/vadim/vte_old_with_coloer.pkl')
    vte_old = pd.read_pickle('/data/old_data/vadim/vte_old_with_coloer.pkl')
    vte_new = pd.read_pickle('/data/old_data/vadim/beilinson.pkl')
    # add 90 days vte
    #add90DaysVt(vte_old, vte_new)
    #rename variables for merging
    preprocessData.renameCols(vte_old)
    # make dummy Good predictor cols
    #makeDummyCols(vte_new)
    # only for padua score calculations
    if padua_related_flag:
        vte_new = vte_new[~vte_new['pauda score'].isna()]
    preprocessOldData(vte_old)
    vte_new = dropUnwantedCols(vte_new)
    correctMistakes(vte_new)
    #add90DaysVt(data)
    vLabelProcess(vte_new)
    bLabelProcess(vte_new)
    vte_new = featuresProcess(vte_new)
    # explore albumin and albuminFirst variables
    #helperFunctions.albuminFirstAlbuminAvg(vte_old, vte_new)
    #helperFunctions.albuminFirstAlbuminAvg(vte_old, vte_new)
    # explore albumin after imputation
    #helperFunctions.albuminFirstAlbuminAvg(vte_old, vte_new)
    # explore creatinine and creatFirst variables
    #helperFunctions.creatMaxCreatinineAvg(vte_old, vte_new)

    #vte_old, vte_new = keepSharedFeatures(vte_old, vte_new)
    # normalize standartize
    #data = normalizationImputation.standartize(data)
    #data = normalizationImputation.normalizeMinMax(data)
    preprocessData.createVandBLabel(vte_new)
    return vte_old, vte_new


def correctMistakes(data):
    # LABELS FIX
    # NOT FIXING LABELS
    pd.options.mode.chained_assignment = None
    data['pe/dvt'].loc[data['BLEEDING'] == 'PULMONARY EMBOLISM AND INFARCTION'] = 'PULMONARY EMBOLISM AND INFARCTION'
    data['BLEEDING'].loc[data['BLEEDING'] == 'PULMONARY EMBOLISM AND INFARCTION'] = np.NaN
    data['BLEEDING'].loc[data['pe/dvt'] == 'HEMORRHAGE OF GNULLTROINTESTINAL TRACT'] = 'HEMORRHAGE OF GNULLTROINTESTINAL TRACT'
    data['pe/dvt'].loc[data['pe/dvt'] == 'HEMORRHAGE OF GNULLTROINTESTINAL TRACT'] = np.NaN

    data['CHF'].loc[data['MI'] == 'ISCHEMIC HEART DIS. ACUTE WITHOUT M.I.'] = 'ISCHEMIC HEART DIS. ACUTE WITHOUT M.I.'
    data['MI'].loc[data['MI'] == 'ISCHEMIC HEART DIS. ACUTE WITHOUT M.I.'] = np.NaN

    data['MI'].loc[data['CHF'] == ' acute (AMI)'] = ' acute (AMI)'
    data['CHF'].loc[data['CHF'] == ' acute (AMI)'] = np.NaN

    data['COPD'].loc[data['PVD'] == 'Pulmonary congestion'] = 'Pulmonary congestion'
    data['PVD'].loc[data['PVD'] == 'Pulmonary congestion'] = np.NaN

    data['CVA'].loc[data['PVD'] == 'OCCLUSION/STENOSIS MULTI/BIL PRECEREBRAL ART. +CEREBRAL INFARCTIO'] = 'OCCLUSION/STENOSIS MULTI/BIL PRECEREBRAL ART. +CEREBRAL INFARCTIO'
    data['PVD'].loc[data['PVD'] == 'OCCLUSION/STENOSIS MULTI/BIL PRECEREBRAL ART. +CEREBRAL INFARCTIO'] = np.NaN

    data['malignant'].loc[data['aids'] == ' BCR-ABL1.positive (CML)'] = ' BCR-ABL1.positive (CML)'
    data['aids'].loc[data['aids'] == ' BCR-ABL1.positive (CML)'] = np.NaN
    data['malignant'].loc[data['aids'] == ' pancreas'] = ' pancreas'
    data['aids'].loc[data['aids'] == ' pancreas'] = np.NaN
    data['malignant'].loc[data['aids'] == 'MALIGNANT NEOPLASM OF UPPER LOBE'] = 'MALIGNANT NEOPLASM OF UPPER LOBE'
    data['aids'].loc[data['aids'] == 'MALIGNANT NEOPLASM OF UPPER LOBE'] = np.NaN
    data['malignant'].loc[data['aids'] == 'MALIGNANT NEOPLASM OF TRANSVERSE NULL'] = 'MALIGNANT NEOPLASM OF TRANSVERSE NULL'
    data['aids'].loc[data['aids'] == 'MALIGNANT NEOPLASM OF TRANSVERSE NULL'] = np.NaN
    data['malignant'].loc[data['aids'] == 'MALIGNANT NEOPLASM OF BODY OF PANCREAS'] = 'MALIGNANT NEOPLASM OF BODY OF PANCREAS'
    data['aids'].loc[data['aids'] == 'MALIGNANT NEOPLASM OF BODY OF PANCREAS'] = np.NaN
    data['malignant'].loc[data['aids'] == 'TUMOR OF URETER - MID URETER'] = 'TUMOR OF URETER - MID URETER'
    data['aids'].loc[data['aids'] == 'TUMOR OF URETER - MID URETER'] = np.NaN
    data['malignant'].loc[data['aids'] == 'LEUKEMIA OF1 CELL TYPE'] = 'LEUKEMIA OF1 CELL TYPE'
    data['aids'].loc[data['aids'] == 'LEUKEMIA OF1 CELL TYPE'] = np.NaN
    data['malignant'].loc[data['aids'] == 'MALIGNANT NEOPLASM OF HEPATIC FLEXURE'] = 'MALIGNANT NEOPLASM OF HEPATIC FLEXURE'
    data['aids'].loc[data['aids'] == 'MALIGNANT NEOPLASM OF HEPATIC FLEXURE'] = np.NaN
    data['malignant'].loc[data['aids'] == 'MALIGNANT NEOPLASM OF DESCENDING NULL'] = 'MALIGNANT NEOPLASM OF DESCENDING NULL'
    data['aids'].loc[data['aids'] == 'MALIGNANT NEOPLASM OF DESCENDING NULL'] = np.NaN
    data['malignant'].loc[data['aids'] == 'MALIGNANT NEOPLASM OF ISLETS OF LANGERHANS'] = 'MALIGNANT NEOPLASM OF ISLETS OF LANGERHANS'
    data['aids'].loc[data['aids'] == 'MALIGNANT NEOPLASM OF ISLETS OF LANGERHANS'] = np.NaN
    data['malignant'].loc[data['aids'] == 'Malignant neoplasm of lower lobe'] = 'Malignant neoplasm of lower lobe'
    data['aids'].loc[data['aids'] == 'Malignant neoplasm of lower lobe'] = np.NaN
    data['malignant'].loc[data['aids'] == 'MALIGNANT NEOPLASM OF1'] = 'MALIGNANT NEOPLASM OF1'
    data['aids'].loc[data['aids'] == 'MALIGNANT NEOPLASM OF1'] = np.NaN
    data['malignant'].loc[data['aids'] == 'MALIGNANT NEOPLASM OF URETHRA'] = 'MALIGNANT NEOPLASM OF URETHRA'
    data['aids'].loc[data['aids'] == 'MALIGNANT NEOPLASM OF URETHRA'] = np.NaN
    data['malignant'].loc[data['aids'] == 'Mixed phenotype acute leukemia (MPAL) with t(9;22)(q34.1;q11.2); BCR-ABL1'] = 'Mixed phenotype acute leukemia (MPAL) with t(9;22)(q34.1;q11.2); BCR-ABL1'
    data['aids'].loc[data['aids'] == 'Mixed phenotype acute leukemia (MPAL) with t(9;22)(q34.1;q11.2); BCR-ABL1'] = np.NaN
    data['malignant'].loc[data['aids'] == 'Chronic myelomonocytic leukemia (CMML)'] = 'Chronic myelomonocytic leukemia (CMML)'
    data['aids'].loc[data['aids'] == 'Chronic myelomonocytic leukemia (CMML)'] = np.NaN
    data['malignant'].loc[data['aids'] == 'CHOLANGIO1'] = 'CHOLANGIO1'
    data['aids'].loc[data['aids'] == 'CHOLANGIO1'] = np.NaN
    data['malignant'].loc[data['aids'] == 'Cancer'] = 'Cancer'
    data['aids'].loc[data['aids'] == 'Cancer'] = np.NaN
    data['malignant'].loc[data['aids'] == 'MALIGNANT NEOPLASM OF OTHER SPECIFIED SITES OF URINARY ORGANS'] = 'MALIGNANT NEOPLASM OF OTHER SPECIFIED SITES OF URINARY ORGANS'
    data['aids'].loc[data['aids'] == 'MALIGNANT NEOPLASM OF OTHER SPECIFIED SITES OF URINARY ORGANS'] = np.NaN
    data['malignant'].loc[data['aids'] == 'LEUKEMIA'] = 'LEUKEMIA'
    data['aids'].loc[data['aids'] == 'LEUKEMIA'] = np.NaN
    data['malignant'].loc[data['aids'] == 'UNSPECIFIED LEUKEMIA'] = 'UNSPECIFIED LEUKEMIA'
    data['aids'].loc[data['aids'] == 'UNSPECIFIED LEUKEMIA'] = np.NaN
    data['malignant'].loc[data['aids'] == ' BCR-ABL11'] = ' BCR-ABL11'
    data['aids'].loc[data['aids'] == ' BCR-ABL11'] = np.NaN
    data['malignant'].loc[data['aids'] == 'MALIGNANT NEOPLASM OF CENTRAL PORTION OF FEMALENULL'] = 'MALIGNANT NEOPLASM OF CENTRAL PORTION OF FEMALENULL'
    data['aids'].loc[data['aids'] == 'MALIGNANT NEOPLASM OF CENTRAL PORTION OF FEMALENULL'] = np.NaN
    data['malignant'].loc[data['aids'] == 'Acute undifferentiated leukemia'] = 'Acute undifferentiated leukemia'
    data['aids'].loc[data['aids'] == 'Acute undifferentiated leukemia'] = np.NaN
    data['malignant'].loc[data['aids'] == 'MALIGNANT NEOPLASM NOS'] = 'MALIGNANT NEOPLASM NOS'
    data['aids'].loc[data['aids'] == 'MALIGNANT NEOPLASM NOS'] = np.NaN
    data['malignant'].loc[data['aids'] == 'MALIGNANT NEOPLASM OF CENTRAL PORTION OF FEMALE BREAST'] = 'MALIGNANT NEOPLASM OF CENTRAL PORTION OF FEMALE BREAST'
    data['aids'].loc[data['aids'] == 'MALIGNANT NEOPLASM OF CENTRAL PORTION OF FEMALE BREAST'] = np.NaN
    data['malignant'].loc[data['aids'] == 'CHRONIC LEUKEMIA OF1 CELLNULLN RELAPSE'] = 'CHRONIC LEUKEMIA OF1 CELLNULLN RELAPSE'
    data['aids'].loc[data['aids'] == 'CHRONIC LEUKEMIA OF1 CELLNULLN RELAPSE'] = np.NaN
    data['malignant'].loc[data['aids'] == 'LEUKEMIA  - ACUTE  MYELOGENOUS (AML)'] = 'LEUKEMIA  - ACUTE  MYELOGENOUS (AML)'
    data['aids'].loc[data['aids'] == 'LEUKEMIA  - ACUTE  MYELOGENOUS (AML)'] = np.NaN
    data['malignant'].loc[data['aids'] == 'MALIGNANT NEOPLASM OF AXILLARY TAIL OF FEMALE BREAST'] = 'MALIGNANT NEOPLASM OF AXILLARY TAIL OF FEMALE BREAST'
    data['aids'].loc[data['aids'] == 'MALIGNANT NEOPLASM OF AXILLARY TAIL OF FEMALE BREAST'] = np.NaN
    data['malignant'].loc[data['aids'] == 'Leukemia'] = 'Leukemia'
    data['aids'].loc[data['aids'] == 'Leukemia'] = np.NaN
    data['malignant'].loc[data['aids'] == 'Acute undifferentiated leukemia 9801/3'] = 'Acute undifferentiated leukemia 9801/3'
    data['aids'].loc[data['aids'] == 'Acute undifferentiated leukemia 9801/3'] = np.NaN
    data['malignant'].loc[data['aids'] == 'LEUKEMIA - CHRONIC MYELOID (CML)'] = 'LEUKEMIA - CHRONIC MYELOID (CML)'
    data['aids'].loc[data['aids'] == 'LEUKEMIA - CHRONIC MYELOID (CML)'] = np.NaN
    data['malignant'].loc[data['aids'] == ' MALIGNANT'] = ' MALIGNANT'
    data['aids'].loc[data['aids'] == ' MALIGNANT'] = np.NaN
    data['malignant'].loc[data['aids'] == 'AML with inv(16)(p13.1q22) or t(16;16)(p13.1;q22);..CBFB-MYH11'] = 'AML with inv(16)(p13.1q22) or t(16;16)(p13.1;q22);..CBFB-MYH11'
    data['aids'].loc[data['aids'] == 'AML with inv(16)(p13.1q22) or t(16;16)(p13.1;q22);..CBFB-MYH11'] = np.NaN
    data['malignant'].loc[data['aids'] == 'MALIGNANT POORLY DIFFERENTIATED 1'] = 'MALIGNANT POORLY DIFFERENTIATED 1'
    data['aids'].loc[data['aids'] == 'MALIGNANT POORLY DIFFERENTIATED 1'] = np.NaN
    data['malignant'].loc[data['aids'] == 'Rectal cancer'] = 'Rectal cancer'
    data['aids'].loc[data['aids'] == 'Rectal cancer'] = np.NaN
    data['malignant'].loc[data['aids'] == 'MALIGNANT NEOPLASM OF ANAL CANAL'] = 'MALIGNANT NEOPLASM OF ANAL CANAL'
    data['aids'].loc[data['aids'] == 'MALIGNANT NEOPLASM OF ANAL CANAL'] = np.NaN
    data['malignant'].loc[data['aids'] == '1 (large intestine)'] = '1 (large intestine)'
    data['aids'].loc[data['aids'] == '1 (large intestine)'] = np.NaN


def dropUnwantedColsOld(data):
    data.drop(columns=['daysPE', 'PE day 8-90', 'aspirinHosp', 'aspirinAdm', 'VTEDischarge', 'VTE_discharge', 'VTE_adm',
                       'prophylaxis=1', 'daysBleed', 'bleeding diagnosis day 8-90', 'DVT8_90', 'daysDVT', 'DVT day 8-90',
                       'PE8_09', 'HB8_90', 'HB10_19', 'HB25_35', 'HB85_90', 'death', 'PLT8_90',
                       'daysDeath', '30 day mortality', 'Final PE', 'Final DVT',
                       'PRBC', 'urinaryCath',
                       #'pauda score',
                       'ETTube', 'bleedingDx', 'Charlson Metastatic solid tumor 3',
                        'HBDelta', 'Charlson total', 'Charlson Metastatic solid tumor', 'Charlson lymphoma or leukemia',
                       'FFP_Y/N',
                       'days admission', 'Charlson Leukemia', 'Charlson lymphoma',
                       'feverFirst',
                       'ICU_Y/N'
                       ], inplace=True)


def dropUnwantedCols(data):
    # most of them are anticoagulants
    data = data.iloc[:, :63]
    data.drop(columns=['AD_DATE', 'plavix', 'aspirin', 'brilinta', 'ptasugrel', 'clexan yes or no', 'DAYS OF CLEXAN', 'CTA',
                       'us', 'echo', 'perfusion', 'numbrer of admission 90 dayes', 'date', 'bleeding/thrombosis.1',
                       'date.1', 'bleeding/thrombosis',
                       # drop saturation beacuse its not in teh old set and we cant imputate
                       #'satuation',
                       # 'pauda score',
                       'd dimer',
                       'days admission',
                       '>38fever', 'bp sys av',
                       'pulse', 'clean yes or no'
                        #, 'AGE'
                       ], errors='ignore', inplace=True)
    return data


def featuresProcess(data):
    data.replace('.', np.nan, inplace=True)
    # fill 0 instead of nan only in the binary variables. numeric should imputate
    data.fillna({x:0 for x in ['MI', 'CHF', 'PVD', 'CVA', 'DEMTIA', 'COPD', 'CTD', 'PUD', 'LIVER D', 'DM2',
          'SEVERE', 'malignant', 'aids', 'surgerylast30d', 'vteprior', 'yhrombophilia',
          # smoking missing values will be 0 beacuse its the mode of the variable
          'smoking']}, inplace=True)
    pd.set_option('display.max_rows', 5000)
    #for column in data.columns:
    #    print(data[column].value_counts(), '\n')
    data['MI'].replace(['AC. M.I. INFEROLATERAL', '1 OF INFEROLATERAL WALL', 'S/P MI - 1',
                        'STEMI - ST ELEVATION 1', '1 - ACUTE', '1 Non ST elevation',
                        'AC. M.I. OTHE LATERAL', '1 - ANTERIOR WALL', 'AC. M.I. ANTEROLATERAL',
                        'MI - 1', 'ACUTE NON ST ELEVATION 1 (NSTE',
                        'POSTERIOR WALL MYOCADIAL INFARCTION (MI)', 'ANEURYSM OF HEART (WALL)',
                        '1 (AMI)', '1 OF UNSP. TYPE OF BYPASS GRAFT', 'OTHER REMOVAL OF CORONARY ARTERY OBSTRUCTION',
                        'OTHER', ' acute (AMI)'], 1, inplace=True)
    data['MI'].replace(['OTHER STAPHYLOCOCCUS INFECTION', 'STREPTOCOCCUS INFEC.', 'Haploidentical transplantation'
                        ], 0, inplace=True)
    data['CHF'].replace(['ACUTE 1', 'ACUTE ON CHRONIC COMBINED SYSTOLIC AND 1',
                         ' CORONARY INSUFFICIENCY(ACUTE)', 'ISCHEMIC HEART DIS. ACUTE WITHOUT M.I.'], 1, inplace=True)
    data['CHF'].replace([' EPISODE OF CARE UNSP.', ' GROUP G', ' INITIAL EPISODE OF CARE'], 0, inplace=True)
    data['PVD'].replace(['DIASTOLIC 1', 's/p 1', 'OTHER AND1 1',
                         '1 due to 1', '1                =        1', 'OCCLUSION/STENOSIS MULTI/BIL PRECEREBRAL ART. +CEREBRAL INFARCTIO',
                         'HEMIPLEGIA AFFECTINGNULL SIDE', 'SPONT. CEREBELLAR HEMORRHAGE', 'SPEECH AND LANGUAGE DEFICIT'], 1, inplace=True)
    data['PVD'].replace(['ANTERIOR NULL'], 0, inplace=True)

    data['CVA'].replace(['Demen1'], 0, inplace=True)
    data['CVA'][data['CVA'] != 0] = 1
    data['DEMTIA'].replace(['OTHER SPECIFIED ALLERGIC ALVEOLITIS AND PNEUMONITIS',
                            'Other specified allergic alveolitis and pneumonitis', ' NULL',
                            'Biopsy of anus', 'RESPIRATORY CONDITIONS DUE TO1 EXTERNAL AGENTS',
                            'nbull', 'PNEUMOCONYOSIS'], 0, inplace=True)
    data['DEMTIA'][data['DEMTIA'] != 0] = 1
    data['COPD'].replace([' NULL', 'nbull'], 0, inplace=True)
    data['COPD'][data['COPD'] != 0] = 1
    data['CTD'].replace([' NULL'], 0, inplace=True)
    data['CTD'][data['CTD'] != 0] = 1
    data['PUD'].replace([' NULL'], 0, inplace=True)
    data['PUD'][data['PUD'] != 0] = 1
    #data['LIVER D'].replace([''], 0, inplace=True)
    data['LIVER D'][data['LIVER D'] != 0] = 1
    data['DM2'].replace(['NULL -NULL', 'UNSPECIFIED HYPERTENSIVE RENAL DISEASE',
                         'MALIGNANT HYPERTENSIVE KIDNEY DISEASE', 'POSTSURGICAL RENAL DIALYSIS STATUS',
                         'Malignant hypertensive nephropathy / accelerated hypertension nephropathy - histologically proven',
                         'MALIGNANT HYPERTENSIVE KIDNEY DIS. WITH 1 STAGE I THROUGH STAGE IV'], 0, inplace=True)
    data['DM2'][data['DM2'] != 0] = 1
    #data['SEVERE'].replace([''], 0, inplace=True)
    data['SEVERE'][data['SEVERE'] != 0] = 1
    data['malignant'].replace(['NULL -NULL', 'BENIGN HYPERTENSIVE RENAL DISEASE',
                               'BENIGN HYPERTENSIVE HEART AND KIDNEY DIS.WITHOUT HEART FAILURE AILURE OR RENAL FAILURE',
                               'UNSPECIFIED HYPERTENSIVE KIDNEY DISEASE', 'UNSP. HYPERTENSIVE RENAL DIS.+ RENAL FAILURE',
                               'BENIGN HYPERTENSIVE KIDNEY DIS. WITH 1NULL OR 1'], 0, inplace=True)
    data['malignant'][data['malignant'] != 0] = 1
    data['aids'].replace([' NULL',
                          'CHRONIC RENAL FAILURE',
                          'CHRONIC RENAL FAILURE',
                          'CHRONIC KIDENY DISEASE',
                          '1 (large intestine)', '1 OF LIVER', 'LIVER 1 PRIMARY'], 0, inplace=True)
    data['aids'][data['aids'] != 0] = 1
    # 'pulse' dropped down
    #data['pulse'].replace('COUMADIN', 0, inplace=True)
    data['satuation'].replace(['NULLNULL', '8NULL', 'NULL0'], 0, inplace=True)
    data['function 1-ind 2- unind'].replace([' NULL', 'EXITUS'], 0, inplace=True)
    #data['>38fever'].replace([' NULL', 'EXITUS'], 0, inplace=True)
    data['surgerylast30d'].replace([' NULL', 'EXITUS', 'nULL'], 0, inplace=True)
    data['surgerylast30d'][data['surgerylast30d'] != 1] = 0
    #VTE PRIOR
    data['vteprior'].replace([' NULL', 'EXITUS', 'ORBITAL HEMORRHAGE', 'DIVERTICULOSIS OF COLON', 'GROSS HEMATURIA'], 0, inplace=True)
    data['yhrombophilia'].replace([' NULL', 'EXITUS', 'DVT             =       Deep vein thrombophlebitis',
                                   'nULL', 'N', 'HEMORRHAGE'], 0, inplace=True)
    data['smoking'].replace(['ELIQUIS', 'N', 2], 0, inplace=True)
    data['smoking'].replace('no', 0, inplace=True)
    data['smoking'].replace('yes', 1, inplace=True)
    #data['days admission'].replace(['no', 'CLEXANE PREFFIL.', 'yes', ' '], 0, inplace=True)
    pd.set_option('display.max_rows', 5000)
    #for column in data.columns:
    #    print(data[column].value_counts(), '\n')
    #print(data.index[data['ferritin'] == '.'].tolist())
    return data


def bLabelProcess(data):
    data['BLEEDING'] = data['BLEEDING'].fillna(0)
    pd.options.display.max_rows = 1000
    #print(data['BLEEDING'].value_counts())
    data['BLEEDING'].replace(['EXITUS', ' NULL', 'NULL           =     NULL obstructiveNULL diseNULLe',
                              'NULL                   =   NULL', 'NULL                   =    NULL', 'N',
                              'NUL', 'NULL SELECTIVE IMMUNOGLOBULIN DEFICIENCIES', 'NIULL', 'NILL',
                              'NULL '], 0, inplace=True)
    #print(data['BLEEDING'].value_counts())
    data['BLEEDING'][data['BLEEDING'] != 0] = 1
    #print(data['BLEEDING'].value_counts())
    data.rename(columns={'BLEEDING': 'bleed'}, inplace=True)


def vLabelProcess(data):
    data['pe/dvt'] = data['pe/dvt'].fillna(0)
    #print(data['pe/dvt'].value_counts())
    data['pe/dvt'].replace([' NULL', 'EXITUS', 'NUL', 'N',
                            'NULL               =        Space occupying lesion',
                            'PERCUTANEOUS ENDOSCOPIC GNULLTROSTOMY (PEG)',
                            'NULL                 =       Percutaneous coronary intervention',
                            'POST HEMORRHAGIC HYDROCEPHALUS', 'HEMORRHAGE OF GNULLTROINTESTINAL TRACT',
                            'NULL           =     NULL obstructiveNULL diseNULLe',
                            'PERCUTANEOUS TRANSLUMINAL CORONARY ANGIOPLNULLTY STATUS'], 0, inplace=True)
    #print(data['pe/dvt'].value_counts())
    data['pe/dvt'][data['pe/dvt'] != 0] = 1
    #print(data['pe/dvt'].value_counts())
    data.rename(columns={'pe/dvt': 'vt'}, inplace=True)


def add90DaysVt(vte_old, vte_new):
    vte_old['pe/dvt'].loc[vte_old['PE8_09'] == 1] = 1
    vte_old['pe/dvt'].loc[vte_old['DVT8_90'] == 1] = 1

    vte_new['pe/dvt'].loc[vte_new['bleeding/thrombosis'] == 'PULMONARY EMBOLISM AND INFARCTION'] = 1
    vte_new.drop(columns='bleeding/thrombosis', inplace=True)
