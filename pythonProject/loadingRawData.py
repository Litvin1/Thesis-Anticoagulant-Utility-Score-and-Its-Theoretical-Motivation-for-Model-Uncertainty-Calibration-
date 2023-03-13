# Vadim Litvinov
import pandas as pd


FULL_RLVNT_VT_DIAG = ['45341', '452', '45182', '45342', '45381', '45184', '45340', '4532', '45382', '45189',
                             '45386', '45385', '45384', '45351', '45383', '4536', '4539', '45119', '4530', '4533',
                             '45375', '45387', '45389', '4510', '45111', '4512', '45181', '45183', '4519', '4531',
                             '45350', '45352', '45371', '45372', '45374', '45376', '45377', '45379', '4551', '4554',
                             '9992', '41519', '99674']

PAPER_KNEE_VT_DIAG = ['4151', '41511', '41513', '41519', '4162', '4510', '4511', '45111', '45119', '4512',
                      '45181', '45189', '4519', '4531', '4532', '4534', '45341', '45342', '45350', '45351',
                      '45352', '4536', '4538', '45389', '4539']
# book list is without upper extremities vte
BOOK_GNRL_VT_DIAG = ['4151', '41511', '41513', '45111', '41519', '4512', '45181', '4519', '4534', '45340',
                     '45341', '45342', '4538', '4539']

BOOK_VT_WUPPER_WOCHRON = ['4151', '41511', '41513', '45111', '41519', '4512', '45181', '4519', '4534', '45340',
                            '45341', '45342', '4538', '4539',
                            # upper
                            '45183', '45184', '45189', '45382', '45383', '45384', '45385', '45386', '45387', '45389', '4532',
                          ]

PART_RLVNT_BLEED_DIAG = ['99811', '5789', '431', '99812', '5781', '430', '5780', '4321', '45620', '56212',
                            '7847', '53240', '85220', '56881', '920', '53551', '51909', '4590', '53541', '5693',
                            '4552', '4230', '72992', '53082', '4329', '53200', '4556']

FULL_RLVNT_BLEED_DIAG = ['99811', '5789', '431', '99812', '5781', '430', '5780', '4321', '45620', '56212',
                            '7847', '53240', '85220', '56881', '920', '53551', '51909', '4590', '53541', '5693',
                            '4552', '4230', '72992', '53082', '4329', '53200', '4556', '6268', '6271', '6266',
                            '37272', '7827', '92410', '9249', '92310', '6269', '9233', '9243', '9210', '36281',
                            '37923', '38869', '59984', '61189', '53501', '7723', '53201', '53220', '53241', '53260',
                            '53261', '85306', '77212', '60883', '92420', '9224', '6214', '86601', '37632', '86411',
                            '6021', '9582', '5967', '7848', '36372', '92300', '38869', '78630', '4555', '7703',
                            '4320', '7592', '86501', '53300', '53340', '6208', '36362', '86611', '53400', '53401',
                            '53440', '53441', '53160', '53511', '53521', '53531', '53561', '2865', '2879']
# intracarnial (inside skull)
PAPER_MJR_BLD = ['430', '431', '4320', '4321', '4329',
# upper gasrointestinal
                 '53100', '53120', '53140', '53160', '53200',
                 '53220', '53240', '53260', '53300', '53320', '53340', '53360', '53400', '53420', '53440',
                 '53460', '5780', ' 5781', '5789',
# lower gastrointestinal
                 '5693',
# other bleeding
                 '2878', '2879', '5967', '7848', '59970', '6271', '4590', '71910', '78630']

RELEVANT_ANTICOAGULANTS = ['warfarin', 'Warfarin', 'heparin', 'Heparin']

LEN_ALL_THROMB = 1
LEN_ALL_BLEED = 1
# LEN_ALL_PROPH = 4

adm_pth = '/data/old_data/vadim/ADMISSIONS.csv'
chrtevnts_pth = '/data/old_data/vadim/CHARTEVENTS.csv'
ditems_pth = '/data/old_data/vadim/D_ITEMS.csv'
diagnoses_pth = '/data/old_data/vadim/DIAGNOSES_ICD.csv'
d_diagnoses_pth = '/data/old_data/vadim/D_ICD_DIAGNOSES.csv'


def csvToPd(name):
    chunksize = 10 ** 6
    # if name == chrtevnts_pth:
    #    chunksize = 10 ** 6
    lst = []
    i = 1
    with pd.read_csv(name, chunksize=chunksize, dtype={'VALUEUOM': str,
                                                       'RESULTSTATUS': str, 'STOPPED': str}) as reader:
        for chunk in reader:
            lst.append(chunk)
            #print(i * chunksize, 'samples loaded')
            i += 1
            #print(i)
            #if name == chrtevnts_pth and i == 4:
            #    break
    return pd.concat(lst)


def loadAll():
    #fn = os.path.join()
    #dictionary_icd = csvToPd('D_ICD_DIAGNOSES.csv')
    return csvToPd(adm_pth),\
           csvToPd(chrtevnts_pth),\
           csvToPd(ditems_pth),\
           csvToPd(diagnoses_pth), \
           csvToPd(d_diagnoses_pth)
