# Vadim Litvinov
from sklearn.preprocessing import MinMaxScaler


def minMaxScalar(data):
    scaler = MinMaxScaler()
    scaler.fit(data)
    data[data.columns] = scaler.fit_transform(data[data.columns])

    return data


def oldScalar(df):
    # drop const col (0 var)
    df = dropNaAndConstCols(df)
    if len(df) < 60000:
        features = df.drop(['vt', 'bleed', 'v and b'], axis=1)
    else:
        features = df.drop(['vt'], axis=1)
    min = features.min()
    max = features.max()
    denom = max - min
    #if denom != 0:
    normalized_df = (features - min) / denom
    del features, min, max, denom
    # drop columns which max-min==0
    #dropNaAndConstCols(normalized_df)
    normalized_df['vt'] = df['vt']
    if len(df) < 60000:
        normalized_df['bleed'] = df['bleed']
        normalized_df['v and b'] = df['v and b']
    del df
    return normalized_df


def standartize(df):
    df = dropNaAndConstCols(df)
    if len(df) < 60000:
        features = df.drop(['vt', 'bleed', 'v and b'], axis=1)
    else:
        features = df.drop(['vt'], axis=1)
    normalized_df = (features - features.mean()) / features.std()
    del features
    # drop columns which had std == 0
    dropNaAndConstCols(normalized_df)
    normalized_df['vt'] = df['vt']
    if len(df) < 60000:
        normalized_df['bleed'] = df['bleed']
        normalized_df['v and b'] = df['v and b']
    del df
    return normalized_df


def dropNaAndConstCols(df):
    #df.dropna(axis=1, how='all', inplace=True)
    return df.loc[:, df.std() > 0.0]
