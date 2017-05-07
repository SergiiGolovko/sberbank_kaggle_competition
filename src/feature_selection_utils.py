import pandas as pd
THRESHOLD = 0.90


def remove_highly_correlated_features(X_train, X_test, threshold=THRESHOLD):

    df = pd.concat([X_train, X_test], ignore_index=True)

    print('Number of features prior to removing highly correlated features: %d'
          % df.shape[1])

    df_corr = df.corr()
    indexes, columns, values = df_corr.index, df_corr.columns, df_corr.values
    drop_columns = []

    for i in range(values.shape[0]):
        for j in range(i+1, values.shape[1]):
            if values[i, j] > THRESHOLD:
                print(indexes[i], columns[j], values[i, j])
                drop_columns.append(columns[j])

    drop_columns = list(set(drop_columns))
    # columns = set(df.columns).difference(drop_columns)

    X_train.drop(drop_columns, axis=1, inplace=True, errors='ignore')
    X_test.drop(drop_columns, axis=1, inplace=True, errors='ignore')

    print('Number of features after removing highly correlated features: %d'
          % X_train.shape[1])

    # return x_train[list(columns)], x_test[list(columns)]
    return X_train, X_test
