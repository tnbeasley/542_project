# Data engineering ----
def import_data():
    """
    This function imports the data and cleans up any undesired features of the table as imported
    """
    import pandas as pd
    
    df = pd.read_csv('Company_Bankruptcy_Prediction.csv')
    return df

def clean_data(df):
    """
    Purpuse: This function cleans up the data.
    
    Input:
        * df = data frame with predictor variables and variable of interest
    
    Output:
        * df = cleaned data frame with following features
        1) Rename column names to replace blank with _
        2) Remove columns with only one value
        3) Remove NA values if needed

    """
    
    # Create a dict to store original column names and new column names 
    ncol = df.shape[1]
    keys = df.columns # original column names
    values = [col.strip().replace(' ','_') for col in df.columns.tolist()] # new column names
    colname_dict = dict(zip(keys, values))
    
    # Rename column names
    df.rename(columns=colname_dict, inplace=True)
    
    # Check which columns have only one value - return indices and original column names
    col_ind = [i for i in range(0, ncol) if len(df.iloc[:,i].value_counts()) <= 1]
    col_names = [df.columns[i] for i in col_ind]
    
    # Drop the columns
    print(f'Dropping columns {col_names} that have only one value')
    df.drop(col_names, axis=1, inplace=True)
    
    # Drop NA values (if needed)
    num_na_rows = (df.apply(lambda x: sum(x.isnull().values), axis=1) > 0).sum()
    print(f'Dropping {num_na_rows} row(s) with NAs')
    df.dropna(inplace=True)
    
    return df

def prepare_model_data(df, y_col, drop_cols = [],
                       test_size = .3, scale_x = True, scale_y = False,
                       oversample = True):
    """
    Purpose: This function prepares the data to be modeled by conducting a train/test split and standaridizng the data (subtracting mean dividing by standard deviation) if specified
    
    Input:
        * df = data frame with predictor variables and variable of interest
        * y_col = column name for variable of interest
        * drop_cols = variables that do not need to be included in the predictor variables (not including the variable of interest)
        * test_size = the proportion of the data frame to isolate for testing
            - default = 30%
        * scale_x = whether to scale the X variables with a StandardScaler
        * scale_y = whether to scale the y variable with a StandardScaler
        * oversample = whether to oversample the training set
        
    Output:
        * X_train: data frame with predictor variables to train model on
        * X_test: data frame with predictor variables to test model on
        * y_train: data frame with actual values of variable of interest for train set
        * y_test: data frame with actual values of variable of interest for test set
        * scalers: dictionary with fit scalers (empty if not applicable)
    """
    
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from imblearn.over_sampling import SMOTE
    import matplotlib.pyplot as plt
    
    drop_cols.append(y_col)
    X = df.drop(drop_cols, axis='columns') 
    y = df[[y_col]]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)
    column_names = X_train.columns
    X_train, X_test, y_train, y_test = X_train.values, X_test.values, \
                                       y_train.values.ravel(), y_test.values.ravel()
    
    def plot_y(y):
        index = [0, 1]
        values = [sum(y == 0), sum(y == 1)]
        plt.bar(index, values)
        plt.ylabel('Count')
        plt.title(f'Distribution of {y_col}');
    
    if oversample:
        n = sum(y_train == 1)
        y_before = y_train.copy()
        print(f'Before oversampling, the minor class of the traing set has {n} samples.')
        oversample = SMOTE()
        X_train, y_train = oversample.fit_resample(X_train, y_train)
        n = sum(y_train == 1)
        print(f'After oversampling, the minor class of the traing set has {n} samples.')
        plt.figure(figsize=(8,4))
        plt.subplot(1,2,1)
        plot_y(y_before)
        plt.subplot(1,2,2)
        plot_y(y_train)
        plt.tight_layout()
    
    scalers = {}
    if scale_x:
        scaler_x = StandardScaler().fit(X_train)
        X_train = X_train.copy()
        X_train = scaler_x.transform(X_train)
        X_test = X_test.copy()
        X_test= scaler_x.transform(X_test)
        scalers['X'] = scaler_x
        
    if scale_y:
        scaler_y = StandardScaler().fit(y_train.reshape(-1,1))
        y_train = y_train.copy()
        y_train = scaler_y.transform(y_train.reshape(-1,1)).ravel()
        y_test = y_test.copy()
        y_test = scaler_y.transform(y_test.reshape(-1,1)).ravel()
        scalers['y'] = scaler_y
        
    return X_train, X_test, y_train, y_test, scalers, column_names