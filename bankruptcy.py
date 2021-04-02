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
    This function cleans up the data
    
    Yang/Leslie
    """
    
    # Fix column names
    
    # Drop columns with few/one value
    
    # Drop NA values (if needed)
    num_na_rows = (df.apply(lambda x: sum(x.isnull().values), axis = 1) > 0).sum()
    print(f'Dropping {num_na_rows} row(s) with NAs')
    df1 = df.dropna()
    
    return clean_df


# df = import_data()
# y_col = 'Bankrupt?'

def prepare_model_data(df, y_col, drop_cols = [],
                       test_size = .3, scale_x = True, scale_y = False):
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
        
    Output:
        * X_train: data frame with predictor variables to train model on
        * X_test: data frame with predictor variables to test model on
        * y_train: data frame with actual values of variable of interest for train set
        * y_test: data frame with actual values of variable of interest for test set
        * scalers: dictionary with fit scalers (empty if not applicable)
    """
    
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    drop_cols.append(y_col)
    X = df.drop(drop_cols, axis='columns') 
    y = df[[y_col]]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)
    X_train, X_test, y_train, y_test = X_train.values, X_test.values, \
                                       y_train.values.ravel(), y_test.values.ravel()
    
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
        
    return X_train, X_test, y_train, y_test, scalers


def kmeans_clustering(X, y):
    """
    Conduct kmeans clustering to categorize companies
    
    Minoo/Amelia
    """
    
    return kmeans_df



# Create models ----


if __name__ == 'main':
    import pandas as pd
    import numpy as np

    df = import_data()
    clean_df = clean_data(df)
    X_train, X_test, y_train, y_test, scalers = prepare_model_data(df = clean_df, y_col = 'Bankrupt?')
