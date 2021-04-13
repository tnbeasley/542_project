# Time code function
def time_it(func):
    import time
    import numpy as np
    def wrapper(*args, **kwargs):
        start = time.time()
        results = func(*args, **kwargs)
        time_elapsed = time.time()-start
        print(f'Time: {np.round(time_elapsed, 2)} seconds')
        return time_elapsed, results
    return wrapper

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


# KMeans ----
@time_it
def kmeans_clustering(X, y):
    """
    Conduct kmeans clustering to categorize companies
    
    Minoo/Amelia
    """
    from sklearn.cluster import KMeans
    
    kmeans = KMeans(n_clusters=2, random_state=0)
    kmeans.fit(X)
    print("KMeans Labels: " , kmeans.labels_)
    print("KMeans cluster centers: ", kmeans.cluster_centers_)

    result = kmeans.predict(y)
    print("KMeans results: ", result)
    
    return result


# Create models ----
@time_it
def xgb(X_train, X_test, y_train, y_test):
    """
    Purpose: This function creates an XGBoost classifier.
    
    Input:
        * X_train
        * X_test
        * y_train
        * y_test
        * scalers: dictionary with fit scalers
    
    Output:
        * clf: a trained XGBoost classifier
        * results: a dict holding train_time, pred_time, acc_train, acc_test, f1_train, and f1_test
    """
    from time import time
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score, f1_score 

    # Instantiate a classifier
    clf = XGBClassifier(learning_rate=0.05, n_estimator=300, max_depth=5)
    
    # Fit the model to the training data
    start1 = time() # get start time
    clf.fit(X_train, y_train)
    end1 = time() # get end time
    
    # Make predictions for the train data and the test data
    y_pred_train = clf.predict(X_train)
    start2 = time() # get start time
    y_pred_test = clf.predict(X_test)
    end2 = time() # get end time
    
    # Evaluate model performance: training time, accuracy, F1 score
    results = {}
    results['train_time'] = end1 - start1 # training time in second
    results['pred_time'] = end2 - start2 # prediction time in second
    results['acc_train'] = accuracy_score(y_train, y_pred_train) # accuracy for train set
    results['acc_test'] = accuracy_score(y_test, y_pred_test) # accuracy for test set
    results['f1_train'] = f1_score(y_train, y_pred_train) # f1 score for train set
    results['f1_test'] = f1_score(y_test, y_pred_test) # f1 score for test set
    
    return clf


@time_it
def logistic_regression(X_train, y_train):
    """
    Amelia
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn import metrics
    import seaborn as sns


    #training the model
    model = LogisticRegression(solver='liblinear', random_state=0)
    model.fit(X_train,y_train)
    #predictions for dataset
    predictions = model.predict(X_test)

    #score method to check accuracy
    score = model.score(X_test, y_test)
    print(score)

    #confusion matrix
    cm = metrics.confusion_matrix(y_test, predictions)
    print(cm)
    #seaborn confusion matrix plot
    plt.figure(figsize=(9,9))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
    plt.ylabel('Actual label');
    plt.xlabel('Predicted label');
    all_sample_title = 'Accuracy Score: {0}'.format(score)
    plt.title(all_sample_title, size = 15);
    return model


@time_it
def nearest_neighbors(X_train, y_train, X_test, y_test,min_n,max_n):
    """
    Leslie
    """
    from sklearn.neighbors import KNeighborsClassifier
    # baseline for for loop
    best_n = 0
    best_training = 0
    best_test = 0
    
    # loop through all possible nearest neighbors from min_n to max_n
    for i in range(min_n,max_n): 
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, y_train)
            
        training = knn.score(X_train, y_train)
        test = knn.score(X_test, y_test)
            
        if test > best_test:
            best_n = i
            best_training = training
            best_test = test
            
    results = {}
    results['Best N'] = best_n # best number of neighbors
    results['Best Training'] = best_training # best training set score
    results['Best Test'] = best_test # best test set score
    
    clf = KNeighborsClassifier(n_neighbors=best_n)
    clf.fit(X_train, y_train)
    
    return clf, results


@time_it
def random_forest(X_train, y_train):
    """
    Minoo
    """
    
    return clf


@time_it
def neural_network(X_train, y_train, X_test, y_test,
                   hidden_layer_size = 32, dropout_size = .1, 
                   patience = 10, batch_size = 2):
    """
    Tanner
    """
    
    import pandas
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.metrics import AUC
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import StratifiedKFold
    
    
    # Create model
    model = Sequential()

    # Add Layers
    model.add(Dense(hidden_layer_size, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dropout(dropout_size))
    model.add(Dense(hidden_layer_size, activation = 'relu'))
    model.add(Dropout(dropout_size))
    model.add(Dense(hidden_layer_size, activation = 'relu'))
    model.add(Dropout(dropout_size))
    model.add(Dense(hidden_layer_size, activation = 'relu'))
    model.add(Dropout(dropout_size))
    model.add(Dense(hidden_layer_size, activation = 'relu'))
    model.add(Dropout(dropout_size))
    model.add(Dense(hidden_layer_size, activation = 'relu'))
    model.add(Dropout(dropout_size))
    model.add(Dense(1, activation='sigmoid'))

    # Callbacks
    es = EarlyStopping(monitor = 'val_loss',
                       patience = patience, 
                       verbose = 1)
    callbacks = [es]

    # Compile model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy'
#         metrics=['accuracy']
    )

    # Fit Model
    model.fit(
        x = X_train, y = y_train,
        batch_size = patience,
        epochs = 5000,
        verbose = 2,
        callbacks = callbacks,
        validation_data = (X_test, y_test)
    )
    
    return model


# Model evaluation ----
def model_statistics(clfs, train_times, X_train, X_test, y_train, y_test):
    """
    Inputs:
    * clfs = list of classifiers to compare
    * X_train = predictor variables for train data set
    * X_test = predictor variables for test data set
    * y_train = variable of interest for train data set
    * y_test = variable of interest for test data set
    
    Output:
    * Data frame containing the following statistics for the classifiers:
        - Training time
        - Accuracy
        - Precision
        - Recall
        - F1 Score
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from scipy import stats
    
    train_preds = [np.round(mod.predict(X_train)) for mod in clfs]
    test_preds =  [np.round(mod.predict(X_test))  for mod in clfs]
    mod_stats = pd.DataFrame({
        'Model':         [str(mod) for mod in clfs],
        'SecondsToTrain':[time for time in train_times],
        'TrainPrecision':[precision_score(y_true = y_train, y_pred = train_preds[i]) for i in range(len(clfs))],
        'TestPrecision': [precision_score(y_true = y_test,  y_pred = test_preds[i]) for i in range(len(clfs))],
        'TrainRecall':   [recall_score(y_true = y_train,    y_pred = train_preds[i]) for i in range(len(clfs))],
        'TestRecall':    [recall_score(y_true = y_test,     y_pred = test_preds[i]) for i in range(len(clfs))],
        'TrainF1':       [f1_score(y_true = y_train,        y_pred = train_preds[i]) for i in range(len(clfs))],
        'TestF1':        [f1_score(y_true = y_test,         y_pred = test_preds[i]) for i in range(len(clfs))],
        'TrainAccuracy': [accuracy_score(y_true = y_train,  y_pred = train_preds[i]) for i in range(len(clfs))],
        'TestAccuracy':  [accuracy_score(y_true = y_test,   y_pred = test_preds[i]) for i in range(len(clfs))]
    })
    
    naive_train_preds = np.repeat(stats.mode(y_train)[0][0], len(y_train))
    naive_test_preds =  np.repeat(stats.mode(y_train)[0][0], len(y_test))
    naive_stats = pd.DataFrame({
        'Model':         ['Naive'],
        'SecondsToTrain':[0],
        'TrainPrecision':[precision_score(y_true = y_train, y_pred = naive_train_preds, zero_division = 0)],
        'TestPrecision': [precision_score(y_true = y_test,  y_pred = naive_test_preds, zero_division = 0)],
        'TrainRecall':   [recall_score(y_true = y_train,    y_pred = naive_train_preds, zero_division = 0)],
        'TestRecall':    [recall_score(y_true = y_test,     y_pred = naive_test_preds, zero_division = 0)],
        'TrainF1':       [f1_score(y_true = y_train,        y_pred = naive_train_preds, zero_division = 0)],
        'TestF1':        [f1_score(y_true = y_test,         y_pred = naive_test_preds, zero_division = 0)],
        'TrainAccuracy': [accuracy_score(y_true = y_train,  y_pred = naive_train_preds)],
        'TestAccuracy':  [accuracy_score(y_true = y_test,   y_pred = naive_test_preds)]
    })
    
    mod_stats = mod_stats.append(naive_stats)
    
    return mod_stats


# Run portion ----
if __name__ == '__main__':
    import pandas as pd
    import numpy as np

    df = import_data()
    clean_df = clean_data(df)

    X_train, X_test, y_train, y_test, scalers = prepare_model_data(
        df = clean_df, y_col = 'Bankrupt?'
    )

    
    # Example_1 for the kmeans_clustering function using numpy arrays as inputs:
    X = np.array([[1, 3], [0, 4], [0, 3],
              [11, 0], [12, 7], [13, 0]])
    y = np.array([[1, 10], [17, 2]])
    result = kmeans_clustering(X, y)
    
    # Example_2 for the kmeans_clustering function using pandas dataframe as inputs:
    data1 = [1,2,3,4,5,6,7,8,9,10]
    data2 = [6,7,1]
    X = pd.DataFrame(data1)
    y = pd.DataFrame(data2)
    result = kmeans_clustering(X, y)
    
    
    # Models 
    # xgboost model
    xgb_time, xgb_clf = xgb(
        X_train, X_test, 
        y_train, y_test
    )
    
    # Nearest Neighbors - not sure if I did this correctly
    knn_time, knn_clf = nearest_neighbors(
        X_train, X_test, 
        y_train, y_test,
        1, 5
    )
    
    # neural network model
    nn_time, nn_clf = neural_network(
        X_train, y_train, 
        X_test, y_test,
        hidden_layer_size = 64,
        dropout_size = .25,
        patience = 50, 
        batch_size = 1
    )
    
    
    # Model Statistics
    model_stats = model_statistics(
        clfs = [xgb_clf, knn_clf, nn_clf], 
        train_times = [xgb_time, knn_time, nn_time],
        X_train = X_train, 
        X_test = X_test, 
        y_train = y_train, 
        y_test = y_test
    )
    
    model_stats
