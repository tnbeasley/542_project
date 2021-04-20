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
    X_train, X_test, y_train, y_test = X_train.values, X_test.values, \
                                       y_train.values.ravel(), y_test.values.ravel()
    
    def plot_y(y):
        index = [0, 1]
        values = [sum(y == 0), sum(y == 1)]
        plt.bar(index, values)
        plt.xticks(index, index)
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
def xgb(X_train, y_train,
        learning_rate=0.5, n_estimator=10, 
        max_depth=5, random_state=0):
    """
    Yang
    Purpose: This function creates an XGBoost classifier.
    
    Input:
        * X_train
        * y_train
    
    Output:
        * clf: a fitted XGBoost classifier
    """
    from xgboost import XGBClassifier
    from xgboost import plot_importance
    import matplotlib.pyplot as plt

    # Instantiate a classifier
    clf = XGBClassifier(learning_rate=learning_rate, 
                        n_estimator=n_estimator, 
                        max_depth=max_depth,
                        random_state=random_state)
    
    # Fit the model to the training data
    clf.fit(X_train, y_train)
    
    # Plot feature importances
    plt.rcParams["figure.figsize"] = (10, 5)
    plot_importance(clf, max_num_features=10)
    
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
def nearest_neighbors(X_train, y_train, X_test, y_test, min_n, max_n):
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
def random_forest(X_train, y_train, n_estimators = 5,
                  criterion = 'entropy', random_state = 0):
    """
    Minoo
    """
    from sklearn.ensemble import RandomForestClassifier
    
    # Fitting Random Forest to the Training set:
    clf = RandomForestClassifier(
        n_estimators = n_estimators, 
        criterion = criterion, 
        random_state = random_state)
    clf.fit(X_train, y_train)
    

    return clf


@time_it
def neural_network(X_train, y_train, X_test, y_test,
                   num_layers = 2,
                   hidden_layer_size = 32, dropout_size = .25, 
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
    for i in np.arange(num_layers-1)+1:
        model.add(Dense(hidden_layer_size, activation = 'relu'))
    model.add(Dropout(dropout_size))
    model.add(Dense(1, activation='sigmoid'))
    
    model.summary()

    # Callbacks
    es = EarlyStopping(monitor = 'val_binary_accuracy',
                       mode = 'max',
                       patience = patience, 
                       verbose = 1)
    callbacks = [es]

    # Compile model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['binary_accuracy']
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

# Ensemble prediction ----
def ensemble_prediction(clfs, model_stats, X):
    clf_votes = pd.DataFrame({'Model':model_stats.Model,
                              'NumVotes':np.round(model_stats.TestF1*100)})
    
    pred_votes = pd.DataFrame({
        'Row':np.arange(X.shape[0]),
        'Votes_0':np.repeat(0, X.shape[0]),
        'Votes_1':np.repeat(0, X.shape[0])
    })
    for clf_num in range(len(clfs)):
        clf = clfs[clf_num]
        preds = clf.predict(X)
        for row in range(len(preds)):
            if preds[row] == 0:
                pred_votes.iloc[row,1] = pred_votes.iloc[row,1] + clf_votes.iloc[clf_num,1]
            if preds[row] == 1:
                pred_votes.iloc[row,2] = pred_votes.iloc[row,2] + clf_votes.iloc[clf_num,1]
    
    pred_votes['FinalPred'] = pred_votes['Votes_1'] > pred_votes['Votes_0']
    pred_votes['FinalPred'] = pred_votes['FinalPred'].map({False:0, True:1})
    
    return pred_votes, pred_votes.FinalPred.values
    

# Partial dependence ----
def partial_dependence(clf, df_X, column, scalers, num_test = 30, show_plot = True):
    """
    Purpose:
        Partial dependence analysis is a way to evaluate how the level of variables impact predictions from a predictive model. To do this, it takes a data frame column and substitutes the column's values with one value. It then makes predictions on this data frame with this substituted column and averages the predictions. Then, a new value is tested in column and the process is repeated to see how changing the value in the column impact predictions.
        
        This function takes the range of values in the data frame and creates a sequence across this range for the length specified in num_test. Then, all these values are tested for the input model and saved into a data frame.
        
    Inputs:
    * clf = predictive model object
    * df_X = data frame with columns for model (don't include y column)
    * column = column to calculate partial dependence values for
    * scalers = scalers dictionary needed to transform original form of data frame
    * num_test = how many values to test for this column (from min to max of actual values)
    * show_plot = whether to show a partial dependence plot
    
    Output:
    * avg_preds = data frame with the average prediction for a given value in the column
    """
    import matplotlib.pyplot as plt
    
    unique_val = df[column].unique()
    
    if len(unique_val) > num_test:
        min_val = unique_val.min()
        max_val = unique_val.max()
        step = (max_val - min_val)/num_test
        val_range = np.arange(min_val, max_val, step)
    else:
        val_range = np.sort(unique_val)
        
    df_copy = df_X.copy()
    avg_preds = {'Value':[], 'AvgPred':[]}
    for val in val_range:
        df_copy[column] = val
        df_copy_scl = scalers['X'].transform(df_copy)
        preds = clf.predict(df_copy_scl)
        avg_pred = preds.mean()
        avg_preds['Value'].append(val)
        avg_preds['AvgPred'].append(avg_pred)
        
    avg_preds = pd.DataFrame(avg_preds)
    avg_preds['column'] = column
    
    if show_plot:
        plt.plot(avg_preds.Value, avg_preds.AvgPred)
        plt.xlabel(column)
        plt.title(f'Partial Dependence Plot for {column}')
        plt.show()
    
    return avg_preds

def partial_dependence_loop(clf, df_X, scalers, num_test, show_plot):
    """
    Purpose:
        This function loops through all the columns in df_X, performing partial dependence analysis on each.
        
    Inputs:
    * clf = predictive model object
    * df_X = data frame with columns for model (don't include y column)
    * scalers = scalers dictionary needed to transform original form of data frame
    * num_test = how many values to test for this column (from min to max of actual values)
    * show_plot = whether to show a partial dependence plot
    
    Output:
    * partial_dependences_df = data frame with test values and average prediction for each column
    * partial_dependences_stats = min, max, and range of average predictions for each column
    """
    partial_dependences = []
    counter = 1
    for col in df_X.columns:
        part_dep = partial_dependence(clf = nn_clf, df_X = df_X, column = col,
                                      scalers = scalers, num_test = num_test, show_plot = show_plot)
        partial_dependences.append(part_dep)
        print(f'Column {counter}/{len(df_X.columns)} complete')
        counter += 1

    partial_dependences_df = pd.concat(partial_dependences)


    partial_dependences_stats = partial_dependences_df\
        .groupby('column')['AvgPred']\
        .agg(['min', 'max'])\
        .reset_index()
    partial_dependences_stats['range'] = partial_dependences_stats['max'] - partial_dependences_stats['min']

    return partial_dependences_df, partial_dependences_stats

def plot_top_partial_dependences(partial_dependences_df, partial_dependences_stats, top_n):
    """
    Purpose:
        Plots the top n partial dependences in terms of the largest ranges of average prediction (biggest differences across values of a column)
        
    Inputs:
    * partial_dependences_df = data frame with test values and average prediction for each column
    * partial_dependences_stats = min, max, and range of average predictions for each column
    * top_n = how many top columns to plot
    """
    import matplotlib.pyplot as plt

    biggest_ranges = partial_dependences_stats\
        .sort_values('range', ascending = False)\
        .head(top_n)
    print(biggest_ranges)

    for col in biggest_ranges.column:
        col_part_dep = partial_dependences_df.loc[partial_dependences_df.column == col]
        plt.plot(col_part_dep.Value, col_part_dep.AvgPred)
        plt.xlabel(f'{col} Test Value')
        plt.title(f'Partial Dependence Plot for {col}')
        plt.show()


# Run portion ----
if __name__ == '__main__':
    import pandas as pd
    import numpy as np

    df = import_data()
    clean_df = clean_data(df)

    X_train, X_test, y_train, y_test, scalers = prepare_model_data(
        df = clean_df, y_col = 'Bankrupt?'
    )
    
    
    # Models 
    # xgboost model
    xgb_time, xgb_clf = xgb(
        X_train, y_train,
        learning_rate=0.5, n_estimator=10, 
        max_depth=5, random_state=0
    )
    
    # Logisitc Regression
    lr_time, lr_clf = logistic_regression(X_train, y_train)
    
    # Nearest Neighbors - not sure if I did this correctly
    knn_time, knn_clf_results = nearest_neighbors(
        X_train, y_train, 
        X_test, y_test,
        min_n = 1, 
        max_n = 5
    )
    knn_clf, knn_results = knn_clf_results
    
    # neural network model
    nn_time, nn_clf = neural_network(
        X_train, y_train, 
        X_test, y_test,
        num_layers = 1,
        hidden_layer_size = 5000,
        dropout_size = .90,
        patience = 10, 
        batch_size = 100
    )
    
    # random forest model
    rf_time, rf_clf = random_forest(
        X_train,
        y_train,
        n_estimators=10,
        criterion = 'entropy',
        random_state = 0
    )
    
    
    # Model Statistics
    model_stats = model_statistics(
        clfs = [xgb_clf, lr_clf, knn_clf, nn_clf, rf_clf], 
        train_times = [xgb_time, lr_time, knn_time, nn_time, rf_time],

        X_train = X_train, 
        X_test = X_test, 
        y_train = y_train, 
        y_test = y_test
    )
    
    model_stats
    
    # Ensemble Prediction
    train_ens_votes, train_ens_pred = ensemble_prediction(
        clfs = [xgb_clf, lr_clf, knn_clf, nn_clf, rf_clf], 
        model_stats = model_stats, 
        X = X_train
    )
    
    test_ens_votes, test_ens_pred = ensemble_prediction(
        clfs = [xgb_clf, lr_clf, knn_clf, nn_clf, rf_clf], 
        model_stats = model_stats, 
        X = X_test
    )
    
    from sklearn.metrics import accuracy_score, f1_score
    train_acc = accuracy_score(y_true = y_train, y_pred = train_ens_pred)
    test_acc = accuracy_score(y_true = y_test, y_pred = test_ens_pred)
    
    train_f1 = f1_score(y_true = y_train, y_pred = train_ens_pred)
    test_f1 = f1_score(y_true = y_test, y_pred = test_ens_pred)
    
    print(f'Train Accuracy: {np.round(train_acc*100, 1)}%')
    print(f'Test Accuracy: {np.round(test_acc*100, 1)}%')
    print(f'Train F1: {np.round(train_f1*100, 1)}%')
    print(f'Test F1: {np.round(test_f1*100, 1)}%')

    
    
    # Partial Dependence Analysis
    df_X = df.drop('Bankrupt?', axis = 1)
    partial_dependences_df, partial_dependences_stats = partial_dependence_loop(
        df_X = df_X, clf = nn_clf, scalers = scalers, num_test = 15, show_plot = False
    )       
    plot_top_partial_dependences(partial_dependences_df, partial_dependences_stats, top_n = 5)

