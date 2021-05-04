from time_it import time_it

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
def logistic_regression(X_train, y_train, X_test, y_test):
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
    
    import pandas as pd
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.metrics import AUC
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import StratifiedKFold
    
    from numpy.random import seed
    seed(1)
    
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

def cross_val(clf, X, y, n_splits=5):
    """
    Inputs:
    * clf = untrained model
    * X = X data
    * y = y data
    
    Output:
    * F1 mean
    * F1 standard deviation
    """
    import numpy as np
    from sklearn.metrics import f1_score, make_scorer
    from sklearn.model_selection import KFold, cross_val_score
    
    scorer = make_scorer(f1_score, average='micro')
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    scores = cross_val_score(clf, X, y, scoring=scorer, cv=cv)
    return round(np.mean(scores), 2), round(np.std(scores), 2)

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
        - F1 Score: mean and standard deviation
        
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from scipy import stats
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.metrics import make_scorer
    
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