from data import import_data, clean_data, prepare_model_data
from models import xgb, logistic_regression, nearest_neighbors, random_forest, neural_network, model_statistics

import pandas as pd
import numpy as np
import pickle

if __name__ == '__main__':
    print('Importing Data')
    df = import_data()
    
    print('Cleaning Data')
    clean_df = clean_data(df)

    print('Train/Test Splitting Data')
    X_train, X_test, y_train, y_test, scalers = prepare_model_data(
        df = clean_df, y_col = 'Bankrupt?'
    )
    
    # Models 
    # xgboost model
    print('Training XGBoost Model')
    xgb_time, xgb_clf = xgb(
        X_train, y_train,
        learning_rate=0.5, n_estimator=10, 
        max_depth=5, random_state=0
    )
    pickle.dump(xgb_clf, open('Models/xgb.sav', 'wb'))
    print('XGBoost Model Saved')
    
    # Logistic Regression
    print('Training Logistic Regression Model')
    lr_time, lr_clf = logistic_regression(X_train, y_train, X_test, y_test)
    pickle.dump(lr_clf, open('Models/lr.sav', 'wb'))
    print('Logistic Regression Model Saved')
    
    # Nearest Neighbors - not sure if I did this correctly
    print('Training K-Nearest Neighbors Model')
    knn_time, knn_clf_results = nearest_neighbors(
        X_train, y_train, 
        X_test, y_test,
        min_n = 1, 
        max_n = 5
    )
    knn_clf, knn_results = knn_clf_results
    pickle.dump(knn_clf, open('Models/knn.sav', 'wb'))
    print('K-Nearest Neighbors Model Saved')
    
    # neural network model
    print('Training Neural Network Model')
    nn_time, nn_clf = neural_network(
        X_train, y_train, 
        X_test, y_test,
        num_layers = 1,
        hidden_layer_size = 512,
        dropout_size = .90,
        patience = 25, 
        batch_size = 500
    )
    nn_clf.save('Models/nn.h5')
    print('Neural Network Model Saved')
    
    # random forest model
    print('Training Random Forest Model')
    rf_time, rf_clf = random_forest(
        X_train,
        y_train,
        n_estimators=10,
        criterion = 'entropy',
        random_state = 0
    )
    pickle.dump(rf_clf, open('Models/rf.sav', 'wb'))
    print('Random Forest Model Saved')
    
    # Model Statistics
    print('Calculating Model Statistics')
       
    model_stats_10 = model_statistics(
        clfs = [xgb_clf, lr_clf, knn_clf, nn_clf, rf_clf], 
        train_times = [xgb_time, lr_time, knn_time, nn_time, rf_time],

        X_train = X_train, 
        X_test = X_test, 
        y_train = y_train, 
        y_test = y_test,
        
        cutoff = .10
    )
    model_stats_10.to_csv('Models/model_stats_10.csv', index = False)
    
    model_stats_25 = model_statistics(
        clfs = [xgb_clf, lr_clf, knn_clf, nn_clf, rf_clf], 
        train_times = [xgb_time, lr_time, knn_time, nn_time, rf_time],

        X_train = X_train, 
        X_test = X_test, 
        y_train = y_train, 
        y_test = y_test,
        
        cutoff = .25
    )
    model_stats_25.to_csv('Models/model_stats_25.csv', index = False)
    
    model_stats_50 = model_statistics(
        clfs = [xgb_clf, lr_clf, knn_clf, nn_clf, rf_clf], 
        train_times = [xgb_time, lr_time, knn_time, nn_time, rf_time],

        X_train = X_train, 
        X_test = X_test, 
        y_train = y_train, 
        y_test = y_test,
        
        cutoff = .5
    )
    model_stats_50.to_csv('Models/model_stats_50.csv', index = False)
    
    model_stats_75 = model_statistics(
        clfs = [xgb_clf, lr_clf, knn_clf, nn_clf, rf_clf], 
        train_times = [xgb_time, lr_time, knn_time, nn_time, rf_time],

        X_train = X_train, 
        X_test = X_test, 
        y_train = y_train, 
        y_test = y_test,
        
        cutoff = .75
    )
    model_stats_75.to_csv('Models/model_stats_75.csv', index = False)
    
    print('Model Statistics Saved')