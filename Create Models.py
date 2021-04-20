from data import import_data, clean_data, prepare_model_data
from models import xgb, logistic_regression, nearest_neighbors, random_forest, neural_network, model_statistics

import pandas as pd
import numpy as np
import pickle

if __name__ == '__main__':
    df = import_data()
    clean_df = clean_data(df)

    X_train, X_test, y_train, y_test, scalers = prepare_model_data(
        df = clean_df, y_col = 'Bankrupt?'
    )
    
    # Models 
    # xgboost model
    xgb_time, xgb_clf = xgb(
        X_train, y_train,
        learning_rate=0.5, n_estimator=10, max_depth=5
    )
    pickle.dump(xgb_clf, open('Models/xgb.sav', 'wb'))
#     xgb_clf = pickle.load(open(filename, 'rb'))
    
    # Logisitc Regression
    lr_time, lr_clf = logistic_regression(X_train, y_train, X_test, y_test)
    pickle.dump(lr_clf, open('Models/lr.sav', 'wb'))
    
    # Nearest Neighbors - not sure if I did this correctly
    knn_time, knn_clf_results = nearest_neighbors(
        X_train, y_train, 
        X_test, y_test,
        min_n = 1, 
        max_n = 5
    )
    knn_clf, knn_results = knn_clf_results
    pickle.dump(knn_clf, open('Models/knn.sav', 'wb'))
    
    # neural network model
    nn_time, nn_clf = neural_network(
        X_train, y_train, 
        X_test, y_test,
        num_layers = 1,
        hidden_layer_size = 1000,
        dropout_size = .90,
        patience = 10, 
        batch_size = 100
    )
    nn_clf.save('Models/nn.h5')
    
    # random forest model
    rf_time, rf_clf = random_forest(
        X_train,
        y_train,
        n_estimators=10,
        criterion = 'entropy',
        random_state = 0
    )
    pickle.dump(rf_clf, open('Models/rf.sav', 'wb'))
    
    
    # Model Statistics
    model_stats = model_statistics(
        clfs = [xgb_clf, lr_clf, knn_clf, nn_clf, rf_clf], 
        train_times = [xgb_time, lr_time, knn_time, nn_time, rf_time],

        X_train = X_train, 
        X_test = X_test, 
        y_train = y_train, 
        y_test = y_test
    )
    
    model_stats.to_csv('Models/model_stats.csv', index = False)
