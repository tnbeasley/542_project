import numpy as np
import pandas as pd
import pickle
from tensorflow import keras

from data import import_data, clean_data, prepare_model_data
from model_diagnostics import partial_dependence, partial_dependence_loop, plot_top_partial_dependences


if __name__ == '__main__':
    df = import_data()
    clean_df = clean_data(df)
    X_train, X_test, y_train, y_test, scalers = prepare_model_data(
        df = clean_df, y_col = 'Bankrupt?'
    )

    # Load Models
    xgb_clf = pickle.load(open('Models/xgb.sav', 'rb'))
    rf_clf = pickle.load(open('Models/rf.sav', 'rb'))
    nn_clf = keras.models.load_model('Models/nn.h5')
    knn_clf = pickle.load(open('Models/knn.sav', 'rb'))
    lr_clf = pickle.load(open('Models/lr.sav', 'rb'))


    df_X = clean_df.drop('Bankrupt?', axis = 1)
    num_tests = 15

    ## XGBoost
    xgb_partial_dependences_df, xgb_partial_dependences_stats = partial_dependence_loop(
        df_X = df_X, clf = xgb_clf, scalers = scalers, 
        num_test = num_tests, show_plot = False
    )
    xgb_partial_dependences_df.to_csv('PartialDependence/xgb_pd_df.csv')
    xgb_partial_dependences_stats.to_csv('PartialDependence/xgb_pd_stats.csv')
    print('XGBoost Complete')


    ## Logistic Regression
    lr_partial_dependences_df, lr_partial_dependences_stats = partial_dependence_loop(
        df_X = df_X, clf = lr_clf, scalers = scalers, 
        num_test = num_tests, show_plot = False
    )
    lr_partial_dependences_df.to_csv('PartialDependence/lr_pd_df.csv')
    lr_partial_dependences_stats.to_csv('PartialDependence/lr_pd_stats.csv')
    print('Logistic Regression Complete')


    ## K-Nearest Neighbors
    knn_partial_dependences_df, knn_partial_dependences_stats = partial_dependence_loop(
        df_X = df_X, clf = knn_clf, scalers = scalers, 
        num_test = num_tests, show_plot = False
    )
    knn_partial_dependences_df.to_csv('PartialDependence/knn_pd_df.csv')
    knn_partial_dependences_stats.to_csv('PartialDependence/knn_pd_stats.csv')
    print('K-Nearest Neighbors Complete')


    ## Neural network
    nn_partial_dependences_df, nn_partial_dependences_stats = partial_dependence_loop(
        df_X = df_X, clf = nn_clf, scalers = scalers, 
        num_test = num_tests, show_plot = False
    )
    nn_partial_dependences_df.to_csv('PartialDependence/nn_pd_df.csv')
    nn_partial_dependences_stats.to_csv('PartialDependence/nn_pd_stats.csv')
    print('Neural Network Complete')


    ## Random Forest
    rf_partial_dependences_df, rf_partial_dependences_stats = partial_dependence_loop(
        df_X = df_X, clf = rf_clf, scalers = scalers, 
        num_test = num_tests, show_plot = False
    )
    rf_partial_dependences_df.to_csv('PartialDependence/rf_pd_df.csv')
    rf_partial_dependences_stats.to_csv('PartialDependence/rf_pd_stats.csv')
    print('Random Forest Complete')
    
    print('Partial dependence files saved in PartialDependence folder')