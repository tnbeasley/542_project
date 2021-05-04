from data import import_data, clean_data, prepare_model_data
from models import model_statistics
from model_diagnostics import ensemble_prediction
from model_diagnostics import partial_dependence, partial_dependence_loop, plot_top_partial_dependences

import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, f1_score
from tensorflow import keras

# Data
df = import_data()
clean_df = clean_data(df)

X_train, X_test, y_train, y_test, scalers, column_names = prepare_model_data(
    df = clean_df, y_col = 'Bankrupt?'
)


# Load Models
rf_clf = pickle.load(open('Models/rf.sav', 'rb'))
xgb_clf = pickle.load(open('Models/xgb.sav', 'rb'))
nn_clf = keras.models.load_model('Models/nn.h5')
knn_clf = pickle.load(open('Models/knn.sav', 'rb'))
lr_clf = pickle.load(open('Models/lr.sav', 'rb'))
model_stats_10 = pd.read_csv('Models/model_stats_10.csv')
model_stats_25 = pd.read_csv('Models/model_stats_25.csv')
model_stats_50 = pd.read_csv('Models/model_stats_50.csv')
model_stats_75 = pd.read_csv('Models/model_stats_75.csv')

# Ensemble Prediction
# train_ens_votes, train_ens_pred = ensemble_prediction(
#     clfs = [xgb_clf, lr_clf, knn_clf, nn_clf, rf_clf], 
#     model_stats = model_stats, 
#     X = X_train
# )

# test_ens_votes, test_ens_pred = ensemble_prediction(
#     clfs = [xgb_clf, lr_clf, knn_clf, nn_clf, rf_clf], 
#     model_stats = model_stats, 
#     X = X_test
# )

# train_acc = accuracy_score(y_true = y_train, y_pred = train_ens_pred)
# test_acc = accuracy_score(y_true = y_test, y_pred = test_ens_pred)

# train_f1 = f1_score(y_true = y_train, y_pred = train_ens_pred)
# test_f1 = f1_score(y_true = y_test, y_pred = test_ens_pred)

# print(f'Train Accuracy: {np.round(train_acc*100, 1)}%')
# print(f'Test Accuracy: {np.round(test_acc*100, 1)}%')
# print(f'\nTrain F1: {np.round(train_f1*100, 1)}%')
# print(f'Test F1: {np.round(test_f1*100, 1)}%')
# print(f'Best Model Test F1: {np.round(model_stats.TestF1.max()*100, 1)}%')


# Top Partial Dependence Plots
top = 3
## XGBoost
xgb_partial_dependences_df = pd.read_csv('PartialDependence/xgb_pd_df.csv')
xgb_partial_dependences_stats = pd.read_csv('PartialDependence/xgb_pd_stats.csv')
plot_top_partial_dependences(xgb_partial_dependences_df, xgb_partial_dependences_stats, top_n = top)

## Logistic Regression
lr_partial_dependences_df = pd.read_csv('PartialDependence/lr_pd_df.csv')
lr_partial_dependences_stats = pd.read_csv('PartialDependence/lr_pd_stats.csv')
plot_top_partial_dependences(lr_partial_dependences_df, lr_partial_dependences_stats, top_n = top)

## K-Nearest Neighbors
knn_partial_dependences_df = pd.read_csv('PartialDependence/knn_pd_df.csv')
knn_partial_dependences_stats = pd.read_csv('PartialDependence/knn_pd_stats.csv')
plot_top_partial_dependences(knn_partial_dependences_df, knn_partial_dependences_stats, top_n = top)

## Neural network
nn_partial_dependences_df = pd.read_csv('PartialDependence/nn_pd_df.csv')
nn_partial_dependences_stats = pd.read_csv('PartialDependence/nn_pd_stats.csv')
plot_top_partial_dependences(nn_partial_dependences_df, nn_partial_dependences_stats, top_n = 5)

## Random Forest
rf_partial_dependences_df = pd.read_csv('PartialDependence/rf_pd_df.csv')
rf_partial_dependences_stats = pd.read_csv('PartialDependence/rf_pd_stats.csv')
plot_top_partial_dependences(rf_partial_dependences_df, rf_partial_dependences_stats, top_n = 5)
# Random Forest Variable Importance Levels
rf_clf.fit(X_train, y_train)
feature_importance = pd.DataFrame({'Variable':column_names,
            'Importance':rf_clf.feature_importances_}).sort_values('Importance', ascending=False)
print("Random Forest Feature Importance:\n")
print(feature_importance.to_string())