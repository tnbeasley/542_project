from data import import_data, clean_data, prepare_model_data
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

X_train, X_test, y_train, y_test, scalers = prepare_model_data(
    df = clean_df, y_col = 'Bankrupt?'
)


# Load Models
rf_clf = pickle.load(open('Models/rf.sav', 'rb'))
xgb_clf = pickle.load(open('Models/xgb.sav', 'rb'))
nn_clf = keras.models.load_model('Models/nn.h5')
knn_clf = pickle.load(open('Models/knn.sav', 'rb'))
lr_clf = pickle.load(open('Models/lr.sav', 'rb'))
model_stats = pd.read_csv('Models/model_stats.csv')


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

train_acc = accuracy_score(y_true = y_train, y_pred = train_ens_pred)
test_acc = accuracy_score(y_true = y_test, y_pred = test_ens_pred)

train_f1 = f1_score(y_true = y_train, y_pred = train_ens_pred)
test_f1 = f1_score(y_true = y_test, y_pred = test_ens_pred)

print(f'Train Accuracy: {np.round(train_acc*100, 1)}%')
print(f'Test Accuracy: {np.round(test_acc*100, 1)}%')
print(f'\nTrain F1: {np.round(train_f1*100, 1)}%')
print(f'Test F1: {np.round(test_f1*100, 1)}%')
print(f'Best Model Test F1: {np.round(model_stats.TestF1.max()*100, 1)}%')
