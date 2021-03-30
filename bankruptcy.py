import pandas as pd
import numpy as np

def import_data():
    """
    This function imports the data and cleans up any undesired features of the table as imported
    """
    df = pd.read_csv('Company_Bankruptcy_Prediction.csv')
    return df

def clean_data(df):
    """
    This function cleans up the data
    """
    return clean_df

def prepare_model_data(df):
    """
    This function prepares the data to be modeled
    """
    return model_df

if __name__ == 'main':
    df = import_data()
    clean_df = clean_data(df)
    model_df = prepare_model_data(clean_df)