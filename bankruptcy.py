# Data engineering ----
def import_data():
    """
    This function imports the data and cleans up any undesired features of the table as imported
    """
    df = pd.read_csv('Company_Bankruptcy_Prediction.csv')
    return df


def clean_data(df):
    """
    This function cleans up the data
    
    Yang/Leslie
    """
    
    # Fix column names
    
    # Drop columns with few/one value
    
    # Check for NA values
    
    return clean_df


def prepare_model_data(df):
    """
    This function prepares the data to be modeled
    
    Tanner
    """
    
    from sklearn.preprocessing import StandardScaler
    
    # Train/Test split
    
    # Standardize data
    
    return model_df


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
    model_df = prepare_model_data(clean_df)