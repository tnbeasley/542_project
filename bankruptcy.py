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
    
    # return clean_df
    pass


def prepare_model_data(df):
    """
    This function prepares the data to be modeled
    
    Tanner
    """
    
    from sklearn.preprocessing import StandardScaler
    
    # Train/Test split
    
    # Standardize data
    
    # return model_df
    pass


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


if __name__ == '__main__':
    import pandas as pd
    import numpy as np

    df = import_data()
    clean_df = clean_data(df)
    model_df = prepare_model_data(clean_df)
    
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

    