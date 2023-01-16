import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN



def cluster_data_and_plot(file_location:str, eps:float, algorithm:str, min_samples:int, data_col1:str, data_col2:str):
    """
    This function loads data from a csv file, clusters it using DBSCAN, and plots the data before and after clustering.

    Parameters:
    file_location (str): The path to the csv file containing the data.
    eps (float): The maximum distance between two samples for them to be considered as in the same neighborhood.
    min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.
    data_col1 (str): The first column  for clustering.
    data_col1 (str): The second column for clustering.
   
    Returns:
    None
    """
    # Loading dataset into DataFrame
    data = pd.read_csv(file_location)
    #droping end rows with NaN values
    data.dropna(how='all',inplace=True)
    # Defining the columns that used for clustering
    X = data[[data_col1, data_col2]]
    print(data)
    
    # Plotting the original data
    plt.scatter(X[data_col1], X[data_col2])
    plt.xlabel(data_col1)
    plt.ylabel(data_col2)
    plt.title("Original Data before Clustering")
    plt.show()
    
    # Initializing and fitting the DBSCAN model
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(X)

    # Cluster labels for each data point
    labels = dbscan.labels_

    # Plotting the clustered data
    plt.scatter(X[data_col1], X[data_col2], c=labels)
    plt.xlabel(data_col1)
    plt.ylabel(data_col2)
    plt.title("Clustered Data")
    plt.show()

#Reading and clustering
cluster_data_and_plot(file_location="C:\\Users\\shobi\\OneDrive\\Desktop\\Anna\\DM\\Sales_Transactions_Dataset_Weekly.csv", eps=4.5, algorithm= 'kd_tree', min_samples=25, data_col1="W21", data_col2="W0")

