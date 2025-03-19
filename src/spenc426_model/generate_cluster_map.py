import pandas as pd
import matplotlib.pyplot as plt

def generate_cluser_map(coords_file="cache/coords_test.csv"):

    # Load the dataset
    df = pd.read_csv(coords_file, header=None, names=["Latitude", "Longitude", "Cluster"])

    # Remove the incorrect header row if present
    df = df.iloc[1:].reset_index(drop=True)

    # Convert columns to numeric values
    df["Latitude"] = pd.to_numeric(df["Latitude"])
    df["Longitude"] = pd.to_numeric(df["Longitude"])
    df["Cluster"] = pd.to_numeric(df["Cluster"])

    # Create a scatter plot of the clustered points
    plt.figure(figsize=(12, 6))
    scatter = plt.scatter(df["Longitude"], df["Latitude"], c=df["Cluster"], cmap="tab10", edgecolors="k", s=50)

    # Add legend for clusters
    plt.legend(*scatter.legend_elements(), title="Clusters")

    # Labels and formatting
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("K-Means Clustering of Geographical Coordinates")

    # Show plot
    plt.savefig("metrics/cluster_map.png")
generate_cluser_map()