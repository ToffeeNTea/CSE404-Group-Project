import pandas as pd
from sklearn.cluster import KMeans
import os

def prepare_data(coords_file="cache/coords.csv", clusters=10, force_recreate=False):
    df = pd.read_csv(coords_file)

    num_clusters = clusters
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    df['region_label'] = kmeans.fit_predict(df[['latitude', 'longitude']])

    coord_test_path = 'cache/coords_test.csv'
    if not os.path.exists(coord_test_path) or force_recreate:
        df.to_csv(coord_test_path, index=False)

    return df