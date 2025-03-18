import logistic_regression
import data_prep
import extract_features
import model_eval
import generate_cluster_map
import json
import os

def main():
    image_folder = "../../database/dataset/"
    coords_file = "cache/coords.csv"
    clusters = 10
    force_recreate = True

    print("Starting data preparation")
    # Load data and generate cluster map
    df = data_prep.prepare_data(coords_file=coords_file, clusters=clusters, force_recreate=force_recreate)
    generate_cluster_map.generate_cluser_map(coords_file="cache/coords_test.csv")

    print("Data preparation complete")
    # Extract features into cache files
    extract_features.load_dataframe(df=df, image_folder="../../database/dataset/", save=True,)

    print("Feature extraction")
    # Train logistic regression model and save it
    logistic_regression.get_results()

    print("Model training complete")
    # Evaluate model
    metrics = model_eval.evaluate_model()

    print("Saving metrics")
    # display and dump metrics
    print(metrics)
    if not os.path.exists('metrics'):
        os.makedirs('metrics')
    json.dump(metrics, open("metrics/metrics.json", "w"))

    output = extract_features.test(rel_image_path='test6.png')
    print(f"Predicted cluster: {output}")
    output = extract_features.test(rel_image_path='test7.png')
    print(f"Predicted cluster: {output}")
    



# entry point
if __name__ == "__main__":
    main()




