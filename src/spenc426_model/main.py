import logistic_regression
import data_prep
import extract_features
import model_eval

def main():
    image_folder = "../../database/dataset"
    clusters = 10
    force_recreate = False

    # Load data
    df = data_prep.prepare_data(image_folder=image_folder, clusters=clusters, force_recreate=force_recreate)

    # Extract features into cache files
    extract_features.extract_features(image_folder="../../database/dataset", df=df)

    # Train logistic regression model
    logistic_regression.get_results()

    # Evaluate model
    model_eval.evaluate_model()


