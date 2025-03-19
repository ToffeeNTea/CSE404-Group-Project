import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def get_results(test_size=0.2, random_state=22):
    """
    Train a logistic regression model on the preprocessed image features and labels.
    """
    # Load preprocessed data
    X = np.load("cache/image_features.npy")
    y = np.load("cache/image_labels.npy")

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Train logistic regression model
    clf = LogisticRegression(max_iter=1000, solver='lbfgs')  # multi_class='multinomial' is default after 0.22
    clf.fit(X_train, y_train)

    # Evaluate model
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    # Save trained model
    import joblib
    joblib.dump(clf, "cache/logistic_regression_model.pkl")