"""Module to evaluate the SIFT features with an SVM classifier."""

import numpy as np

from sklearn.svm import LinearSVC

def main() -> None:
    """Main function to evaluate the SIFT features with an SVM classifier."""
    # Load the processed data
    data = np.load("./processed_cifar10_sift.npz", allow_pickle=True)
    X_train, X_test, y_train, y_test = (
        data["X_train"],
        data["X_test"],
        data["y_train"],
        data["y_test"],
    )

    # Train an SVM classifier
    svm = LinearSVC(random_state=42, dual="auto")
    svm.fit(X_train, y_train)

    # Evaluate the model
    accuracy = svm.score(X_test, y_test)
    print(f"SVM accuracy: {accuracy:.2f}")


if __name__ == "__main__":
    main()
