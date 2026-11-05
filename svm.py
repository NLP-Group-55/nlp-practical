"""Linear SVM classifier for encoded text features."""

from sklearn.svm import LinearSVC


def train_svm(features, labels, *, seed=42):
    """Fit a linear SVM and return it ready to predict on new features."""
    model = LinearSVC(random_state=seed)
    return model.fit(features, labels)
