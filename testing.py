import pytest
import numpy as np
from Final_LDA import LDAModel, RDAModelResults, manual_train_test_split  


# Sample data for testing
@pytest.fixture
def sample_data():
    # Create a small dataset
    X = np.array([[0.1, 0.2], [0.2, 0.3], [0.1, 0.4], [0.5, 0.6]])
    y = np.array([0, 0, 1, 1])
    return X, y


def test_lda_model_fit(sample_data):
    X, y = sample_data
    model = LDAModel(nComponents=1)
    model.fit(X, y)

    assert model.Eigens is not None
    assert model.classMeans is not None
    assert len(model.classMeans) == len(np.unique(y))


def test_lda_model_transform(sample_data):
    X, y = sample_data
    model = LDAModel(nComponents=1)
    model.fit(X, y)
    transformed_X = model.transform(X)

    assert transformed_X.shape[1] == model.nComponents


def test_rda_model_results_predict(sample_data):
    X, y = sample_data
    model = LDAModel(nComponents=1)
    model.fit(X, y)
    model_results = RDAModelResults(model)
    y_pred = model_results.predict(X)

    assert y_pred.shape == y.shape
    assert np.all(np.isin(y_pred, np.unique(y)))  # Predictions should be within the classes


def test_manual_train_test_split(sample_data):
    X, y = sample_data
    X_train, X_test, y_train, y_test = manual_train_test_split(X, y, test_size=0.5, random_state=42)

    assert X_train.shape[0] + X_test.shape[0] == X.shape[0]
    assert y_train.shape[0] + y_test.shape[0] == y.shape[0]


if __name__ == "__main__":
    pytest.main()
