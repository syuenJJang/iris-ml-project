import numpy as np
import pytest

from src.data.loader import load_data
from src.features.preprocessing import preprocess
from src.models.model import IrisModel


@pytest.fixture
def sample_data():
    """Fixture for sample data"""
    df = load_data()
    X, y = preprocess(df[:100])
    return X, y


def test_model_initialization():
    """Test model initialization"""
    model = IrisModel()
    assert model.is_trained is False


def test_model_training(sample_data):
    """Test model training"""
    X, y = sample_data
    model = IrisModel()
    model.train(X, y)

    assert model.is_trained is True


def test_model_prediction_shape(sample_data):
    """Test prediction output shape"""
    X, y = sample_data
    model = IrisModel()
    model.train(X, y)

    predictions = model.predict(X[:10])
    assert predictions.shape == (10,)


def test_model_prediction_range(sample_data):
    """Test prediction value range"""
    X, y = sample_data
    model = IrisModel()
    model.train(X, y)

    predictions = model.predict(X)
    assert np.all(np.isin(predictions, [0, 1, 2]))


def test_model_predict_proba(sample_data):
    """Test probability predictions"""
    X, y = sample_data
    model = IrisModel()
    model.train(X, y)

    probas = model.predict_proba(X[:5])
    assert probas.shape == (5, 3)
    assert np.allclose(probas.sum(axis=1), 1.0)


def test_model_predict_without_training():
    """Test prediction fails without training"""
    model = IrisModel()
    X = np.random.rand(10, 4)

    with pytest.raises(ValueError, match="Model must be trained first"):
        model.predict(X)
