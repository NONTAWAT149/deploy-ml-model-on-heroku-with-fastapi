"""
Write at least 3 unit tests.
Unit testing ML can be hard due to the stochasticity
-- at least test if any ML functions return the expected type.
"""
import os

def test_trained_model():
    assert os.path.isfile("./model/model.joblib"), 'model file not found'

def test_encoder():
    assert os.path.isfile("./model/encoder.joblib"), 'encoder file not found'

def test_lb_encoder():
    assert os.path.isfile("./model/lb.joblib"), 'lb encoder file not found'