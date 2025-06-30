from scripts.rollback import rollback_model
import os

def test_rollback():
    # Simulate rollback
    rollback_model()
    assert os.path.exists('models/latest_model.pkl')