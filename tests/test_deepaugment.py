# (C) 2024 Peter Norvig

import pytest
import pandas as pd
from deepaugment.deepaugment import DeepAugment

def test_deepaugment_integration():
    """
    Tests the DeepAugment optimizer for a few iterations on the CIFAR-10 dataset.
    """
    # Given
    config = {
        "train_set_size": 100, # Use a small subset for testing
        "opt_samples": 1,
        "opt_last_n_epochs": 1,
        "opt_initial_points": 1,
        "child_epochs": 1,
        "child_batch_size": 8,
    }

    # When
    deepaug = DeepAugment("cifar10", config=config)
    best_policies = deepaug.optimize(2) # Run for 2 iterations

    # Then
    assert isinstance(best_policies, pd.DataFrame)
    assert not best_policies.empty
