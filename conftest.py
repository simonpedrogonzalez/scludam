import numpy as np
import pytest

@pytest.fixture(autouse=True)
def random():
    np.random.seed(0)
