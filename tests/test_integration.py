from pathlib import Path

import pytest

from src.data_loader import load_network
from src.network import expansion_cost
from src.validation import validate_physical_solution


DATA_DIR = Path(__file__).parents[1] / "data"
REFERENCE = {(6, 2): 4, (6, 4): 2, (6, 5): 0, (3, 5): 1}


def test_reference_expansion_cross_validation():
    network = load_network(DATA_DIR)
    result = validate_physical_solution(network, REFERENCE)
    assert result.is_feasible
    assert expansion_cost(network, REFERENCE) == pytest.approx(1000)
