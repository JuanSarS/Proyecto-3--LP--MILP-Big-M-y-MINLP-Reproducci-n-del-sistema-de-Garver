from pathlib import Path

import pytest

from src.data_loader import load_network
from src.minlp import solve_minlp
from src.network import expansion_cost


DATA_DIR = Path(__file__).parents[1] / "data"


def _solve(network):
    try:
        return solve_minlp(network)
    except RuntimeError as exc:
        message = str(exc).lower()
        if "gurobi" in message and any(
            word in message for word in ("license", "restricted", "required", "not found")
        ):
            pytest.skip(str(exc))
        raise


def test_minlp_solution_is_validated_by_dc_power_flow():
    network = load_network(DATA_DIR)
    result = _solve(network)
    assert result.power_flow.is_feasible
    assert result.power_flow.thermal_violations == ()
    assert result.objective == pytest.approx(expansion_cost(network, result.expansion))


def test_minlp_expansion_respects_each_corridor_limit():
    network = load_network(DATA_DIR)
    result = _solve(network)
    assert set(result.expansion) == set(network.candidate_by_key)
    for key, count in result.expansion.items():
        assert isinstance(count, int)
        assert 0 <= count <= network.candidate_by_key[key].max_lines
