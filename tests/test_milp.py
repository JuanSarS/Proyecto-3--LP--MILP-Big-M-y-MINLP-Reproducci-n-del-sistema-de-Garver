from pathlib import Path

import pytest

from src.data_loader import load_network
from src.milp_bigm import solve_milp
from src.network import expansion_cost


DATA_DIR = Path(__file__).parents[1] / "data"


def _solve(network):
    try:
        return solve_milp(network)
    except RuntimeError as exc:
        message = str(exc).lower()
        if "gurobi" in message and any(
            word in message for word in ("license", "restricted", "required", "not found")
        ):
            pytest.skip(str(exc))
        raise


def test_milp_solution_is_a_feasible_generic_expansion():
    result = _solve(load_network(DATA_DIR))
    network = load_network(DATA_DIR)
    assert result.power_flow.is_feasible
    assert result.power_flow.thermal_violations == ()
    assert result.objective == pytest.approx(expansion_cost(network, result.expansion))
    for key, count in result.expansion.items():
        assert 0 <= count <= network.candidate_by_key[key].max_lines


def test_milp_counts_candidate_circuits_separately_from_existing_lines():
    result = _solve(load_network(DATA_DIR))
    branches = result.power_flow.expanded_network.branches
    for key, corridor in load_network(DATA_DIR).candidate_by_key.items():
        assert sum(b.kind == "new" and b.corridor == key for b in branches) == result.expansion[key]
        assert sum(b.kind == "existing" and b.corridor == key for b in branches) == sum(
            (line.from_bus, line.to_bus) == key
            for line in load_network(DATA_DIR).existing_lines
        )
