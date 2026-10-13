from pathlib import Path

import pytest

from src.data_loader import load_network
from src.lp_villasana import solve_lp_villasana


DATA_DIR = Path(__file__).parents[1] / "data"


@pytest.fixture()
def network():
    return load_network(DATA_DIR)


def test_lp_connects_and_balances_network(network):
    result = solve_lp_villasana(network)
    assert result.converged
    assert result.is_feasible
    assert result.fictitious_pd_pu == pytest.approx(0.0, abs=1e-8)
    assert sum(result.expansion.values()) > 0
    assert len(result.iterations) > 1


def test_lp_respects_candidate_limits(network):
    result = solve_lp_villasana(network)
    for key, count in result.expansion.items():
        assert 0 <= count <= network.candidate_by_key[key].max_lines


def test_lp_iteration_limit_is_honored(network):
    result = solve_lp_villasana(network, max_iterations=0)
    assert not result.converged
    assert len(result.iterations) == 1
