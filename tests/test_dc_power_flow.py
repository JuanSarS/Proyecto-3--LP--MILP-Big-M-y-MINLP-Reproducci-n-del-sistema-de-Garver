from pathlib import Path

import numpy as np
import pytest

from src.data_loader import load_network
from src.dc_power_flow import check_thermal_limits, solve_dc_power_flow


DATA_DIR = Path(__file__).parents[1] / "data"
REFERENCE_EXPANSION = {(6, 2): 4, (6, 4): 2, (6, 5): 0, (3, 5): 1}


@pytest.fixture()
def network():
    return load_network(DATA_DIR)


def test_initial_flow_reports_unbalanced_island(network):
    result = solve_dc_power_flow(network)
    assert not result.is_feasible
    assert result.angles is None
    assert any("net injection 5.45" in message for message in result.diagnostics)
    assert set(result.islands) == {frozenset({1, 2, 3, 4, 5}), frozenset({6})}


def test_reference_expansion_angles_and_balance(network):
    result = solve_dc_power_flow(network, REFERENCE_EXPANSION)
    assert result.is_feasible
    assert result.angles == pytest.approx(
        [0.10599816, 0.31100276, 0.18700092, 0.29648574, 0.0, 0.57866375],
        abs=2e-7,
    )
    assert np.max(np.abs(result.bus_residuals)) <= 1e-10
    assert check_thermal_limits(result)


def test_reference_expansion_flows(network):
    result = solve_dc_power_flow(network, REFERENCE_EXPANSION)
    existing = {
        item.branch.corridor: item.flow_mw
        for item in result.branch_flows
        if item.branch.kind == "existing"
    }
    assert existing[(1, 2)] == pytest.approx(-51.25115, abs=1e-3)
    assert existing[(1, 4)] == pytest.approx(-31.74793, abs=1e-3)
    assert existing[(1, 5)] == pytest.approx(52.99908, abs=1e-3)
    assert existing[(2, 3)] == pytest.approx(62.00092, abs=1e-3)
    assert existing[(2, 4)] == pytest.approx(3.62925, abs=1e-3)
    assert existing[(3, 5)] == pytest.approx(93.50046, abs=1e-3)
    assert result.flows_by_corridor_mw[(6, 2)] == pytest.approx(356.88132, abs=1e-3)
    assert result.flows_by_corridor_mw[(6, 4)] == pytest.approx(188.11868, abs=1e-3)
    assert result.flows_by_corridor_mw[(3, 5)] == pytest.approx(187.00092, abs=1e-3)
