from pathlib import Path

import numpy as np
import pytest

from src.data_loader import load_network
from src.network import (
    build_bbus,
    build_incidence,
    connected_components,
    expand_network,
    expansion_cost,
)


DATA_DIR = Path(__file__).parents[1] / "data"


@pytest.fixture()
def network():
    return load_network(DATA_DIR)


def test_existing_bbus_is_generated_and_keeps_isolated_bus(network):
    bbus = build_bbus(network.bus_ids, network.existing_lines)
    assert np.allclose(bbus, bbus.T)
    assert np.allclose(bbus.sum(axis=1), 0.0)
    assert bbus[network.bus_ids.index(4), network.bus_ids.index(4)] == pytest.approx(4.1666667)
    assert bbus[network.bus_ids.index(5), network.bus_ids.index(5)] == pytest.approx(10.0)
    assert np.allclose(bbus[5, :], 0.0)
    assert np.allclose(bbus[:, 5], 0.0)


def test_incidence_supports_nonconsecutive_ids():
    incidence = build_incidence((10, 20, 30), [])
    assert incidence.shape == (0, 3)
    custom = build_incidence((10, 20, 30), [
        type("Line", (), {"from_bus": 30, "to_bus": 10})()
    ])
    assert custom.shape == (1, 3)
    assert np.allclose(custom.sum(axis=1), 0.0)


def test_initial_network_has_unbalanced_island(network):
    islands = connected_components(network.bus_ids, network.existing_lines)
    assert set(islands) == {frozenset({1, 2, 3, 4, 5}), frozenset({6})}
    bus6 = next(bus for bus in network.buses if bus.bus == 6)
    assert bus6.net_injection_pu == pytest.approx(5.45)


def test_reference_expansion_preserves_existing_parallel_circuit(network):
    expanded = expand_network(
        network, {(6, 2): 4, (6, 4): 2, (6, 5): 0, (3, 5): 1}
    )
    assert len([b for b in expanded.new_branches if b.corridor == (6, 2)]) == 4
    assert len([b for b in expanded.new_branches if b.corridor == (6, 4)]) == 2
    assert len([b for b in expanded.new_branches if b.corridor == (3, 5)]) == 1
    assert len([b for b in expanded.branches if b.corridor == (3, 5)]) == 2
    assert len(connected_components(expanded.bus_ids, expanded.branches)) == 1
    assert expansion_cost(network, {(6, 2): 4, (6, 4): 2, (6, 5): 0, (3, 5): 1}) == pytest.approx(1000)


def test_expansion_rejects_noninteger_or_out_of_range(network):
    with pytest.raises(TypeError):
        expand_network(network, {(6, 2): 1.5})
    with pytest.raises(ValueError):
        expand_network(network, {(6, 2): 5})
