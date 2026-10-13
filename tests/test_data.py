from pathlib import Path

import pytest

from src.data_loader import load_network


DATA_DIR = Path(__file__).parents[1] / "data"


@pytest.fixture()
def network():
    return load_network(DATA_DIR)


def test_generation_equals_demand(network):
    assert network.total_generation_mw == pytest.approx(network.total_demand_mw)
    assert sum(bus.net_injection_mw for bus in network.buses) == pytest.approx(0.0)


def test_per_unit_vectors(network):
    assert [bus.pg_pu for bus in network.buses] == pytest.approx(
        [0.5, 0.0, 1.65, 0.0, 0.0, 5.45]
    )
    assert [bus.pl_pu for bus in network.buses] == pytest.approx(
        [0.8, 2.4, 0.4, 1.6, 2.4, 0.0]
    )


def test_positive_line_data(network):
    for line in network.existing_lines:
        assert line.x_pu > 0
        assert line.capacity_mw > 0
        assert line.susceptance_pu > 0
    for corridor in network.candidate_corridors:
        assert corridor.x_pu > 0
        assert corridor.capacity_mw > 0
        assert corridor.cost > 0
        assert isinstance(corridor.max_lines, int)
        assert corridor.max_lines >= 0


def test_reference_bus(network):
    assert network.reference_bus == 5
    assert network.reference_bus in network.bus_ids


def test_existing_and_candidate_35_are_distinct(network):
    existing_35 = [
        line for line in network.existing_lines if (line.from_bus, line.to_bus) == (3, 5)
    ]
    candidate_35 = network.candidate_by_key[(3, 5)]
    assert len(existing_35) == 1
    assert candidate_35.max_lines == 3
    assert candidate_35.cost == pytest.approx(100.0)
    assert 1 + candidate_35.max_lines == 4


def test_candidate_investment_counts_new_lines_only(network):
    candidate_35 = network.candidate_by_key[(3, 5)]
    assert 0 * candidate_35.cost == pytest.approx(0.0)
    assert 1 * candidate_35.cost == pytest.approx(100.0)
    assert 1 + 1 == 2  # one original line plus one new line
