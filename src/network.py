"""Generic network-topology utilities for the DC power-flow model."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Iterable, Mapping, Sequence

import numpy as np

from .data_loader import CandidateCorridor, ExistingLine, NetworkData


Line = ExistingLine | CandidateCorridor


@dataclass(frozen=True)
class PhysicalLine:
    """One physical circuit, retaining its origin and corridor metadata."""

    from_bus: int
    to_bus: int
    x_pu: float
    susceptance_pu: float
    capacity_mw: float
    kind: str
    corridor: tuple[int, int]
    circuit_index: int


@dataclass(frozen=True)
class ExpandedNetwork:
    """Physical network after adding the requested candidate circuits."""

    bus_ids: tuple[int, ...]
    base_mva: float
    reference_bus: int
    buses: tuple
    branches: tuple[PhysicalLine, ...]
    expansion: dict[tuple[int, int], int]

    @property
    def existing_branches(self) -> tuple[PhysicalLine, ...]:
        return tuple(branch for branch in self.branches if branch.kind == "existing")

    @property
    def new_branches(self) -> tuple[PhysicalLine, ...]:
        return tuple(branch for branch in self.branches if branch.kind == "new")


def _endpoints(line: object) -> tuple[int, int]:
    return int(line.from_bus), int(line.to_bus)


def build_incidence(
    bus_ids: Sequence[int], lines: Iterable[Line | PhysicalLine]
) -> np.ndarray:
    """Build A with +1 at ``from_bus`` and -1 at ``to_bus``."""

    positions = {bus_id: index for index, bus_id in enumerate(bus_ids)}
    matrix = np.zeros((0, len(bus_ids)), dtype=float)
    rows: list[np.ndarray] = []
    for line in lines:
        from_bus, to_bus = _endpoints(line)
        if from_bus not in positions or to_bus not in positions:
            raise ValueError(f"Line endpoint ({from_bus}, {to_bus}) is not in bus_ids")
        row = np.zeros(len(bus_ids), dtype=float)
        row[positions[from_bus]] = 1.0
        row[positions[to_bus]] = -1.0
        rows.append(row)
    return np.vstack(rows) if rows else matrix


def build_bbus(
    bus_ids: Sequence[int], lines: Iterable[Line | PhysicalLine]
) -> np.ndarray:
    """Build Bbus by accumulating positive branch susceptances."""

    positions = {bus_id: index for index, bus_id in enumerate(bus_ids)}
    bbus = np.zeros((len(bus_ids), len(bus_ids)), dtype=float)
    for line in lines:
        from_bus, to_bus = _endpoints(line)
        if from_bus not in positions or to_bus not in positions:
            raise ValueError(f"Line endpoint ({from_bus}, {to_bus}) is not in bus_ids")
        i, j = positions[from_bus], positions[to_bus]
        b = float(line.susceptance_pu)
        bbus[i, i] += b
        bbus[j, j] += b
        bbus[i, j] -= b
        bbus[j, i] -= b
    return bbus


def connected_components(
    bus_ids: Sequence[int], lines: Iterable[Line | PhysicalLine]
) -> tuple[frozenset[int], ...]:
    """Return connected components, including isolated buses."""

    adjacency = {bus_id: set() for bus_id in bus_ids}
    for line in lines:
        from_bus, to_bus = _endpoints(line)
        if from_bus not in adjacency or to_bus not in adjacency:
            raise ValueError(f"Line endpoint ({from_bus}, {to_bus}) is not in bus_ids")
        adjacency[from_bus].add(to_bus)
        adjacency[to_bus].add(from_bus)

    remaining = set(bus_ids)
    components: list[frozenset[int]] = []
    while remaining:
        start = min(remaining, key=lambda value: list(bus_ids).index(value))
        stack = [start]
        component: set[int] = set()
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            remaining.discard(current)
            stack.extend(adjacency[current] - component)
        components.append(frozenset(component))
    return tuple(components)


def expand_network(
    network: NetworkData, expansion: Mapping[tuple[int, int], int] | None = None
) -> ExpandedNetwork:
    """Materialize existing circuits plus the requested new circuits."""

    requested = dict(expansion or {})
    candidates = network.candidate_by_key
    unknown = set(requested) - set(candidates)
    if unknown:
        raise ValueError(f"Expansion contains unknown corridors: {sorted(unknown)}")
    normalized: dict[tuple[int, int], int] = {}
    for key, value in requested.items():
        corridor = candidates[key]
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"Expansion for {key} must be an integer")
        count = int(value)
        if not 0 <= count <= corridor.max_lines:
            raise ValueError(
                f"Expansion for {key} must satisfy 0 <= n <= {corridor.max_lines}"
            )
        normalized[key] = count

    branches: list[PhysicalLine] = []
    existing_counts: dict[tuple[int, int], int] = {}
    for line in network.existing_lines:
        key = (line.from_bus, line.to_bus)
        existing_counts[key] = existing_counts.get(key, 0) + 1
        branches.append(
            PhysicalLine(
                line.from_bus,
                line.to_bus,
                line.x_pu,
                line.susceptance_pu,
                line.capacity_mw,
                "existing",
                key,
                existing_counts[key],
            )
        )
    for corridor in network.candidate_corridors:
        key = (corridor.from_bus, corridor.to_bus)
        for index in range(normalized.get(key, 0)):
            branches.append(
                PhysicalLine(
                    corridor.from_bus,
                    corridor.to_bus,
                    corridor.x_pu,
                    corridor.susceptance_pu,
                    corridor.capacity_mw,
                    "new",
                    key,
                    index + 1,
                )
            )
    return ExpandedNetwork(
        network.bus_ids,
        network.base_mva,
        network.reference_bus,
        network.buses,
        tuple(branches),
        normalized,
    )


def expansion_cost(
    network: NetworkData, expansion: Mapping[tuple[int, int], int] | None = None
) -> float:
    """Return the investment cost of new circuits only."""

    expanded = expand_network(network, expansion)
    return sum(
        expanded.expansion.get((corridor.from_bus, corridor.to_bus), 0) * corridor.cost
        for corridor in network.candidate_corridors
    )
