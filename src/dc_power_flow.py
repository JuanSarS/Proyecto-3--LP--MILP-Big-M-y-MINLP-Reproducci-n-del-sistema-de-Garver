"""Numpy-based DC power-flow solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .data_loader import NetworkData
from .network import (
    ExpandedNetwork,
    PhysicalLine,
    build_bbus,
    build_incidence,
    connected_components,
    expand_network,
)


@dataclass(frozen=True)
class BranchFlow:
    """Flow and thermal information for one physical circuit."""

    branch: PhysicalLine
    flow_pu: float
    flow_mw: float
    utilization: float


@dataclass(frozen=True)
class DCPowerFlowResult:
    """Auditable result of a DC power-flow attempt."""

    angles: np.ndarray | None
    branch_flows: tuple[BranchFlow, ...]
    bus_residuals: np.ndarray | None
    islands: tuple[frozenset[int], ...]
    is_feasible: bool
    max_loading: float | None
    expanded_network: ExpandedNetwork
    bbus: np.ndarray
    diagnostics: tuple[str, ...]
    thermal_violations: tuple[BranchFlow, ...]

    @property
    def flows_by_corridor_mw(self) -> dict[tuple[int, int], float]:
        totals: dict[tuple[int, int], float] = {}
        for item in self.branch_flows:
            totals[item.branch.corridor] = totals.get(item.branch.corridor, 0.0) + item.flow_mw
        return totals


def _specified_injections(network: ExpandedNetwork) -> np.ndarray:
    return np.array([bus.net_injection_pu for bus in network.buses], dtype=float)


def check_thermal_limits(
    result: DCPowerFlowResult, tolerance: float = 1e-9
) -> bool:
    """Return whether every physical circuit respects its individual capacity."""

    return all(item.utilization <= 1.0 + tolerance for item in result.branch_flows)


def solve_dc_power_flow(
    network: NetworkData,
    expansion: Mapping[tuple[int, int], int] | None = None,
    *,
    balance_tolerance: float = 1e-8,
    singular_tolerance: float = 1e-12,
) -> DCPowerFlowResult:
    """Solve P_spec = Bbus @ theta without pseudoinverses."""

    physical = expand_network(network, expansion)
    branches = physical.branches
    bbus = build_bbus(physical.bus_ids, branches)
    islands = connected_components(physical.bus_ids, branches)
    injections = _specified_injections(physical)
    positions = {bus_id: index for index, bus_id in enumerate(physical.bus_ids)}
    diagnostics: list[str] = []
    for island in islands:
        imbalance = sum(injections[positions[bus_id]] for bus_id in island)
        if abs(imbalance) > balance_tolerance:
            diagnostics.append(
                f"Island {sorted(island)} has net injection {imbalance:.12g} pu; "
                "a stationary DC solution does not exist."
            )
    if len(islands) > 1:
        diagnostics.append(
            "The network has multiple islands; each balanced island would require "
            "its own reference angle."
        )
    if diagnostics:
        return DCPowerFlowResult(
            None, (), None, islands, False, None, physical, bbus,
            tuple(diagnostics), ()
        )

    reference_index = positions[physical.reference_bus]
    reduced_indices = [i for i in range(len(physical.bus_ids)) if i != reference_index]
    reduced = bbus[np.ix_(reduced_indices, reduced_indices)]
    if reduced.size and np.linalg.matrix_rank(reduced, tol=singular_tolerance) < len(reduced):
        diagnostics.append(
            "Reduced Bbus is singular; verify connectivity and independent reference "
            "angles for every island."
        )
        return DCPowerFlowResult(
            None, (), None, islands, False, None, physical, bbus,
            tuple(diagnostics), ()
        )

    angles = np.zeros(len(physical.bus_ids), dtype=float)
    if reduced_indices:
        angles[reduced_indices] = np.linalg.solve(reduced, injections[reduced_indices])
    incidence = build_incidence(physical.bus_ids, branches)
    susceptances = np.array([branch.susceptance_pu for branch in branches])
    flows_pu = susceptances * (incidence @ angles)
    branch_flows = tuple(
        BranchFlow(
            branch,
            float(flow),
            float(flow * physical.base_mva),
            abs(float(flow * physical.base_mva)) / branch.capacity_mw,
        )
        for branch, flow in zip(branches, flows_pu)
    )
    residuals = injections - incidence.T @ flows_pu
    violations = tuple(
        item for item in branch_flows if item.utilization > 1.0 + balance_tolerance
    )
    max_loading = max((item.utilization for item in branch_flows), default=0.0)
    return DCPowerFlowResult(
        angles,
        branch_flows,
        residuals,
        islands,
        not violations and np.max(np.abs(residuals), initial=0.0) <= balance_tolerance,
        max_loading,
        physical,
        bbus,
        tuple(diagnostics),
        violations,
    )
