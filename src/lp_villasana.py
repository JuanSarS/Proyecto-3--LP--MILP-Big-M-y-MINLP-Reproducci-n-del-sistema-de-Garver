"""Interactive Villasana-Garver linear programming procedure."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Mapping

import numpy as np

from .data_loader import NetworkData
from .network import build_incidence, expand_network, expansion_cost

try:
    from scipy.optimize import linprog
except ImportError:  # pragma: no cover
    linprog = None


@dataclass(frozen=True)
class LPIterationResult:
    iteration: int
    objective: float
    angles: np.ndarray
    pd_plus: dict[tuple[int, int], float]
    pd_minus: dict[tuple[int, int], float]
    required_flow: dict[tuple[int, int], float]
    selected_corridor: tuple[int, int] | None
    expansion_before: dict[tuple[int, int], int]
    expansion_after: dict[tuple[int, int], int]


@dataclass(frozen=True)
class LPExpansionResult:
    status: str
    iterations: tuple[LPIterationResult, ...]
    final_expansion: dict[tuple[int, int], int]
    investment_cost: float
    final_power_flow: object
    runtime: float
    diagnostics: tuple[str, ...] = ()

    @property
    def converged(self) -> bool:
        return self.status == "CONVERGED"

    @property
    def is_feasible(self) -> bool:
        return self.converged and self.final_power_flow.is_feasible

    @property
    def fictitious_pd_pu(self) -> float:
        if not self.iterations:
            return float("inf")
        return sum(self.iterations[-1].required_flow.values())

    @property
    def expansion(self) -> dict[tuple[int, int], int]:
        return self.final_expansion


def _solve_iteration(
    network: NetworkData, expansion: Mapping[tuple[int, int], int]
) -> tuple[np.ndarray, dict[tuple[int, int], float], dict[tuple[int, int], float], float]:
    if linprog is None:
        raise RuntimeError("scipy is required for the interactive LP")
    physical = expand_network(network, expansion)
    buses = physical.bus_ids
    reference = buses.index(physical.reference_bus)
    reduced = [i for i in range(len(buses)) if i != reference]
    branches = physical.branches
    corridors = tuple(network.candidate_by_key)
    ntheta, nbranches = len(reduced), len(branches)
    plus_start = ntheta + nbranches
    minus_start = plus_start + len(corridors)
    nvar = minus_start + len(corridors)
    theta_pos = {bus_index: index for index, bus_index in enumerate(reduced)}
    branch_incidence = build_incidence(buses, branches)
    candidate_incidence = build_incidence(
        buses, [network.candidate_by_key[key] for key in corridors]
    )
    objective = np.zeros(nvar)
    for index, key in enumerate(corridors):
        objective[plus_start + index] = network.candidate_by_key[key].cost
        objective[minus_start + index] = network.candidate_by_key[key].cost
    equality: list[np.ndarray] = []
    rhs: list[float] = []
    for branch_index, branch in enumerate(branches):
        row = np.zeros(nvar)
        row[ntheta + branch_index] = 1.0
        if branch.from_bus != physical.reference_bus:
            row[theta_pos[buses.index(branch.from_bus)]] -= branch.susceptance_pu
        if branch.to_bus != physical.reference_bus:
            row[theta_pos[buses.index(branch.to_bus)]] += branch.susceptance_pu
        equality.append(row)
        rhs.append(0.0)
    injections = np.array([bus.net_injection_pu for bus in physical.buses])
    for bus_index in range(len(buses)):
        row = np.zeros(nvar)
        row[ntheta:ntheta + nbranches] = branch_incidence[:, bus_index]
        row[plus_start:plus_start + len(corridors)] = candidate_incidence[:, bus_index] * -1.0
        row[minus_start:] = candidate_incidence[:, bus_index]
        equality.append(row)
        rhs.append(float(injections[bus_index]))
    bounds = [(None, None)] * ntheta
    bounds += [
        (-branch.capacity_mw / physical.base_mva, branch.capacity_mw / physical.base_mva)
        for branch in branches
    ]
    bounds += [(0.0, None)] * (2 * len(corridors))
    result = linprog(objective, A_eq=equality, b_eq=rhs, bounds=bounds, method="highs")
    if not result.success:
        raise RuntimeError(f"LP iteration failed: {result.message}")
    angles = np.zeros(len(buses))
    angles[reduced] = result.x[:ntheta]
    pd_plus = {
        key: float(result.x[plus_start + index]) for index, key in enumerate(corridors)
    }
    pd_minus = {
        key: float(result.x[minus_start + index]) for index, key in enumerate(corridors)
    }
    return angles, pd_plus, pd_minus, float(result.fun)


def solve_lp_villasana(
    network: NetworkData,
    *,
    max_iterations: int = 100,
    fictitious_tolerance: float = 1e-8,
) -> LPExpansionResult:
    """Solve the fictitious-flow LP and install the largest required corridor."""

    if max_iterations < 0:
        raise ValueError("max_iterations must be non-negative")
    started = perf_counter()
    expansion = {key: 0 for key in network.candidate_by_key}
    records: list[LPIterationResult] = []
    diagnostics: list[str] = []
    for iteration in range(1, max_iterations + 2):
        before = dict(expansion)
        angles, plus, minus, objective = _solve_iteration(network, before)
        required = {key: plus[key] + minus[key] for key in before}
        active = {key: value for key, value in required.items() if value > fictitious_tolerance}
        if not active:
            records.append(LPIterationResult(
                iteration, objective, angles, plus, minus, required, None, before, before
            ))
            break
        eligible = [
            key for key in active
            if expansion[key] < network.candidate_by_key[key].max_lines
        ]
        if not eligible:
            diagnostics.append("Fictitious flow remains but all relevant corridors reached max_lines.")
            records.append(LPIterationResult(
                iteration, objective, angles, plus, minus, required, None, before, before
            ))
            break
        selected = max(eligible, key=lambda key: (required[key], key))
        after = dict(before)
        after[selected] += 1
        records.append(LPIterationResult(
            iteration, objective, angles, plus, minus, required, selected, before, after
        ))
        expansion = after
        if iteration > max_iterations:
            diagnostics.append("Maximum LP iterations reached.")
            break
    status = "CONVERGED" if records and not records[-1].selected_corridor else "INFEASIBLE"
    from .dc_power_flow import solve_dc_power_flow
    final_flow = solve_dc_power_flow(network, expansion)
    return LPExpansionResult(
        status, tuple(records), expansion, expansion_cost(network, expansion),
        final_flow,         perf_counter() - started, tuple(diagnostics)
    )


run_lp = solve_lp_villasana
