"""Cross-validation helpers shared by optimization models and scripts."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from .data_loader import NetworkData
from .dc_power_flow import DCPowerFlowResult, solve_dc_power_flow
from .network import expansion_cost


def validate_expansion(network: NetworkData, expansion: Mapping[tuple[int, int], int]) -> None:
    """Raise ValueError when an expansion violates corridor bounds."""

    candidate_by_key = network.candidate_by_key
    for key, value in expansion.items():
        if key not in candidate_by_key:
            raise ValueError(f"Unknown expansion corridor: {key}")
        if isinstance(value, bool) or int(value) != value:
            raise ValueError(f"Expansion count must be an integer: {key}={value}")
        if not 0 <= value <= candidate_by_key[key].max_lines:
            raise ValueError(f"Expansion count out of range: {key}={value}")


def validate_physical_solution(
    network: NetworkData,
    expansion: Mapping[tuple[int, int], int],
    *,
    tolerance: float = 1e-8,
) -> DCPowerFlowResult:
    """Solve and validate a proposed expansion independently of an optimizer."""

    validate_expansion(network, expansion)
    result = solve_dc_power_flow(network, expansion)
    if not result.is_feasible:
        raise ValueError(f"Expansion is not physically feasible: {result.diagnostics}")
    if result.bus_residuals is None or np.max(np.abs(result.bus_residuals)) > tolerance:
        raise ValueError("Physical solution has excessive nodal residuals")
    if result.thermal_violations:
        raise ValueError("Physical solution violates thermal limits")
    return result


def compare_expansions(
    network: NetworkData,
    expansions: Mapping[str, Mapping[tuple[int, int], int]],
) -> dict[str, dict[str, object]]:
    """Return costs and physical validation facts for several model results."""

    comparison: dict[str, dict[str, object]] = {}
    for method, expansion in expansions.items():
        flow = validate_physical_solution(network, expansion)
        comparison[method] = {
            "expansion": dict(expansion),
            "investment_cost": expansion_cost(network, expansion),
            "angles": flow.angles.copy() if flow.angles is not None else None,
            "max_residual_pu": float(np.max(np.abs(flow.bus_residuals))),
            "max_loading": flow.max_loading,
        }
    return comparison
