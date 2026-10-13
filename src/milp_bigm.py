"""Transmission expansion planning with a linear big-M formulation.

The model creates one binary variable per possible new circuit.  This is
intentional: a corridor's ``max_lines`` is a limit on *new* circuits and
existing circuits are only represented by the data in ``existing_lines``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .data_loader import NetworkData
from .dc_power_flow import DCPowerFlowResult, solve_dc_power_flow


@dataclass
class ExpansionSolution:
    """Solution returned by either planning formulation."""

    expansion: dict[tuple[int, int], int]
    objective: float
    status: int
    model: Any
    power_flow: DCPowerFlowResult

    @property
    def dc_result(self) -> DCPowerFlowResult:
        """Backward-friendly name for the independent DC validation."""

        return self.power_flow


def _gurobi():
    try:
        import gurobipy as gp
    except ImportError as exc:  # pragma: no cover - exercised without optional dep
        raise RuntimeError("gurobipy is required for transmission planning") from exc
    return gp


def build_model(
    network: NetworkData, *, angle_bound: float = 10.0, big_m: float = 1.0
):
    """Build and return the solver model and its variable dictionaries."""
    if big_m <= 0:
        raise ValueError("big_m must be positive")
    gp = _gurobi()
    model = gp.Model("transmission_expansion_milp")
    buses = network.bus_ids
    pos = {bus: i for i, bus in enumerate(buses)}
    theta = model.addVars(buses, lb=-angle_bound, ub=angle_bound, name="theta")
    model.addConstr(theta[network.reference_bus] == 0.0, name="reference")

    flow: dict[tuple[str, int], Any] = {}
    # Existing circuits are always present and have their DC equality.
    for k, line in enumerate(network.existing_lines):
        p = model.addVar(lb=-line.capacity_mw, ub=line.capacity_mw, name=f"p_e[{k}]")
        flow[("e", k)] = p
        model.addConstr(
            p == network.base_mva * line.susceptance_pu
            * (theta[line.from_bus] - theta[line.to_bus]),
            name=f"ohm_e[{k}]",
        )

    install: dict[tuple[int, int], list[Any]] = {}
    candidate_flow: dict[tuple[int, int], list[Any]] = {}
    for c, corridor in enumerate(network.candidate_corridors):
        key = (corridor.from_bus, corridor.to_bus)
        install[key], candidate_flow[key] = [], []
        # The slots are ordered only for deterministic model names.
        for slot in range(corridor.max_lines):
            y = model.addVar(vtype=gp.GRB.BINARY, name=f"y[{c},{slot}]")
            p = model.addVar(
                lb=-corridor.capacity_mw, ub=corridor.capacity_mw,
                name=f"p_n[{c},{slot}]",
            )
            install[key].append(y)
            candidate_flow[key].append(p)
            model.addConstr(p <= corridor.capacity_mw * y, name=f"limit+[{c},{slot}]")
            model.addConstr(p >= -corridor.capacity_mw * y, name=f"limit-[{c},{slot}]")
            delta = theta[corridor.from_bus] - theta[corridor.to_bus]
            model.addConstr(p - network.base_mva * corridor.susceptance_pu * delta
                            <= network.base_mva * big_m * corridor.susceptance_pu * (1 - y),
                            name=f"ohm+[{c},{slot}]")
            model.addConstr(p - network.base_mva * corridor.susceptance_pu * delta
                            >= -network.base_mva * big_m * corridor.susceptance_pu * (1 - y),
                            name=f"ohm-[{c},{slot}]")

    for bus in buses:
        expression = gp.LinExpr()
        for k, line in enumerate(network.existing_lines):
            sign = 1 if line.from_bus == bus else -1 if line.to_bus == bus else 0
            if sign:
                expression += sign * flow[("e", k)]
        for key, lines in candidate_flow.items():
            for p in lines:
                sign = 1 if key[0] == bus else -1 if key[1] == bus else 0
                if sign:
                    expression += sign * p
        injection = next(item.net_injection_mw for item in network.buses if item.bus == bus)
        model.addConstr(expression == injection, name=f"balance[{bus}]")

    model.setObjective(
        gp.quicksum(
            corridor.cost * y
            for corridor in network.candidate_corridors
            for key in [(corridor.from_bus, corridor.to_bus)]
            for y in install[key]
        ),
        gp.GRB.MINIMIZE,
    )
    model._install = install
    return model, theta, install


def solve_milp(network: NetworkData, *, angle_bound: float = 10.0,
               big_m: float = 1.0,
               time_limit: float | None = None) -> ExpansionSolution:
    """Optimize the expansion and independently validate it with DC power flow."""
    gp = _gurobi()
    model, _theta, install = build_model(network, angle_bound=angle_bound, big_m=big_m)
    if time_limit is not None:
        model.Params.TimeLimit = time_limit
    try:
        model.optimize()
    except gp.GurobiError as exc:
        raise RuntimeError(f"Gurobi could not solve the MILP: {exc}") from exc
    if model.SolCount == 0:
        raise RuntimeError(f"MILP has no incumbent solution (status {model.Status})")
    expansion = {
        key: sum(int(round(y.X)) for y in variables)
        for key, variables in install.items()
    }
    power_flow = solve_dc_power_flow(network, expansion)
    if not power_flow.is_feasible:
        raise RuntimeError(f"MILP expansion failed DC validation: {power_flow.diagnostics}")
    return ExpansionSolution(expansion, float(model.ObjVal), model.Status, model, power_flow)


optimize_milp = solve_milp
build_milp_model = build_model
solve_milp_bigm = solve_milp
