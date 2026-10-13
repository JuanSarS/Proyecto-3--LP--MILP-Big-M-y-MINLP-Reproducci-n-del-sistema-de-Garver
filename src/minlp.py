"""Nonlinear transmission expansion formulation.

For a corridor with ``n`` new circuits, total flow is
``n * b * (theta_from - theta_to)`` and its thermal limit is ``n * capacity``.
Existing circuits are never included in ``n``.
"""

from __future__ import annotations

from .data_loader import NetworkData
from .dc_power_flow import solve_dc_power_flow
from .milp_bigm import ExpansionSolution, _gurobi


def build_model(network: NetworkData, *, angle_bound: float = 10.0):
    gp = _gurobi()
    model = gp.Model("transmission_expansion_minlp")
    theta = model.addVars(network.bus_ids, lb=-angle_bound, ub=angle_bound, name="theta")
    model.addConstr(theta[network.reference_bus] == 0.0, name="reference")
    count = {}
    total_flow = {}
    existing_flow = []
    for k, line in enumerate(network.existing_lines):
        p = model.addVar(lb=-line.capacity_mw, ub=line.capacity_mw, name=f"pe[{k}]")
        existing_flow.append(p)
        model.addConstr(
            p == network.base_mva * line.susceptance_pu
            * (theta[line.from_bus] - theta[line.to_bus]),
            name=f"oe[{k}]",
        )
    for c, corridor in enumerate(network.candidate_corridors):
        key = (corridor.from_bus, corridor.to_bus)
        n = model.addVar(vtype=gp.GRB.INTEGER, lb=0, ub=corridor.max_lines, name=f"n[{c}]")
        p = model.addVar(lb=-corridor.max_lines * corridor.capacity_mw,
                         ub=corridor.max_lines * corridor.capacity_mw, name=f"p[{c}]")
        count[key], total_flow[key] = n, p
        delta = theta[corridor.from_bus] - theta[corridor.to_bus]
        model.addConstr(p == network.base_mva * corridor.susceptance_pu * n * delta,
                        name=f"ohm[{c}]")
        model.addConstr(p <= corridor.capacity_mw * n, name=f"limit+[{c}]")
        model.addConstr(p >= -corridor.capacity_mw * n, name=f"limit-[{c}]")

    for bus in network.bus_ids:
        expr = gp.LinExpr()
        for p, line in zip(existing_flow, network.existing_lines):
            if line.from_bus == bus:
                expr += p
            elif line.to_bus == bus:
                expr -= p
        for key, p in total_flow.items():
            if key[0] == bus:
                expr += p
            elif key[1] == bus:
                expr -= p
        injection = next(b.net_injection_mw for b in network.buses if b.bus == bus)
        model.addConstr(expr == injection, name=f"balance[{bus}]")
    model.setObjective(gp.quicksum(
        corridor.cost * count[(corridor.from_bus, corridor.to_bus)]
        for corridor in network.candidate_corridors
    ), gp.GRB.MINIMIZE)
    model.Params.NonConvex = 2
    model._count = count
    return model, count


def solve_minlp(network: NetworkData, *, angle_bound: float = 10.0,
                time_limit: float | None = None) -> ExpansionSolution:
    gp = _gurobi()
    model, count = build_model(network, angle_bound=angle_bound)
    if time_limit is not None:
        model.Params.TimeLimit = time_limit
    try:
        model.optimize()
    except gp.GurobiError as exc:
        raise RuntimeError(f"Gurobi could not solve the MINLP: {exc}") from exc
    if model.SolCount == 0:
        raise RuntimeError(f"MINLP has no incumbent solution (status {model.Status})")
    expansion = {key: int(round(var.X)) for key, var in count.items()}
    power_flow = solve_dc_power_flow(network, expansion)
    if not power_flow.is_feasible:
        raise RuntimeError(f"MINLP expansion failed DC validation: {power_flow.diagnostics}")
    return ExpansionSolution(expansion, float(model.ObjVal), model.Status, model, power_flow)


optimize_minlp = solve_minlp
build_minlp_model = build_model
