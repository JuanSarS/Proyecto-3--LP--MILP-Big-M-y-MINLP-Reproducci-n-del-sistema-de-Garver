"""CSV reporting for model results."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping

from .network import expansion_cost


CORRIDORS = ((6, 2), (6, 4), (6, 5), (3, 5))


def _write(path: Path, rows: Iterable[Mapping[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_solution_csv(path: str | Path, method_result: object, network: object) -> None:
    expansion = getattr(method_result, "final_expansion", None)
    if expansion is None:
        expansion = method_result.expansion
    raw_status = getattr(method_result, "status", "unknown")
    status = {2: "OPTIMAL", 9: "TIME_LIMIT"}.get(raw_status, raw_status)
    row = {"method": type(method_result).__name__,
           "status": status,
           "investment_cost": expansion_cost(network, expansion)}
    for from_bus, to_bus in CORRIDORS:
        row[f"n_{from_bus}{to_bus}"] = expansion.get((from_bus, to_bus), 0)
    flow = getattr(method_result, "final_power_flow", None)
    if flow is None:
        flow = getattr(method_result, "power_flow", None)
    if flow is not None and flow.angles is not None:
        for bus_id, angle in zip(network.bus_ids, flow.angles):
            row[f"theta_{bus_id}"] = float(angle)
        row["max_residual_pu"] = float(max(abs(flow.bus_residuals)))
        row["max_loading"] = flow.max_loading
    _write(Path(path), [row], list(row))


def write_flows_csv(path: str | Path, flow_result: object) -> None:
    rows = []
    for item in flow_result.branch_flows:
        rows.append({
            "kind": item.branch.kind,
            "from_bus": item.branch.from_bus,
            "to_bus": item.branch.to_bus,
            "corridor": f"{item.branch.corridor[0]}-{item.branch.corridor[1]}",
            "circuit_index": item.branch.circuit_index,
            "flow_mw": item.flow_mw,
            "capacity_mw": item.branch.capacity_mw,
            "utilization": item.utilization,
        })
    _write(Path(path), rows, list(rows[0]) if rows else [
        "kind", "from_bus", "to_bus", "corridor", "circuit_index",
        "flow_mw", "capacity_mw", "utilization"
    ])


def write_lp_iterations(path: str | Path, iterations: Iterable[object], network: object) -> None:
    rows = []
    for item in iterations:
        row = {
            "iteration": item.iteration,
            "objective": item.objective,
            "selected_corridor": item.selected_corridor or "",
            "investment_cost": expansion_cost(network, item.expansion_after),
        }
        for from_bus, to_bus in CORRIDORS:
            row[f"pd_{from_bus}{to_bus}"] = item.required_flow.get((from_bus, to_bus), 0.0)
            row[f"n_{from_bus}{to_bus}"] = item.expansion_after.get((from_bus, to_bus), 0)
        rows.append(row)
    fields = list(rows[0]) if rows else ["iteration", "objective"]
    _write(Path(path), rows, fields)


def write_comparison(path: str | Path, rows: Iterable[Mapping[str, object]]) -> None:
    rows = list(rows)
    fields = list(rows[0]) if rows else ["method", "status"]
    _write(Path(path), rows, fields)
