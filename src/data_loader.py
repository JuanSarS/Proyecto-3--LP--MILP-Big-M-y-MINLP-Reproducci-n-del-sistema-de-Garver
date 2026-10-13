"""Load and validate the common Garver transmission-network dataset."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Bus:
    """Bus data in MW and per unit."""

    bus: int
    pg_mw: float
    pl_mw: float
    pg_pu: float
    pl_pu: float

    @property
    def net_injection_mw(self) -> float:
        """Return generation minus demand in MW."""

        return self.pg_mw - self.pl_mw

    @property
    def net_injection_pu(self) -> float:
        """Return generation minus demand in per unit."""

        return self.pg_pu - self.pl_pu


@dataclass(frozen=True)
class ExistingLine:
    """An existing transmission line."""

    from_bus: int
    to_bus: int
    x_pu: float
    susceptance_pu: float
    capacity_mw: float
    capacity_pu: float


@dataclass(frozen=True)
class CandidateCorridor:
    """A candidate corridor; max_lines counts new lines only."""

    from_bus: int
    to_bus: int
    x_pu: float
    susceptance_pu: float
    capacity_mw: float
    capacity_pu: float
    cost: float
    max_lines: int


@dataclass(frozen=True)
class NetworkData:
    """Validated network data shared by all optimization models."""

    base_mva: float
    buses: tuple[Bus, ...]
    existing_lines: tuple[ExistingLine, ...]
    candidate_corridors: tuple[CandidateCorridor, ...]
    reference_bus: int

    @property
    def bus_ids(self) -> tuple[int, ...]:
        return tuple(bus.bus for bus in self.buses)

    @property
    def total_generation_mw(self) -> float:
        return sum(bus.pg_mw for bus in self.buses)

    @property
    def total_demand_mw(self) -> float:
        return sum(bus.pl_mw for bus in self.buses)

    @property
    def candidate_by_key(self) -> dict[tuple[int, int], CandidateCorridor]:
        return {
            (corridor.from_bus, corridor.to_bus): corridor
            for corridor in self.candidate_corridors
        }


def _read_csv(path: Path, required_columns: Iterable[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required data file does not exist: {path}")
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {path}")
        missing = set(required_columns) - set(reader.fieldnames)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        return list(reader)


def _parse_float(value: str, field: str, path: Path) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid numeric value for {field} in {path}: {value!r}") from error


def _parse_int(value: str, field: str, path: Path) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid integer value for {field} in {path}: {value!r}") from error
    if str(parsed) != str(value).strip():
        raise ValueError(f"{field} must be an integer in {path}: {value!r}")
    return parsed


def _validate_bus_ids(bus_ids: set[int], endpoints: Iterable[tuple[int, int]]) -> None:
    for from_bus, to_bus in endpoints:
        if from_bus not in bus_ids or to_bus not in bus_ids:
            raise ValueError(
                f"Line endpoint ({from_bus}, {to_bus}) is not present in buses.csv"
            )
        if from_bus == to_bus:
            raise ValueError(f"Self-loop is not allowed: ({from_bus}, {to_bus})")


def load_network(
    data_dir: str | Path,
    *,
    base_mva: float = 100.0,
    reference_bus: int = 5,
    balance_tolerance: float = 1e-9,
) -> NetworkData:
    """Load CSV data, convert MW quantities to pu, and validate invariants."""

    if base_mva <= 0:
        raise ValueError("base_mva must be positive")
    directory = Path(data_dir)
    buses_path = directory / "buses.csv"
    existing_path = directory / "existing_lines.csv"
    candidates_path = directory / "candidate_corridors.csv"

    bus_rows = _read_csv(buses_path, ("bus", "pg_mw", "pl_mw"))
    line_rows = _read_csv(existing_path, ("from_bus", "to_bus", "x_pu", "capacity_mw"))
    corridor_rows = _read_csv(
        candidates_path,
        ("from_bus", "to_bus", "x_pu", "capacity_mw", "cost", "max_lines"),
    )

    buses: list[Bus] = []
    seen_buses: set[int] = set()
    for row in bus_rows:
        bus = _parse_int(row["bus"], "bus", buses_path)
        if bus in seen_buses:
            raise ValueError(f"Duplicate bus identifier: {bus}")
        seen_buses.add(bus)
        pg_mw = _parse_float(row["pg_mw"], "pg_mw", buses_path)
        pl_mw = _parse_float(row["pl_mw"], "pl_mw", buses_path)
        if pg_mw < 0 or pl_mw < 0:
            raise ValueError(f"Generation and demand must be non-negative at bus {bus}")
        buses.append(Bus(bus, pg_mw, pl_mw, pg_mw / base_mva, pl_mw / base_mva))

    if reference_bus not in seen_buses:
        raise ValueError(f"Reference bus {reference_bus} is not present in buses.csv")

    existing_lines: list[ExistingLine] = []
    for row in line_rows:
        from_bus = _parse_int(row["from_bus"], "from_bus", existing_path)
        to_bus = _parse_int(row["to_bus"], "to_bus", existing_path)
        x_pu = _parse_float(row["x_pu"], "x_pu", existing_path)
        capacity_mw = _parse_float(row["capacity_mw"], "capacity_mw", existing_path)
        if x_pu <= 0 or capacity_mw <= 0:
            raise ValueError("Existing-line reactances and capacities must be positive")
        existing_lines.append(
            ExistingLine(
                from_bus, to_bus, x_pu, 1.0 / x_pu, capacity_mw, capacity_mw / base_mva
            )
        )

    candidate_corridors: list[CandidateCorridor] = []
    for row in corridor_rows:
        from_bus = _parse_int(row["from_bus"], "from_bus", candidates_path)
        to_bus = _parse_int(row["to_bus"], "to_bus", candidates_path)
        x_pu = _parse_float(row["x_pu"], "x_pu", candidates_path)
        capacity_mw = _parse_float(row["capacity_mw"], "capacity_mw", candidates_path)
        cost = _parse_float(row["cost"], "cost", candidates_path)
        max_lines = _parse_int(row["max_lines"], "max_lines", candidates_path)
        if x_pu <= 0 or capacity_mw <= 0 or cost <= 0:
            raise ValueError("Candidate reactances, capacities, and costs must be positive")
        if max_lines < 0:
            raise ValueError("Candidate max_lines must be non-negative")
        candidate_corridors.append(
            CandidateCorridor(
                from_bus,
                to_bus,
                x_pu,
                1.0 / x_pu,
                capacity_mw,
                capacity_mw / base_mva,
                cost,
                max_lines,
            )
        )

    _validate_bus_ids(
        seen_buses,
        [
            *((line.from_bus, line.to_bus) for line in existing_lines),
            *((corridor.from_bus, corridor.to_bus) for corridor in candidate_corridors),
        ],
    )
    if abs(sum(bus.net_injection_mw for bus in buses)) > balance_tolerance * base_mva:
        raise ValueError("Total generation and demand are not balanced")

    return NetworkData(
        base_mva,
        tuple(buses),
        tuple(existing_lines),
        tuple(candidate_corridors),
        reference_bus,
    )
