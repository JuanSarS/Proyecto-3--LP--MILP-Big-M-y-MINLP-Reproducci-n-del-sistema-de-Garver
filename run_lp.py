"""Run the iterative LP expansion heuristic on the bundled data."""

from pathlib import Path

from src.data_loader import load_network
from src.reporting import write_flows_csv, write_lp_iterations, write_solution_csv
from src.lp_villasana import solve_lp_villasana


def main() -> int:
    network = load_network(Path(__file__).parent / "data")
    result = solve_lp_villasana(network)
    results_dir = Path(__file__).parent / "results"
    write_lp_iterations(results_dir / "lp_iterations.csv", result.iterations, network)
    write_solution_csv(results_dir / "lp_solution.csv", result, network)
    write_flows_csv(results_dir / "lp_flows.csv", result.final_power_flow)
    print(f"expansion={result.expansion}")
    print(f"investment_cost={result.investment_cost:g}")
    print(f"fictitious_pd_pu={result.fictitious_pd_pu:.6g}")
    print(f"converged={result.converged}; iterations={len(result.iterations)}")
    print(f"physical_validation={result.final_power_flow.is_feasible}")
    return 0 if result.converged else 1


if __name__ == "__main__":
    raise SystemExit(main())
