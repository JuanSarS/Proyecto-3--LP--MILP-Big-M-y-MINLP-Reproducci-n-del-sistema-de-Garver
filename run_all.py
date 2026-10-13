"""Run LP, MILP and MINLP and write a consolidated comparison report."""

from pathlib import Path

from src.data_loader import load_network
from src.lp_villasana import solve_lp_villasana
from src.milp_bigm import solve_milp
from src.minlp import solve_minlp
from src.reporting import write_comparison
from src.validation import compare_expansions


def main() -> int:
    root = Path(__file__).parent
    network = load_network(root / "data")
    print("=" * 60)
    print("TRANSMISSION EXPANSION PLANNING - GARVER 6 BUS")
    print("=" * 60)
    lp = solve_lp_villasana(network)
    milp = solve_milp(network)
    minlp = solve_minlp(network)
    results = {"LP": lp.final_expansion, "MILP": milp.expansion, "MINLP": minlp.expansion}
    comparison = compare_expansions(network, results)
    rows = []
    for method, item in comparison.items():
        expansion = item["expansion"]
        angles = item["angles"]
        row = {
            "method": method,
            "status": "CONVERGED" if method == "LP" else "OPTIMAL",
            "objective_model": lp.iterations[-1].objective if method == "LP" else (
                milp.objective if method == "MILP" else minlp.objective
            ),
            "investment_cost": item["investment_cost"],
            "max_residual_pu": item["max_residual_pu"],
            "max_loading": item["max_loading"],
        }
        for key in ((6, 2), (6, 4), (6, 5), (3, 5)):
            row[f"n_{key[0]}{key[1]}"] = expansion.get(key, 0)
        for index, angle in enumerate(angles, start=1):
            row[f"theta_{index}"] = angle
        rows.append(row)
        print(f"{method}: {expansion}, cost={item['investment_cost']}, validation=OK")
    write_comparison(root / "results" / "comparison.csv", rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
