from pathlib import Path

from src.data_loader import load_network
from src.minlp import solve_minlp
from src.reporting import write_flows_csv, write_solution_csv


if __name__ == "__main__":
    network = load_network(Path(__file__).parent / "data")
    result = solve_minlp(network)
    results_dir = Path(__file__).parent / "results"
    write_solution_csv(results_dir / "minlp_solution.csv", result, network)
    write_flows_csv(results_dir / "minlp_flows.csv", result.power_flow)
    print("Status:", result.model.Status)
    print("Expansion:", result.expansion)
    print("Investment cost:", result.objective)
    print("Maximum loading:", result.power_flow.max_loading)
