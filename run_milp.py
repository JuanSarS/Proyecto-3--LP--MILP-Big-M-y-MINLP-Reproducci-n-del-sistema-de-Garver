import os
from pathlib import Path
from dotenv import load_dotenv

from src.data_loader import load_network
from src.milp_bigm import solve_milp
from src.reporting import write_flows_csv, write_solution_csv


ENV_PATH = Path(__file__).resolve().parents[1] / "data" / "base_data.env"
load_dotenv(ENV_PATH)

BASE_MVA = os.getenv("BASE_MVA")
REFERENCE_BUS= os.getenv("REFERENCE_BUS")
kwargs = {}
if BASE_MVA and BASE_MVA.strip():
    kwargs["base_mva"] = float(BASE_MVA)

if REFERENCE_BUS and REFERENCE_BUS.strip():
    kwargs["reference_bus"] = int(REFERENCE_BUS)

if __name__ == "__main__":
    network = load_network(Path(__file__).parent / "data", **kwargs)
    result = solve_milp(network)
    results_dir = Path(__file__).parent / "results"
    write_solution_csv(results_dir / "milp_solution.csv", result, network)
    write_flows_csv(results_dir / "milp_flows.csv", result.power_flow)
    print("Status:", result.model.Status)
    print("Expansion:", result.expansion)
    print("Investment cost:", result.objective)
    print("Maximum loading:", result.power_flow.max_loading)
