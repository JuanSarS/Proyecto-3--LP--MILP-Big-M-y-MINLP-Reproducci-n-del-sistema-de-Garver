"""Generate the four technical figures used by the Garver TEP report.

The script reads the project's CSV inputs and model outputs.  It writes
vector PDF figures and 300-dpi PNG fallbacks under the project-level
``figures`` directory, next to the LaTeX report.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


def project_root() -> Path:
    """Return the directory containing ``Codigo`` and the final report."""

    return Path(__file__).resolve().parents[2]


ROOT = project_root()
CODE = ROOT / "Codigo"
DATA = CODE / "data"
RESULTS = CODE / "results"
FIGURES = CODE / "figures"

BUS_POSITIONS = {
    1: (0.8, 2.8),
    2: (3.0, 3.5),
    3: (5.5, 3.0),
    4: (2.7, 1.2),
    5: (5.1, 1.1),
    6: (7.8, 2.2),
}

EXISTING_OFFSETS = {
    (1, 2): 0.00,
    (1, 4): 0.00,
    (1, 5): 0.07,
    (2, 3): 0.00,
    (2, 4): -0.06,
    (3, 5): 0.00,
}

CANDIDATE_OFFSETS = {
    (6, 2): 0.10,
    (6, 4): -0.10,
    (6, 5): 0.12,
    (3, 5): 0.16,
}


def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })


def require_files(*paths: Path) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Faltan archivos requeridos. Ejecute primero `python run_all.py` "
            f"si faltan resultados: {', '.join(missing)}"
        )


def load_project_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = (DATA / "buses.csv", DATA / "existing_lines.csv", DATA / "candidate_corridors.csv")
    require_files(*paths)
    return tuple(pd.read_csv(path) for path in paths)  # type: ignore[return-value]


def save_figure(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _line_geometry(from_bus: int, to_bus: int, offset: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    start = np.asarray(BUS_POSITIONS[from_bus], dtype=float)
    end = np.asarray(BUS_POSITIONS[to_bus], dtype=float)
    vector = end - start
    normal = np.array([-vector[1], vector[0]], dtype=float)
    normal /= np.linalg.norm(normal)
    return start + normal * offset, end + normal * offset


def draw_bus(ax: plt.Axes, bus: int, buses: pd.DataFrame, *, show_generation: bool) -> None:
    x, y = BUS_POSITIONS[bus]
    future = bus == 6
    ax.scatter(
        [x], [y], s=285, facecolor="white" if not future else "#e6e6e6",
        edgecolor="black", linewidth=1.0, marker="o" if not future else "s",
        zorder=5,
    )
    ax.text(x, y, str(bus), ha="center", va="center", fontsize=9, weight="bold", zorder=6)
    row = buses.loc[buses["bus"] == bus].iloc[0]
    if show_generation:
        label = f"G {row.pg_mw:.0f} / D {row.pl_mw:.0f} MW"
        ax.text(x, y - 0.38, label, ha="center", va="top", fontsize=6.5)
    if future:
        ax.text(x, y + 0.43, "Futura", ha="center", va="bottom", fontsize=7, style="italic")


def draw_segment(ax: plt.Axes, from_bus: int, to_bus: int, *, style: str, color: str,
                 width: float, offset: float = 0.0, zorder: int = 2) -> None:
    start, end = _line_geometry(from_bus, to_bus, offset)
    ax.plot([start[0], end[0]], [start[1], end[1]], linestyle=style,
            color=color, linewidth=width, solid_capstyle="round", zorder=zorder)


def draw_label(ax: plt.Axes, from_bus: int, to_bus: int, text: str, *,
               offset: float, along: float = 0.5, fontsize: float = 6.2,
               color: str = "black") -> None:
    start, end = _line_geometry(from_bus, to_bus, offset)
    point = start * (1 - along) + end * along
    vector = end - start
    normal = np.array([-vector[1], vector[0]], dtype=float)
    normal /= np.linalg.norm(normal)
    point += normal * (0.12 if offset >= 0 else -0.12)
    ax.text(point[0], point[1], text, ha="center", va="center", fontsize=fontsize,
            color=color, bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.2),
            zorder=7)


def draw_network_base(ax: plt.Axes, buses: pd.DataFrame, existing: pd.DataFrame,
                      candidates: pd.DataFrame, *, show_generation: bool,
                      show_candidates: bool = True) -> None:
    for row in existing.itertuples(index=False):
        key = (int(row.from_bus), int(row.to_bus))
        draw_segment(ax, *key, style="-", color="#202020", width=1.35,
                     offset=EXISTING_OFFSETS.get(key, 0.0))
    if show_candidates:
        for row in candidates.itertuples(index=False):
            key = (int(row.from_bus), int(row.to_bus))
            draw_segment(ax, *key, style=(0, (4, 2)), color="#555555", width=1.0,
                         offset=CANDIDATE_OFFSETS.get(key, 0.12))
    for bus in buses["bus"].astype(int):
        draw_bus(ax, bus, buses, show_generation=show_generation)
    ax.set_xlim(0.15, 8.65)
    ax.set_ylim(0.45, 4.25)
    ax.set_aspect("equal")
    ax.axis("off")


def generate_garver_initial(buses: pd.DataFrame, existing: pd.DataFrame,
                            candidates: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.15, 3.65))
    draw_network_base(ax, buses, existing, candidates, show_generation=True)
    for row in existing.itertuples(index=False):
        key = (int(row.from_bus), int(row.to_bus))
        text = f"x={row.x_pu:.2f}\n{row.capacity_mw:.0f} MW"
        draw_label(ax, *key, text, offset=EXISTING_OFFSETS.get(key, 0.0), fontsize=5.8)
    candidate_label_positions = {
        (6, 2): (6.05, 3.55),
        (6, 4): (5.25, 1.65),
        (6, 5): (7.0, 1.25),
        (3, 5): (6.05, 2.45),
    }
    for row in candidates.itertuples(index=False):
        key = (int(row.from_bus), int(row.to_bus))
        text = (f"{key[0]}-{key[1]}\n$x$={row.x_pu:.2f}, "
                f"{row.capacity_mw:.0f} MW\n$c$={row.cost:.0f}, $n_{{max}}$={row.max_lines:d}")
        x, y = candidate_label_positions[key]
        ax.text(x, y, text, ha="center", va="center", fontsize=5.5, color="#333333",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.86, pad=1.2),
                zorder=7)
    legend = [
        Line2D([0], [0], color="#202020", lw=1.35, label="Línea existente"),
        Line2D([0], [0], color="#555555", lw=1.0, ls=(0, (4, 2)), label="Corredor candidato"),
        Line2D([0], [0], marker="o", color="black", markerfacecolor="white", lw=0, label="Barra"),
        Line2D([0], [0], marker="s", color="black", markerfacecolor="#e6e6e6", lw=0, label="Barra futura"),
    ]
    ax.legend(handles=legend, loc="lower left", ncol=4, frameon=False,
              bbox_to_anchor=(0.0, -0.03), handletextpad=0.4, columnspacing=1.0)
    save_figure(fig, "garver_initial")


def generate_lp_evolution() -> None:
    path = RESULTS / "lp_iterations.csv"
    require_files(path)
    frame = pd.read_csv(path)
    required = {"iteration", "pd_62", "pd_64", "pd_65", "pd_35", "selected_corridor"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"lp_iterations.csv no contiene columnas: {sorted(missing)}")
    fig, ax = plt.subplots(figsize=(3.45, 2.35))
    styles = [
        ("pd_62", "6-2", "o", "#222222"),
        ("pd_64", "6-4", "s", "#555555"),
        ("pd_65", "6-5", "^", "#777777"),
        ("pd_35", "3-5", "D", "#111111"),
    ]
    x = frame["iteration"].to_numpy()
    for column, label, marker, color in styles:
        ax.plot(x, frame[column], marker=marker, markersize=3.8, linewidth=1.05,
                color=color, markerfacecolor="white", markeredgewidth=0.8, label=label)
    for row in frame.itertuples(index=False):
        selected = str(row.selected_corridor)
        if selected and selected != "nan":
            values = [getattr(row, col) for col, _, _, _ in styles]
            maximum = max(values)
            ax.annotate(selected.strip("()").replace(", ", "-"), (row.iteration, maximum),
                        xytext=(0, 5), textcoords="offset points", ha="center", fontsize=5.5)
    ax.set_xlabel("Iteración")
    ax.set_ylabel(r"$P_D$ [pu]")
    ax.set_xticks(frame["iteration"].astype(int))
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#d0d0d0", linewidth=0.45)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False, ncol=2, handlelength=1.8,
              columnspacing=0.8)
    fig.tight_layout(pad=0.35)
    save_figure(fig, "lp_evolution")


def _box(ax: plt.Axes, xy: tuple[float, float], width: float, height: float,
         title: str, body: str, *, face: str = "white", edge: str = "#202020") -> None:
    x, y = xy
    patch = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.02,rounding_size=0.03",
                           facecolor=face, edgecolor=edge, linewidth=0.9)
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height - 0.16, title, ha="center", va="top",
            weight="bold", fontsize=8)
    ax.text(x + width / 2, y + height / 2 - 0.03, body, ha="center", va="center",
            fontsize=6.5, linespacing=1.35)


def _arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float],
           *, style: str = "-|>", color: str = "#303030", rad: float = 0.0) -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=10,
                                 linewidth=0.9, color=color,
                                 connectionstyle=f"arc3,rad={rad}"))


def generate_architecture() -> None:
    fig, ax = plt.subplots(figsize=(7.18, 3.55))
    ax.set_xlim(0, 10.7)
    ax.set_ylim(0, 5.2)
    ax.axis("off")
    _box(ax, (0.15, 1.85), 1.45, 1.55, "Datos de entrada",
         "buses.csv\nexisting_lines.csv\ncandidate_corridors.csv", face="#f5f5f5")
    _box(ax, (1.95, 2.0), 1.65, 1.3, "Carga / red",
         "data_loader.py\nnetwork.py\nBbus · islas · expansiones")
    _box(ax, (4.0, 3.2), 1.55, 1.15, "LP interactivo",
         "lp_villasana.py\nSciPy linprog", face="#fafafa")
    _box(ax, (4.0, 1.85), 1.55, 1.15, "MILP Big-M",
         "milp_bigm.py\nGurobi", face="#fafafa")
    _box(ax, (4.0, 0.5), 1.55, 1.15, "MINLP bilineal",
         "minlp.py\nGurobi / NonConvex=2", face="#fafafa")
    _box(ax, (6.05, 1.85), 1.7, 1.55, "Validación física",
         "Reconstrucción física\ndc_power_flow.py\nvalidation.py", face="#eeeeee")
    _box(ax, (8.25, 1.85), 1.8, 1.55, "Reportes",
         "reporting.py\nresults/*.csv", face="#f5f5f5")
    _arrow(ax, (1.6, 2.62), (1.95, 2.62))
    for y in (3.78, 2.42, 1.06):
        _arrow(ax, (3.62, 2.65), (4.0, y), rad=0.0)
    for y in (3.78, 2.42, 1.06):
        _arrow(ax, (5.55, y), (6.05, 2.62), rad=0.0)
    _arrow(ax, (7.75, 2.62), (8.25, 2.62))
    ax.text(5.9, 4.65, "Una base de datos común → tres formulaciones → una validación física común",
            ha="center", va="center", fontsize=8.5, weight="bold")
    save_figure(fig, "architecture")


def _flow_arrow(ax: plt.Axes, from_bus: int, to_bus: int, flow: float,
                *, offset: float, color: str, curvature: float = 0.0) -> None:
    actual_from, actual_to = (from_bus, to_bus) if flow >= 0 else (to_bus, from_bus)
    start, end = _line_geometry(actual_from, actual_to, offset)
    vector = end - start
    start = start + 0.18 * vector
    end = end - 0.18 * vector
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=9,
                                 linewidth=0.9, color=color,
                                 connectionstyle=f"arc3,rad={curvature}", zorder=4))


def generate_garver_final(buses: pd.DataFrame, existing: pd.DataFrame,
                          candidates: pd.DataFrame) -> None:
    comparison_path = RESULTS / "comparison.csv"
    flows_path = RESULTS / "milp_flows.csv"
    require_files(comparison_path, flows_path)
    comparison = pd.read_csv(comparison_path)
    flows = pd.read_csv(flows_path)
    if "MILP" in set(comparison["method"]):
        row = comparison.loc[comparison["method"] == "MILP"].iloc[0]
    else:
        row = comparison.iloc[0]
    expansion = {
        (int(c.from_bus), int(c.to_bus)): int(row[f"n_{int(c.from_bus)}{int(c.to_bus)}"])
        for c in candidates.itertuples(index=False)
    }
    fig, ax = plt.subplots(figsize=(7.15, 3.8))
    for line in existing.itertuples(index=False):
        key = (int(line.from_bus), int(line.to_bus))
        draw_segment(ax, *key, style="-", color="#202020", width=1.35,
                     offset=EXISTING_OFFSETS.get(key, 0.0))
    for key, count in expansion.items():
        if count:
            draw_segment(ax, *key, style=(0, (5, 2)), color="#303030", width=1.2,
                         offset=CANDIDATE_OFFSETS.get(key, 0.12), zorder=3)
    for bus in buses["bus"].astype(int):
        draw_bus(ax, bus, buses, show_generation=False)
        theta = float(row[f"theta_{bus}"])
        x, y = BUS_POSITIONS[bus]
        ax.text(x + 0.22, y + 0.22, rf"$\theta={theta:.3f}$", fontsize=6.3)
    flow_groups = flows.groupby(["kind", "corridor"], sort=False)
    for (kind, corridor), group in flow_groups:
        from_bus, to_bus = map(int, corridor.split("-"))
        total = float(group["flow_mw"].sum())
        if corridor == "3-5":
            continue
        if kind == "existing":
            offset = EXISTING_OFFSETS.get((from_bus, to_bus), 0.0)
            color = "#202020"
            text = f"{abs(total):.2f} MW"
            draw_label(ax, from_bus, to_bus, text, offset=offset, fontsize=6.0)
        else:
            offset = CANDIDATE_OFFSETS.get((from_bus, to_bus), 0.12)
            color = "#303030"
            count = len(group)
            text = f"{abs(total):.2f} MW\n{count} nuevas"
            draw_label(ax, from_bus, to_bus, text, offset=offset, along=0.6, fontsize=6.0)
        _flow_arrow(ax, from_bus, to_bus, total, offset=offset, color=color)
    draw_label(ax, 3, 5, f"{abs(float(flows.loc[flows.corridor == '3-5', 'flow_mw'].sum())):.2f} MW\n"
               "1 existente + 1 nueva", offset=0.16, along=0.58, fontsize=5.7)
    total_35 = float(flows.loc[flows.corridor == "3-5", "flow_mw"].sum())
    _flow_arrow(ax, 3, 5, total_35, offset=0.16, color="#303030")
    legend = [
        Line2D([0], [0], color="#202020", lw=1.35, label="Existente"),
        Line2D([0], [0], color="#303030", lw=1.2, ls=(0, (5, 2)), label="Nueva"),
        Line2D([0], [0], marker=">", color="#202020", lw=0, label="Sentido del flujo"),
    ]
    ax.legend(handles=legend, loc="lower left", ncol=3, frameon=False,
              bbox_to_anchor=(0, -0.03), handletextpad=0.4)
    ax.text(8.45, 0.62, f"Máx. utilización: {100 * flows.utilization.max():.2f}%",
            ha="center", fontsize=7, bbox=dict(facecolor="white", edgecolor="#555555", pad=3))
    ax.set_xlim(0.15, 8.65)
    ax.set_ylim(0.45, 4.25)
    ax.set_aspect("equal")
    ax.axis("off")
    save_figure(fig, "garver_final")


def main() -> int:
    configure_style()
    buses, existing, candidates = load_project_data()
    generate_garver_initial(buses, existing, candidates)
    generate_lp_evolution()
    generate_architecture()
    generate_garver_final(buses, existing, candidates)
    for name in ("garver_initial", "lp_evolution", "architecture", "garver_final"):
        print(f"generated: {FIGURES / (name + '.pdf')}")
        print(f"generated: {FIGURES / (name + '.png')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
