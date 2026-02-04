"""
F1 Driver Data Visualizations.

Generates 15 charts covering driver performance across the 2015-2024 seasons.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

CHARTS_DIR = os.path.join(os.path.dirname(__file__), "charts")
DPI = 150

DRIVER_COLORS = [
    "#E91E63", "#2196F3", "#FF9800", "#4CAF50", "#9C27B0",
    "#00BCD4", "#F44336", "#795548", "#3F51B5", "#8BC34A",
    "#FFEB3B", "#607D8B", "#FF5722", "#009688", "#673AB7",
]


# ── Chart 1: Championship Points by Season ──────────────────────────────────

def plot_championship_points_by_season(standings_df: pd.DataFrame) -> plt.Figure:
    """Line chart of championship points per season for top drivers."""
    # Drivers who finished top 5 in any season
    top_ids = standings_df[standings_df["position"] <= 5]["driver_id"].unique()
    df = standings_df[standings_df["driver_id"].isin(top_ids)]

    fig, ax = plt.subplots(figsize=(14, 8))
    for i, (driver_id, grp) in enumerate(df.groupby("driver_id")):
        grp = grp.sort_values("season")
        name = grp["driver_name"].iloc[0]
        ax.plot(grp["season"], grp["points"], "o-", label=name,
                color=DRIVER_COLORS[i % len(DRIVER_COLORS)], linewidth=2, markersize=5)

    ax.set_title("Championship Points by Season (2015-2024)", fontsize=16, fontweight="bold")
    ax.set_xlabel("Season", fontsize=12)
    ax.set_ylabel("Points", fontsize=12)
    ax.set_xticks(range(2015, 2025))
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    return fig


# ── Chart 2: Total Wins ─────────────────────────────────────────────────────

def plot_total_wins_bar(standings_df: pd.DataFrame) -> plt.Figure:
    """Horizontal bar chart of total wins by driver."""
    wins = standings_df.groupby("driver_name")["wins"].sum().sort_values()
    wins = wins[wins > 0].tail(15)

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(wins.index, wins.values, color="#2196F3")
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{int(width)}", va="center", fontsize=10)

    ax.set_title("Total Race Wins (2015-2024)", fontsize=16, fontweight="bold")
    ax.set_xlabel("Wins", fontsize=12)
    ax.grid(axis="x", alpha=0.3)
    return fig


# ── Chart 3: Podium Breakdown ───────────────────────────────────────────────

def plot_podium_breakdown(results_df: pd.DataFrame) -> plt.Figure:
    """Stacked horizontal bar chart of podium finishes (1st, 2nd, 3rd)."""
    podiums = results_df[results_df["position"].isin([1, 2, 3])].copy()
    pivot = podiums.pivot_table(index="driver_name", columns="position",
                                aggfunc="size", fill_value=0)
    pivot.columns = ["1st", "2nd", "3rd"]
    pivot["total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("total").tail(15).drop(columns="total")

    fig, ax = plt.subplots(figsize=(12, 8))
    pivot.plot.barh(stacked=True, ax=ax,
                    color=["#FFD700", "#C0C0C0", "#CD7F32"])
    ax.set_title("Podium Breakdown (2015-2024)", fontsize=16, fontweight="bold")
    ax.set_xlabel("Podium Finishes", fontsize=12)
    ax.legend(title="Position")
    ax.grid(axis="x", alpha=0.3)
    return fig


# ── Chart 4: Points Heatmap ─────────────────────────────────────────────────

def plot_points_heatmap(standings_df: pd.DataFrame) -> plt.Figure:
    """Heatmap of points scored per driver per season."""
    # Only drivers with 3+ seasons
    season_counts = standings_df.groupby("driver_id")["season"].nunique()
    regulars = season_counts[season_counts >= 3].index
    df = standings_df[standings_df["driver_id"].isin(regulars)]

    pivot = df.pivot_table(index="driver_name", columns="season",
                           values="points", fill_value=0)
    # Sort by total points
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax,
                linewidths=0.5, cbar_kws={"label": "Points"})
    ax.set_title("Points Heatmap by Driver & Season", fontsize=16, fontweight="bold")
    ax.set_ylabel("")
    ax.set_xlabel("Season", fontsize=12)
    return fig


# ── Chart 5: Win Percentage Donut ────────────────────────────────────────────

def plot_win_percentage_donut(results_df: pd.DataFrame) -> plt.Figure:
    """Donut chart showing share of wins among top drivers."""
    wins = results_df[results_df["position"] == 1]["driver_name"].value_counts()
    top = wins.head(8)
    others = wins.iloc[8:].sum()
    if others > 0:
        top["Others"] = others

    fig, ax = plt.subplots(figsize=(10, 10))
    colors = DRIVER_COLORS[:len(top) - (1 if "Others" in top.index else 0)]
    if "Others" in top.index:
        colors.append("#BDBDBD")

    wedges, texts, autotexts = ax.pie(
        top.values, labels=top.index, autopct="%1.1f%%",
        colors=colors, pctdistance=0.8, startangle=90)
    centre = plt.Circle((0, 0), 0.55, fc="white")
    ax.add_patch(centre)
    ax.text(0, 0, f"{wins.sum()}\nTotal Wins", ha="center", va="center",
            fontsize=14, fontweight="bold")
    ax.set_title("Win Share (2015-2024)", fontsize=16, fontweight="bold")
    return fig


# ── Chart 6: Pole Positions ─────────────────────────────────────────────────

def plot_pole_positions_bar(qualifying_df: pd.DataFrame) -> plt.Figure:
    """Horizontal bar chart of pole positions by driver."""
    poles = qualifying_df[qualifying_df["position"] == 1]["driver_name"].value_counts()
    poles = poles.sort_values().tail(15)

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(poles.index, poles.values, color="#FF9800")
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{int(width)}", va="center", fontsize=10)

    ax.set_title("Pole Positions (2015-2024)", fontsize=16, fontweight="bold")
    ax.set_xlabel("Poles", fontsize=12)
    ax.grid(axis="x", alpha=0.3)
    return fig


# ── Chart 7: Average Finishing Position ──────────────────────────────────────

def plot_avg_finishing_position(results_df: pd.DataFrame) -> plt.Figure:
    """Line chart of average finishing position per season (inverted Y)."""
    finished = results_df[~results_df["is_dnf"]].copy()
    # Drivers with 3+ seasons
    season_counts = finished.groupby("driver_id")["season"].nunique()
    regulars = season_counts[season_counts >= 3].index
    finished = finished[finished["driver_id"].isin(regulars)]

    # Top 10 by best career average
    career_avg = finished.groupby("driver_id")["position"].mean()
    top10 = career_avg.nsmallest(10).index

    fig, ax = plt.subplots(figsize=(14, 8))
    for i, driver_id in enumerate(top10):
        grp = finished[finished["driver_id"] == driver_id]
        season_avg = grp.groupby("season")["position"].mean()
        name = grp["driver_name"].iloc[0]
        ax.plot(season_avg.index, season_avg.values, "o-", label=name,
                color=DRIVER_COLORS[i % len(DRIVER_COLORS)], linewidth=2, markersize=5)

    ax.invert_yaxis()
    ax.set_title("Average Finishing Position by Season (lower is better)",
                 fontsize=16, fontweight="bold")
    ax.set_xlabel("Season", fontsize=12)
    ax.set_ylabel("Avg. Finishing Position", fontsize=12)
    ax.set_xticks(range(2015, 2025))
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    return fig


# ── Chart 8: Season Points Progression ──────────────────────────────────────

def plot_season_points_progression(results_df: pd.DataFrame, season: int = 2024) -> plt.Figure:
    """Cumulative points race-by-race for a given season."""
    df = results_df[results_df["season"] == season].copy()
    # Top 5 by final points
    total_pts = df.groupby("driver_id")["points"].sum().nlargest(5)
    top_ids = total_pts.index

    fig, ax = plt.subplots(figsize=(16, 8))
    for i, driver_id in enumerate(top_ids):
        grp = df[df["driver_id"] == driver_id].sort_values("round")
        cumulative = grp.groupby("round")["points"].sum().cumsum()
        name = grp["driver_name"].iloc[0]
        ax.plot(cumulative.index, cumulative.values, "o-", label=name,
                color=DRIVER_COLORS[i % len(DRIVER_COLORS)], linewidth=2, markersize=4)

    ax.set_title(f"{season} Championship Points Progression", fontsize=16, fontweight="bold")
    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel("Cumulative Points", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    return fig


# ── Chart 9: DNF Rate Comparison ────────────────────────────────────────────

def plot_dnf_rate_comparison(results_df: pd.DataFrame) -> plt.Figure:
    """Bar chart of DNF rates for drivers with 50+ starts."""
    stats = results_df.groupby("driver_name").agg(
        total=("is_dnf", "count"),
        dnfs=("is_dnf", "sum")
    )
    stats = stats[stats["total"] >= 50]
    stats["dnf_rate"] = stats["dnfs"] / stats["total"] * 100
    stats = stats.sort_values("dnf_rate")

    fig, ax = plt.subplots(figsize=(10, 10))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(stats)))
    ax.barh(stats.index, stats["dnf_rate"], color=colors)

    ax.set_title("DNF Rate (2015-2024, min 50 starts)", fontsize=16, fontweight="bold")
    ax.set_xlabel("DNF Rate (%)", fontsize=12)
    ax.grid(axis="x", alpha=0.3)
    return fig


# ── Chart 10: Teammate Head-to-Head ─────────────────────────────────────────

def plot_teammate_comparison(results_df: pd.DataFrame, standings_df: pd.DataFrame) -> plt.Figure:
    """Grouped bar chart of teammate head-to-head race finishes."""
    # Build teammate pairs from standings (same constructor, same season)
    pairs = []
    for (season, constructor), grp in standings_df.groupby(["season", "constructor"]):
        drivers = grp["driver_id"].tolist()
        if len(drivers) == 2:
            pairs.append({
                "season": season,
                "constructor": constructor,
                "d1": drivers[0],
                "d2": drivers[1],
                "d1_name": grp[grp["driver_id"] == drivers[0]]["driver_name"].iloc[0],
                "d2_name": grp[grp["driver_id"] == drivers[1]]["driver_name"].iloc[0],
            })

    # For each pair, count races where each driver finished ahead
    pair_results = []
    for p in pairs:
        season_results = results_df[results_df["season"] == p["season"]]
        d1_results = season_results[season_results["driver_id"] == p["d1"]]
        d2_results = season_results[season_results["driver_id"] == p["d2"]]

        merged = d1_results[["round", "position"]].merge(
            d2_results[["round", "position"]], on="round", suffixes=("_d1", "_d2"))
        # Only races where both finished
        both_finished = merged.dropna(subset=["position_d1", "position_d2"])
        d1_ahead = (both_finished["position_d1"] < both_finished["position_d2"]).sum()
        d2_ahead = (both_finished["position_d1"] > both_finished["position_d2"]).sum()

        pair_results.append({
            "label": f"{p['d1_name']} vs {p['d2_name']}\n({p['season']} {p['constructor']})",
            "d1_name": p["d1_name"],
            "d2_name": p["d2_name"],
            "d1_ahead": d1_ahead,
            "d2_ahead": d2_ahead,
            "total": d1_ahead + d2_ahead,
        })

    # Pick top 10 most competitive (closest ratio, with enough races)
    pair_results = [p for p in pair_results if p["total"] >= 10]
    pair_results.sort(key=lambda p: abs(p["d1_ahead"] - p["d2_ahead"]))
    pair_results = pair_results[:10]

    fig, ax = plt.subplots(figsize=(14, 10))
    y = np.arange(len(pair_results))
    height = 0.35

    labels = [p["label"] for p in pair_results]
    d1_vals = [p["d1_ahead"] for p in pair_results]
    d2_vals = [p["d2_ahead"] for p in pair_results]

    ax.barh(y - height / 2, d1_vals, height, label="Driver 1", color="#2196F3")
    ax.barh(y + height / 2, d2_vals, height, label="Driver 2", color="#FF9800")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title("Teammate Head-to-Head (Race Finishes)", fontsize=16, fontweight="bold")
    ax.set_xlabel("Races Finished Ahead", fontsize=12)
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    return fig


# ── Chart 11: Constructor Loyalty Timeline ──────────────────────────────────

def plot_constructor_loyalty(standings_df: pd.DataFrame) -> plt.Figure:
    """Gantt-style chart showing which team each driver raced for per season."""
    # Top 15 drivers by total points
    total_pts = standings_df.groupby("driver_id")["points"].sum().nlargest(15)
    top_ids = total_pts.index
    df = standings_df[standings_df["driver_id"].isin(top_ids)]

    constructors = df["constructor"].unique()
    cmap = {c: DRIVER_COLORS[i % len(DRIVER_COLORS)] for i, c in enumerate(constructors)}

    # Sort drivers by total points
    driver_order = df.groupby("driver_name")["points"].sum().sort_values().index.tolist()

    fig, ax = plt.subplots(figsize=(14, 8))
    for i, driver_name in enumerate(driver_order):
        grp = df[df["driver_name"] == driver_name]
        for _, row in grp.iterrows():
            ax.barh(i, 1, left=row["season"] - 0.5, height=0.7,
                    color=cmap[row["constructor"]], edgecolor="white", linewidth=0.5)

    ax.set_yticks(range(len(driver_order)))
    ax.set_yticklabels(driver_order)
    ax.set_xlabel("Season", fontsize=12)
    ax.set_title("Constructor Loyalty Timeline (2015-2024)", fontsize=16, fontweight="bold")
    ax.set_xticks(range(2015, 2025))

    # Legend for constructors
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=cmap[c], label=c) for c in constructors]
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    return fig


# ── Chart 12: Grid vs Finish Scatter ────────────────────────────────────────

def plot_grid_vs_finish(results_df: pd.DataFrame) -> plt.Figure:
    """Scatter plot of grid position vs finishing position."""
    finished = results_df[~results_df["is_dnf"] & (results_df["grid"] > 0)].copy()

    # Top 5 by total points for highlighting
    total_pts = finished.groupby("driver_id")["points"].sum().nlargest(5)
    top_ids = total_pts.index

    fig, ax = plt.subplots(figsize=(10, 10))

    # Others in grey
    others = finished[~finished["driver_id"].isin(top_ids)]
    ax.scatter(others["grid"], others["position"], alpha=0.1, color="#BDBDBD", s=15)

    # Top 5 highlighted
    for i, driver_id in enumerate(top_ids):
        grp = finished[finished["driver_id"] == driver_id]
        name = grp["driver_name"].iloc[0]
        ax.scatter(grp["grid"], grp["position"], alpha=0.5, s=25, label=name,
                   color=DRIVER_COLORS[i])

    # Reference line (grid == finish)
    ax.plot([0, 22], [0, 22], "--", color="black", alpha=0.3, label="No change")

    ax.set_title("Grid Position vs Finishing Position", fontsize=16, fontweight="bold")
    ax.set_xlabel("Grid Position", fontsize=12)
    ax.set_ylabel("Finishing Position", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 22)
    ax.grid(alpha=0.3)
    return fig


# ── Chart 13: Points Per Race Box Plot ──────────────────────────────────────

def plot_points_per_race_boxplot(results_df: pd.DataFrame) -> plt.Figure:
    """Box plot of points scored per race for top drivers."""
    total_pts = results_df.groupby("driver_id")["points"].sum().nlargest(12)
    top_ids = total_pts.index
    df = results_df[results_df["driver_id"].isin(top_ids)].copy()

    # Map driver_id to name
    id_to_name = df.groupby("driver_id")["driver_name"].first()
    # Sort by median points
    medians = df.groupby("driver_id")["points"].median().loc[top_ids].sort_values(ascending=False)
    ordered_names = [id_to_name[did] for did in medians.index]

    fig, ax = plt.subplots(figsize=(14, 8))
    data_to_plot = [df[df["driver_name"] == name]["points"].values for name in ordered_names]
    bp = ax.boxplot(data_to_plot, labels=ordered_names, vert=True, patch_artist=True)
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(DRIVER_COLORS[i % len(DRIVER_COLORS)])
        patch.set_alpha(0.7)

    ax.set_title("Points Per Race Distribution (2015-2024)", fontsize=16, fontweight="bold")
    ax.set_ylabel("Points", fontsize=12)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)
    return fig


# ── Chart 14: Wins by Season (Stacked) ──────────────────────────────────────

def plot_wins_by_season_stacked(results_df: pd.DataFrame) -> plt.Figure:
    """Stacked bar chart showing race wins per driver per season."""
    wins = results_df[results_df["position"] == 1].copy()
    pivot = wins.pivot_table(index="season", columns="driver_name",
                             aggfunc="size", fill_value=0)
    # Only drivers who won at least 1 race
    pivot = pivot.loc[:, pivot.sum() > 0]
    # Sort columns by total wins
    pivot = pivot[pivot.sum().sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(14, 8))
    pivot.plot.bar(stacked=True, ax=ax,
                   color=DRIVER_COLORS[:len(pivot.columns)])

    ax.set_title("Wins by Season (2015-2024)", fontsize=16, fontweight="bold")
    ax.set_xlabel("Season", fontsize=12)
    ax.set_ylabel("Number of Wins", fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    return fig


# ── Chart 15: Career Trajectory Radar ───────────────────────────────────────

def plot_career_trajectory_radar(standings_df: pd.DataFrame,
                                  results_df: pd.DataFrame,
                                  qualifying_df: pd.DataFrame) -> plt.Figure:
    """Radar chart comparing top 5 drivers across multiple metrics."""
    # Top 5 by total points
    total_pts = standings_df.groupby("driver_id")["points"].sum().nlargest(5)
    top_ids = total_pts.index

    categories = ["Total Points", "Wins", "Poles", "Podiums", "Consistency", "Avg Finish"]
    n = len(categories)

    # Compute raw metrics per driver
    metrics = {}
    for driver_id in top_ids:
        s = standings_df[standings_df["driver_id"] == driver_id]
        r = results_df[results_df["driver_id"] == driver_id]
        q = qualifying_df[qualifying_df["driver_id"] == driver_id]
        name = s["driver_name"].iloc[0]

        total_races = len(r)
        dnfs = r["is_dnf"].sum()
        finished = r[~r["is_dnf"]]

        metrics[driver_id] = {
            "name": name,
            "total_points": s["points"].sum(),
            "wins": r[r["position"] == 1].shape[0],
            "poles": q[q["position"] == 1].shape[0],
            "podiums": r[r["position"].isin([1, 2, 3])].shape[0],
            "consistency": 1 - (dnfs / total_races) if total_races > 0 else 0,
            "avg_finish": finished["position"].mean() if len(finished) > 0 else 20,
        }

    # Normalize to 0-1
    keys = ["total_points", "wins", "poles", "podiums", "consistency", "avg_finish"]
    all_vals = {k: [metrics[d][k] for d in top_ids] for k in keys}

    def normalize(vals, invert=False):
        mn, mx = min(vals), max(vals)
        if mx == mn:
            return [0.5] * len(vals)
        normed = [(v - mn) / (mx - mn) for v in vals]
        if invert:
            normed = [1 - v for v in normed]
        return normed

    normalized = {}
    for i, driver_id in enumerate(top_ids):
        normalized[driver_id] = [
            normalize(all_vals["total_points"])[i],
            normalize(all_vals["wins"])[i],
            normalize(all_vals["poles"])[i],
            normalize(all_vals["podiums"])[i],
            normalize(all_vals["consistency"])[i],
            normalize(all_vals["avg_finish"], invert=True)[i],  # lower is better
        ]

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"polar": True})
    for i, driver_id in enumerate(top_ids):
        values = normalized[driver_id] + normalized[driver_id][:1]
        name = metrics[driver_id]["name"]
        ax.plot(angles, values, "o-", label=name,
                color=DRIVER_COLORS[i], linewidth=2, markersize=5)
        ax.fill(angles, values, alpha=0.1, color=DRIVER_COLORS[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_title("Career Trajectory Comparison (Top 5 Drivers)",
                 fontsize=16, fontweight="bold", pad=20)
    ax.legend(bbox_to_anchor=(1.2, 1), loc="upper left", fontsize=10)
    return fig


# ── Orchestrator ─────────────────────────────────────────────────────────────

def generate_all_f1_visualizations(data: dict[str, pd.DataFrame]) -> None:
    """Generate all 15 F1 visualizations and save as PNGs."""
    os.makedirs(CHARTS_DIR, exist_ok=True)

    standings = data["standings"]
    results = data["results"]
    qualifying = data["qualifying"]

    charts = [
        ("f1_championship_points_by_season", plot_championship_points_by_season, [standings]),
        ("f1_total_wins", plot_total_wins_bar, [standings]),
        ("f1_podium_breakdown", plot_podium_breakdown, [results]),
        ("f1_points_heatmap", plot_points_heatmap, [standings]),
        ("f1_win_percentage_donut", plot_win_percentage_donut, [results]),
        ("f1_pole_positions", plot_pole_positions_bar, [qualifying]),
        ("f1_avg_finishing_position", plot_avg_finishing_position, [results]),
        ("f1_2024_points_progression", plot_season_points_progression, [results, 2024]),
        ("f1_dnf_rate", plot_dnf_rate_comparison, [results]),
        ("f1_teammate_comparison", plot_teammate_comparison, [results, standings]),
        ("f1_constructor_loyalty", plot_constructor_loyalty, [standings]),
        ("f1_grid_vs_finish", plot_grid_vs_finish, [results]),
        ("f1_points_per_race_boxplot", plot_points_per_race_boxplot, [results]),
        ("f1_wins_by_season_stacked", plot_wins_by_season_stacked, [results]),
        ("f1_career_trajectory_radar", plot_career_trajectory_radar, [standings, results, qualifying]),
    ]

    for name, func, args in charts:
        print(f"  Generating {name}...")
        try:
            fig = func(*args)
            fig.tight_layout()
            filepath = os.path.join(CHARTS_DIR, f"{name}.png")
            fig.savefig(filepath, dpi=DPI, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved {name}.png")
        except Exception as e:
            print(f"  ERROR generating {name}: {e}")
