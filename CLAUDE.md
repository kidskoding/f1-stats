# CLAUDE.md

Project context for Claude Code sessions.

## Project Overview

F1 statistics and visualization tool that fetches Formula 1 driver data (2015-2024) from the Jolpica F1 API and generates charts. Also includes classification metric visualizations (confusion matrices, precision/recall/F1 charts) as a secondary feature.

## Tech Stack

- **Python 3.13+** (managed via `uv` with `pyproject.toml`)
- **pandas** for data manipulation
- **matplotlib / seaborn** for visualizations
- **scikit-learn** for classification metrics
- **requests** for API calls

## Project Structure

```
main.py                 # Entry point — runs ML metrics + F1 visualizations
f1_data.py              # Data fetching from Jolpica API with local JSON caching
f1_metrics.py           # Classification metric calculations (F1, precision, recall)
f1_visualizations.py    # 15 F1 driver performance charts
visualizations.py       # Classification metric charts (confusion matrix, bar charts)
charts/                 # Output directory for generated PNG charts
f1_cache/               # Cached API responses (JSON files, gitignored)
pyproject.toml          # Project config and dependencies
```

## Key Patterns

- **API caching**: All Jolpica API responses are cached as JSON in `f1_cache/`. Delete the cache directory to re-fetch.
- **Pagination**: Race results and qualifying data are paginated (100 per page) and automatically merged across pages.
- **Rate limiting**: 1-second delay between API requests, exponential backoff on 429 responses.
- **Matplotlib backend**: Uses `Agg` (non-interactive) backend. All charts saved as PNGs at 150 DPI to `charts/`.
- **Chart output**: Charts are saved to `charts/` directory, not displayed interactively.

## Running

```bash
uv run python main.py
```

This will:
1. Print classification metrics reports (spam detection + sentiment analysis)
2. Generate classification visualization PNGs
3. Fetch F1 data from the API (cached after first run)
4. Generate 15 F1 visualization PNGs

## Data Sources

- **F1 data**: [Jolpica F1 API](https://api.jolpi.ca/ergast/f1) (Ergast-compatible), seasons 2015-2024
- **Classification data**: Hardcoded example datasets in `main.py` (spam detection binary, sentiment multi-class)

## Adding New Visualizations

1. Add a new `plot_*` function in `f1_visualizations.py` (F1 charts) or `visualizations.py` (ML charts)
2. Each plot function should accept DataFrames and return a `plt.Figure`
3. Register the chart in the `charts` list inside `generate_all_f1_visualizations()`
4. Follow the existing pattern: `(filename, function, [args])`

## Conventions

- All plot functions return `plt.Figure` objects (caller handles saving/closing)
- Driver filtering: Use top-N by total points or drivers with 3+ seasons for readability
- Color palette: Use `DRIVER_COLORS` list for consistent driver coloring across charts
- DNF detection: The `_is_dnf()` function in `f1_data.py` determines retirements vs finishes
