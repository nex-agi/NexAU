# NorthAU Experiment Viewer

A Streamlit web application for viewing and analyzing experiment results from the NorthAU evaluation system.

## Features

- **Sessions Overview**: View all evaluation sessions with filtering and sorting
- **Experiment Comparison**: Compare configurations and their performance metrics
- **Interactive Analytics**: Visualize score distributions, metrics comparisons, and performance trade-offs
- **Detailed Results**: Drill down into individual experiments and item results
- **Configuration Inspection**: View system prompts, LLM configs, and tool descriptions

## Quick Start

1. Make sure you have run some experiments to populate the database:
   ```bash
   # Example: Run some evaluation experiments first
   uv run examples/your_experiment_script.py
   ```

2. Start the Streamlit app:
   ```bash
   uv run streamlit run streamlit_simple.py
   ```
   
   Note: If you encounter import issues, use `streamlit_simple.py` instead of `streamlit_app.py` - it has fewer dependencies and direct database access.

3. Open your browser to `http://localhost:8501`

## Navigation

The application has two main views:

### Sessions Overview
- Lists all evaluation sessions in the database
- Filter by status, mode, and minimum score
- Shows summary statistics across all sessions

### Session Details
- Detailed view of a specific session
- Three tabs:
  - **Overview**: Tabular view of all experiments
  - **Analytics**: Charts and visualizations
  - **Detailed Results**: Configuration details and item-level results

## Database

The application reads from the SQLite database located at `data/autotuning.db`. Make sure this file exists and contains experiment data.

## Requirements

- Python 3.12+
- Dependencies are automatically managed by `uv`
- Main packages: `streamlit`, `plotly`, `pandas`

## Troubleshooting

- If you see "No evaluation sessions found", run some experiments first
- If database connection fails, check that `data/autotuning.db` exists
- Use the "Clear Cache & Refresh" button in the sidebar if data seems stale