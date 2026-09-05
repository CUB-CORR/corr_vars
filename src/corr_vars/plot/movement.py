from __future__ import annotations

import matplotlib.pyplot as plt
import polars as pl

from corr_vars.definitions import ObsLevel

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from corr_vars.core.cohort import Cohort


def plot_movements(cohort: Cohort, case_id: str) -> Figure:
    if cohort.obs_level != ObsLevel.ICU_STAY:
        raise ValueError("Only cohorts with obs level 'ICU_STAY' are supported.")

    # Sort the group by icu_admission
    gr = cohort._obs.select(
        "case_id", "icu_admission", "icu_discharge", "icu_id"
    ).filter(pl.col("case_id").eq(case_id))

    if gr["icu_id"].dtype == pl.List(pl.Utf8):
        gr = gr.explode("icu_id")

    gr = gr.sort("icu_admission")

    # Create a figure and axis
    y_size = len(gr["icu_id"].unique()) * 2
    fig, ax = plt.subplots(figsize=(15, y_size))

    # Get unique icu_ids and assign them y-values
    icu_ids = gr["icu_id"].unique()
    y_positions = range(len(icu_ids))
    icu_id_to_y = dict(zip(icu_ids, y_positions))

    # Plot horizontal lines for each stay
    for idx, stay in enumerate(gr.iter_rows(named=True)):
        y = icu_id_to_y[stay["icu_id"]]
        ax.plot([stay["icu_admission"], stay["icu_discharge"]], [y, y], linewidth=4)

        # Calculate the midpoint for the label
        midpoint = (
            stay["icu_admission"] + (stay["icu_discharge"] - stay["icu_admission"]) / 2
        )

        # Add the label with row number
        ax.text(
            midpoint,
            y,
            f"{idx + 1}",
            horizontalalignment="center",
            verticalalignment="center",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7},
        )

    # Set y-ticks and labels
    ax.set_yticks(y_positions)
    ax.set_yticklabels(icu_ids)

    # Set labels and title
    ax.set_xlabel("Time")
    ax.set_ylabel("ICU ID")
    ax.set_title(f"ICU Stays Over Time for Case ID: {case_id}")

    # Format x-axis as dates
    plt.gcf().autofmt_xdate()

    # Adjust layout and display the plot
    plt.tight_layout()
    return fig
