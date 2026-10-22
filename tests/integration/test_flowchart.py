from __future__ import annotations

import polars as pl
import pytest
from conftest import EmptyCohort
from matplotlib.figure import Figure

from corr_vars.core.change_tracker import ChangeTrackerPipeline
from corr_vars.plot.flowchart import profile_counts

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def _ids(n: int) -> pl.DataFrame:
    return pl.DataFrame({"id": range(n)})


@pytest.fixture
def tracker() -> ChangeTrackerPipeline:
    """1000 -> 600 (screen) -> 600 (removes nobody) -> 500 -> 200."""
    t = ChangeTrackerPipeline("id", _ids(1000), initial_description="All stays")
    t.add_step(_ids(600), "Included screen: metabolic acidosis")
    t.add_step(_ids(600), "Included adult patients")
    t.add_step(_ids(500), "Excluded SOFA score < 4")
    step = t.add_step(_ids(200), "Excluded 1234 rows due to renal replacement therapy")
    step.add_substep(_ids(300), "RRT before admission", "id")
    step.add_substep(_ids(450), "RRT after admission", "id")
    return t


def _texts(fig: Figure) -> list[str]:
    return [t.get_text() for t in fig.axes[0].texts]


def test_cascade_without_substeps() -> None:
    t = ChangeTrackerPipeline("id", _ids(10), initial_description="All stays")
    t.add_step(_ids(4), "Excluded odd")
    fig = t.create_flowchart()
    texts = _texts(fig)
    assert isinstance(fig, Figure)
    assert "All stays (n=10)" in texts
    assert "Excluded odd (n=4)" in texts
    assert "Dropped (n=6)" in texts


def test_cascade_with_substeps(tracker: ChangeTrackerPipeline) -> None:
    texts = _texts(tracker.create_flowchart(title="Flow"))
    assert "Flow" in texts
    assert "Dropped (n=300)" in texts
    assert "Dropped (n=0)" in texts
    # substeps list the number each criterion alone removes
    assert {"200", "50", "RRT before admission", "RRT after admission"} <= set(texts)


def test_profile_counts_sorted_and_footnoted(tracker: ChangeTrackerPipeline) -> None:
    counts = profile_counts(tracker, screen_step="screen")
    assert counts.screened == 600
    assert counts.included == 200
    assert counts.reasons == [
        ("renal replacement therapy", 300),
        ("SOFA score < 4", 100),
    ]
    assert counts.zero_reasons == ["not adult patients"]


def test_profile_sort_is_stable() -> None:
    t = ChangeTrackerPipeline("id", _ids(100))
    t.add_step(_ids(90), "Excluded first")
    t.add_step(_ids(80), "Excluded second")
    t.add_step(_ids(50), "Excluded third")
    counts = profile_counts(t)
    assert counts.screened == 100
    assert counts.reasons == [("third", 30), ("first", 10), ("second", 10)]


def test_profile_renders(tracker: ChangeTrackerPipeline) -> None:
    fig = tracker.create_flowchart(
        layout="profile",
        screen_step="screen",
        labels={"SOFA": "SOFA below 4"},
        thousands_sep="'",
    )
    texts = _texts(fig)
    assert "400 excluded" in texts
    assert "SOFA below 4" in texts
    assert "600 assessed for eligibility" in texts
    assert "200 included in the analysis" in texts
    assert any("Criteria that excluded no patients" in t for t in texts)


def test_profile_thousands_sep() -> None:
    t = ChangeTrackerPipeline("id", _ids(12000))
    t.add_step(_ids(2000), "Excluded many")
    texts = _texts(t.create_flowchart(layout="profile", thousands_sep="'"))
    assert "12'000 assessed for eligibility" in texts


def test_profile_raises_on_imbalance(tracker: ChangeTrackerPipeline) -> None:
    # A step that does not start where the previous one ended breaks the chain.
    tracker.steps[-1].before = tracker.steps[1].before
    with pytest.raises(ValueError, match="does not balance"):
        tracker.create_flowchart(layout="profile", screen_step="screen")


def test_profile_raises_when_step_adds() -> None:
    t = ChangeTrackerPipeline("id", _ids(10))
    t.add_step(_ids(5), "Excluded half")
    t.add_step(_ids(8), "Added back")
    with pytest.raises(ValueError, match="adds 3 observations"):
        t.create_flowchart(layout="profile")


def test_profile_raises_on_unknown_screen_step(
    tracker: ChangeTrackerPipeline,
) -> None:
    with pytest.raises(ValueError, match="matches no tracker step"):
        profile_counts(tracker, screen_step="nope")


def test_profile_arms(tracker: ChangeTrackerPipeline) -> None:
    obs = _ids(200).with_columns(treated=pl.col("id").lt(80))
    fig = tracker.create_flowchart(
        layout="profile",
        obs=obs,
        screen_step="screen",
        groupby="treated",
        group_labels={True: "Treated", False: "Control"},
        group_order=["Treated", "Control"],
        arm_text={"Treated": "{n} received treatment"},
    )
    texts = _texts(fig)
    assert "80 received treatment" in texts
    assert "120 Control" in texts


def test_profile_arms_must_sum_to_included(tracker: ChangeTrackerPipeline) -> None:
    obs = _ids(150).with_columns(treated=pl.col("id").lt(80))
    with pytest.raises(ValueError, match="Arm sizes sum to 150"):
        tracker.create_flowchart(
            layout="profile", obs=obs, screen_step="screen", groupby="treated"
        )


def test_profile_groupby_requires_obs(tracker: ChangeTrackerPipeline) -> None:
    with pytest.raises(ValueError, match="requires obs"):
        tracker.create_flowchart(layout="profile", groupby="treated")


def test_unknown_layout(tracker: ChangeTrackerPipeline) -> None:
    with pytest.raises(ValueError, match="Unknown layout"):
        tracker.create_flowchart(layout="sideways")  # type: ignore[arg-type]


@pytest.mark.parametrize("layout", ["cascade", "profile"])
@pytest.mark.parametrize("suffix", ["pdf", "svg", "png"])
def test_output_file(
    tracker: ChangeTrackerPipeline, tmp_path: Path, layout: str, suffix: str
) -> None:
    out = tmp_path / f"flowchart.{suffix}"
    tracker.create_flowchart(output_file=out, layout=layout)  # type: ignore[arg-type]
    head = out.read_bytes()[:64]
    expected = {"pdf": b"%PDF", "svg": b"<?xml", "png": b"\x89PNG"}[suffix]
    assert head.startswith(expected)


def test_cohort_figureone(tracker: ChangeTrackerPipeline) -> None:
    cohort = EmptyCohort(obs=_ids(200).rename({"id": "icu_stay_id"}))
    cohort._change_tracker = tracker
    assert isinstance(cohort.figureone, Figure)
    fig = cohort.to_figureone(layout="profile", screen_step="screen")
    assert "200 included in the analysis" in _texts(fig)
