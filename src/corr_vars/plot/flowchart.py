"""Cohort flowcharts drawn with matplotlib.

Two layouts read the same :class:`~corr_vars.core.change_tracker.ChangeTrackerPipeline`:

- ``cascade`` — every tracker step in sequence, each with a dashed side box holding
  the number of observations it removed (and its substeps, if any). The audit view.
- ``profile`` — a Lancet-style trial profile: screened → one box listing every
  exclusion reason → included → optional arms. The publication view.

Layout uses explicit coordinates: the axes fill the figure, one data unit is one
inch and y runs downward from the top of the page, so box heights computed from
the font size are exact. Text widths are measured with ``TextPath`` before any
figure exists, which lets every wrap point and the page height be fixed up front.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from functools import lru_cache

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.textpath import TextPath

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from os import PathLike

    import polars as pl
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from corr_vars.core.change_tracker import ChangeTrackerPipeline


# Scaffolding the change tracker puts around a criterion: the leading verb, the row
# count spliced in by Cohort ("Excluded 12 rows due to ...") and a trailing "(n=...)".
_INCLUDE_RE = re.compile(r"^\s*include[ds]?\s+", re.IGNORECASE)
_EXCLUDE_RE = re.compile(r"^\s*exclude[ds]?\s+", re.IGNORECASE)
_ROWS_RE = re.compile(r"\b\d[\d',]*\s+rows?\s+due\s+to\s+", re.IGNORECASE)
_TRAIL_N_RE = re.compile(r"\s*\(\s*n\s*=\s*[\d',]+\s*\)\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class FlowchartStyle:
    """Geometry, typography and number format of a flowchart.

    All lengths are in inches and double as drawing coordinates. Override single
    fields by passing them as keyword arguments to the flowchart functions, e.g.
    ``create_flowchart(thousands_sep="'", page_width=7.0)``.
    """

    page_width: float = 8.27  # A4 portrait; the page height is derived
    margin: float = 0.25  # left/right page margin
    top_margin: float = 0.30
    bottom_margin: float = 0.25

    spine_x: float = 2.45  # the vertical trunk
    main_box_width: float = 3.40
    side_x: float = 5.05  # left edge of the excluded/dropped boxes
    arm_box_width: float = 2.00
    arm_gap: float = 0.30

    pad_x: float = 0.10  # text inset inside a box
    pad_y: float = 0.09
    line_spacing: float = 1.24  # baseline-to-baseline, as a multiple of the font size
    count_col_width: float = 0.42  # minimum width of the right-aligned count column
    count_gap: float = 0.10  # count column -> reason text
    header_gap: float = 0.09  # side-box header -> first reason
    item_gap: float = 0.035  # leading between reasons

    side_top_offset: float = 0.30  # box above -> side box top
    drop_after_side: float = 0.40  # side box bottom -> next main box top
    min_drop: float = 0.70  # shortest main -> main drop when a side box is drawn
    split_gap: float = 0.60  # included box -> arms
    split_bar_frac: float = 0.50  # where in that gap the horizontal bar sits
    footnote_gap: float = 0.32
    title_gap: float = 0.20  # title -> first box

    font_family: str = "DejaVu Sans"
    font_size: float = 9.0
    footnote_size: float = 7.5
    title_size: float = 11.0
    box_linewidth: float = 0.8
    connector_linewidth: float = 0.8
    arrow_size: float = 8.0
    ink: str = "black"
    fill: str = "white"

    thousands_sep: str = ","
    dpi: int = 300  # raster output only

    def line_height(self, size: float | None = None) -> float:
        return (self.font_size if size is None else size) * self.line_spacing / 72.0

    def fmt(self, n: int) -> str:
        return f"{int(n):,}".replace(",", self.thousands_sep)


@dataclass
class Box:
    """A drawn rectangle, in page inches, y measured downward from the top."""

    x: float  # left edge
    y: float  # top edge
    w: float
    h: float

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def bottom(self) -> float:
        return self.y + self.h

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2


@dataclass
class ProfileCounts:
    """Every number the ``profile`` layout draws."""

    screened: int
    included: int
    reasons: list[tuple[str, int]]  # (label, n removed), descending
    zero_reasons: list[str]  # steps that removed nobody
    arms: list[tuple[str, int]]  # (label, n), in display order

    @property
    def excluded(self) -> int:
        return self.screened - self.included


# ---------------------------------------------------------------------------
# Text measurement
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1 << 14)
def text_width(text: str, size: float, family: str = "DejaVu Sans") -> float:
    """Rendered width of ``text`` in inches, measured with ``TextPath``."""
    if not text:
        return 0.0
    prop = FontProperties(family=family, size=size)
    return TextPath((0, 0), text, size=size, prop=prop).get_extents().width / 72.0


def wrap_text(
    text: str, width: float, size: float, family: str = "DejaVu Sans"
) -> list[str]:
    """Greedy word wrap to ``width`` inches.

    A single word wider than the line is kept whole rather than broken mid-word.
    Explicit newlines start a new line.
    """
    lines: list[str] = []
    for paragraph in text.split("\n"):
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            if text_width(candidate, size, family) <= width:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------
def _box_height(n_lines: int, style: FlowchartStyle, extra: float = 0.0) -> float:
    return 2 * style.pad_y + n_lines * style.line_height() + extra


def _new_figure(style: FlowchartStyle, page_height: float) -> tuple[Figure, Axes]:
    # Created through pyplot so the active backend (e.g. the notebook's inline
    # display hook) is resolved, then closed at once: pyplot does not show it
    # again, and a notebook displays the returned object exactly once.
    fig = plt.figure(figsize=(style.page_width, page_height), facecolor=style.fill)
    plt.close(fig)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, style.page_width)
    ax.set_ylim(page_height, 0)
    ax.set_axis_off()
    return fig, ax


def _draw_box(
    ax: Axes, box: Box, style: FlowchartStyle, *, dashed: bool = False
) -> None:
    ax.add_patch(
        Rectangle(
            (box.x, box.y),
            box.w,
            box.h,
            facecolor=style.fill,
            edgecolor=style.ink,
            linewidth=style.box_linewidth,
            linestyle="--" if dashed else "-",
            joinstyle="miter",
            zorder=2,
        )
    )


def _draw_lines(
    ax: Axes,
    lines: Sequence[str],
    x: float,
    top: float,
    style: FlowchartStyle,
    *,
    size: float | None = None,
    ha: str = "left",
    weight: str = "normal",
) -> float:
    """Draw ``lines`` on a line grid starting at ``top``; return the next free y."""
    size = style.font_size if size is None else size
    line_h = style.line_height(size)
    y = top
    for line in lines:
        ax.text(
            x,
            y + line_h / 2,
            line,
            ha=ha,
            va="center",
            fontsize=size,
            fontfamily=style.font_family,
            fontweight=weight,
            color=style.ink,
            zorder=3,
        )
        y += line_h
    return y


def _segment(
    ax: Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    style: FlowchartStyle,
) -> None:
    ax.add_line(
        Line2D(
            [start[0], end[0]],
            [start[1], end[1]],
            color=style.ink,
            linewidth=style.connector_linewidth,
            solid_capstyle="butt",
            zorder=1,
        )
    )


def _arrow(
    ax: Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    style: FlowchartStyle,
) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "-|>",
            "color": style.ink,
            "linewidth": style.connector_linewidth,
            "shrinkA": 0,
            "shrinkB": 0,
            "mutation_scale": style.arrow_size,
        },
        annotation_clip=False,
        zorder=1,
    )


def _split(
    ax: Axes,
    x: float,
    y_from: float,
    y_bar: float,
    targets: Sequence[tuple[float, float]],
    style: FlowchartStyle,
) -> None:
    """Vertical drop, a horizontal bar, then one arrowed drop per (centre_x, top)."""
    _segment(ax, (x, y_from), (x, y_bar), style)
    xs = [cx for cx, _ in targets]
    _segment(ax, (min(xs + [x]), y_bar), (max(xs + [x]), y_bar), style)
    for cx, top in targets:
        _arrow(ax, (cx, y_bar), (cx, top), style)


def _count_column_width(counts: Sequence[str], style: FlowchartStyle) -> float:
    measured = max(
        (text_width(c, style.font_size, style.font_family) for c in counts),
        default=0.0,
    )
    return max(style.count_col_width, measured)


def _wrap_items(
    items: Sequence[tuple[str, str]], text_w: float, style: FlowchartStyle
) -> list[tuple[str, list[str]]]:
    return [
        (count, wrap_text(label, text_w, style.font_size, style.font_family))
        for count, label in items
    ]


def _items_height(
    wrapped: Sequence[tuple[str, list[str]]], style: FlowchartStyle
) -> float:
    if not wrapped:
        return 0.0
    n_lines = sum(len(lines) for _, lines in wrapped)
    return n_lines * style.line_height() + (len(wrapped) - 1) * style.item_gap


def _draw_items(
    ax: Axes,
    wrapped: Sequence[tuple[str, list[str]]],
    count_right: float,
    text_x: float,
    top: float,
    style: FlowchartStyle,
) -> float:
    """Hanging-indent list: counts right-aligned in their own column, text beside."""
    y = top
    for i, (count, lines) in enumerate(wrapped):
        if i:
            y += style.item_gap
        _draw_lines(ax, [count], count_right, y, style, ha="right")
        y = _draw_lines(ax, lines, text_x, y, style)
    return y


def _title_height(title: str | None, style: FlowchartStyle) -> tuple[list[str], float]:
    if not title:
        return [], 0.0
    lines = wrap_text(
        title,
        style.page_width - 2 * style.margin,
        style.title_size,
        style.font_family,
    )
    return lines, len(lines) * style.line_height(style.title_size) + style.title_gap


def _draw_title(ax: Axes, lines: Sequence[str], style: FlowchartStyle) -> None:
    if lines:
        _draw_lines(
            ax,
            lines,
            style.page_width / 2,
            style.top_margin,
            style,
            size=style.title_size,
            ha="center",
            weight="bold",
        )


def _finish(
    fig: Figure, output_file: PathLike | str | None, style: FlowchartStyle
) -> Figure:
    if output_file is not None:
        # No bbox_inches="tight": the page size is the layout.
        fig.savefig(output_file, dpi=style.dpi, facecolor=style.fill)
    return fig


def _resolve_style(
    style: FlowchartStyle | None, overrides: Mapping[str, Any]
) -> FlowchartStyle:
    base = style or FlowchartStyle()
    return replace(base, **overrides) if overrides else base


# ---------------------------------------------------------------------------
# Cascade layout
# ---------------------------------------------------------------------------
def cascade_flowchart(
    tracker: ChangeTrackerPipeline,
    *,
    title: str | None = None,
    output_file: PathLike | str | None = None,
    step_text: str = "{description} (n={n})",
    dropped_text: str = "Dropped (n={n})",
    style: FlowchartStyle | None = None,
    **style_overrides: Any,
) -> Figure:
    """Draw every tracker step in sequence, each with a dashed box of what it removed.

    Args:
        tracker: The change tracker to draw.
        title: Optional title above the chart.
        output_file: If given, save the figure there; the format follows the extension.
        step_text: Template of a main box; ``{description}`` and ``{n}`` (remaining).
        dropped_text: Header of a dropped box; ``{n}`` (removed by the step).
        style: Base style; ``style_overrides`` replace single fields of it.

    Returns:
        matplotlib.figure.Figure: The flowchart.
    """
    st = _resolve_style(style, style_overrides)
    main_text_w = st.main_box_width - 2 * st.pad_x
    side_w = st.page_width - st.margin - st.side_x
    side_text_w = side_w - 2 * st.pad_x

    def main_lines(description: str, n: int) -> list[str]:
        text = step_text.format(description=description, n=st.fmt(n))
        return wrap_text(text, main_text_w, st.font_size, st.font_family)

    title_lines, title_h = _title_height(title, st)
    main_x = st.spine_x - st.main_box_width / 2

    lines = main_lines(tracker.initial_description, tracker.initial_snapshot.nuniques)
    boxes: list[tuple[Box, list[str]]] = [
        (
            Box(
                main_x,
                st.top_margin + title_h,
                st.main_box_width,
                _box_height(len(lines), st),
            ),
            lines,
        )
    ]
    sides: list[tuple[Box, list[str], list[tuple[str, list[str]]], float]] = []

    for step in tracker.steps:
        prev = boxes[-1][0]
        removed = max(0, step.before.nuniques - step.after.nuniques)
        header = wrap_text(
            dropped_text.format(n=st.fmt(removed)),
            side_text_w,
            st.font_size,
            st.font_family,
        )
        sub_counts = [
            st.fmt(max(0, sub.before.nuniques - sub.after.nuniques))
            for sub in step.substeps
        ]
        count_w = _count_column_width(sub_counts, st)
        wrapped = _wrap_items(
            [(c, sub.description) for c, sub in zip(sub_counts, step.substeps)],
            side_text_w - count_w - st.count_gap,
            st,
        )
        extra = (st.header_gap + _items_height(wrapped, st)) if wrapped else 0.0
        side = Box(
            st.side_x,
            prev.bottom + st.side_top_offset,
            side_w,
            _box_height(len(header), st, extra=extra),
        )
        sides.append((side, header, wrapped, count_w))

        lines = main_lines(step.description, step.after.nuniques)
        top = max(side.bottom + st.drop_after_side, prev.bottom + st.min_drop)
        boxes.append(
            (Box(main_x, top, st.main_box_width, _box_height(len(lines), st)), lines)
        )

    page_h = boxes[-1][0].bottom + st.bottom_margin
    fig, ax = _new_figure(st, page_h)
    _draw_title(ax, title_lines, st)

    for box, lines in boxes:
        _draw_box(ax, box, st)
        _draw_lines(ax, lines, box.x + st.pad_x, box.y + st.pad_y, st)
    for (prev, _), (nxt, _) in zip(boxes, boxes[1:]):
        _arrow(ax, (st.spine_x, prev.bottom), (st.spine_x, nxt.y), st)

    for side, header, wrapped, count_w in sides:
        _draw_box(ax, side, st, dashed=True)
        y = _draw_lines(ax, header, side.x + st.pad_x, side.y + st.pad_y, st)
        count_right = side.x + st.pad_x + count_w
        _draw_items(
            ax, wrapped, count_right, count_right + st.count_gap, y + st.header_gap, st
        )
        _arrow(ax, (st.spine_x, side.cy), (side.x, side.cy), st)

    return _finish(fig, output_file, st)


# ---------------------------------------------------------------------------
# Profile layout
# ---------------------------------------------------------------------------
def _label(description: str, labels: Mapping[str, str] | None) -> str:
    """Display text for a step: the first ``labels`` key contained in the
    description (case-insensitive), else the description stripped of tracker
    scaffolding. An inclusion criterion reads as "not ..." since the box lists
    who was removed."""
    for key, text in (labels or {}).items():
        if key.lower() in description.lower():
            return text
    text = _TRAIL_N_RE.sub("", description.strip())
    negate = bool(_INCLUDE_RE.match(text))
    text = _ROWS_RE.sub("", _EXCLUDE_RE.sub("", _INCLUDE_RE.sub("", text))).strip()
    if not text:
        return description.strip()
    # Lower-case the first letter unless it starts an acronym ("SOFA", "ICU").
    if len(text) == 1 or not text[1].isupper():
        text = text[:1].lower() + text[1:]
    return f"not {text}" if negate else text


def profile_counts(
    tracker: ChangeTrackerPipeline,
    obs: pl.DataFrame | None = None,
    *,
    screen_step: str | None = None,
    labels: Mapping[str, str] | None = None,
    groupby: str | None = None,
    group_labels: Mapping[Any, str] | None = None,
    group_order: Sequence[str] | None = None,
) -> ProfileCounts:
    """Reduce the tracker to the tiers of a trial profile and check they balance.

    Raises:
        ValueError: If ``screen_step`` matches no step, a step adds observations, the
            reasons do not sum to screened − included, ``groupby`` is given without
            ``obs`` or is not a column of it, or the arms do not sum to included.
    """
    steps = list(tracker.steps)
    if screen_step is None:
        screened = tracker.initial_snapshot.nuniques
        counted = steps
    else:
        start = next(
            (
                i
                for i, step in enumerate(steps)
                if screen_step.lower() in step.description.lower()
            ),
            None,
        )
        if start is None:
            raise ValueError(
                f"screen_step {screen_step!r} matches no tracker step. Steps: "
                f"{[step.description for step in steps]}"
            )
        screened = steps[start].after.nuniques
        counted = steps[start + 1 :]
    included = steps[-1].after.nuniques if steps else screened

    reasons: list[tuple[str, int]] = []
    zero_reasons: list[str] = []
    for step in counted:
        removed = step.before.nuniques - step.after.nuniques
        if removed < 0:
            raise ValueError(
                f"Step {step.description!r} adds {-removed} observations; the profile "
                f"layout requires a chain where every step only removes. Use "
                f"layout='cascade' instead."
            )
        label = _label(step.description, labels)
        if removed:
            reasons.append((label, removed))
        else:
            zero_reasons.append(label)
    # Stable sort: equal counts keep their tracker order.
    reasons.sort(key=lambda item: item[1], reverse=True)

    total = sum(n for _, n in reasons)
    if total != screened - included:
        raise ValueError(
            f"Flowchart does not balance: exclusion reasons sum to {total}, but "
            f"screened - included = {screened} - {included} = {screened - included}. "
            f"The profile layout requires a simple narrowing chain (each step starts "
            f"from the previous step's result). Use layout='cascade' instead."
        )

    arms: list[tuple[str, int]] = []
    if groupby is not None:
        if obs is None:
            raise ValueError("groupby requires obs to count the arms.")
        if groupby not in obs.columns:
            raise ValueError(f"groupby column {groupby!r} not found in obs.")
        counts: dict[str, int] = {}
        for key, n in obs.group_by(groupby).len().iter_rows():
            label = (group_labels or {}).get(key, str(key))
            counts[label] = counts.get(label, 0) + n
        arms = [
            (label, counts.pop(label)) for label in group_order or () if label in counts
        ]
        arms += sorted(counts.items())
        arm_total = sum(n for _, n in arms)
        if arm_total != included:
            raise ValueError(
                f"Arm sizes sum to {arm_total}, but the tracker's final step has "
                f"{included} observations; obs and the change tracker disagree."
            )

    return ProfileCounts(
        screened=screened,
        included=included,
        reasons=reasons,
        zero_reasons=zero_reasons,
        arms=arms,
    )


def profile_flowchart(
    tracker: ChangeTrackerPipeline,
    obs: pl.DataFrame | None = None,
    *,
    title: str | None = None,
    output_file: PathLike | str | None = None,
    screen_step: str | None = None,
    labels: Mapping[str, str] | None = None,
    groupby: str | None = None,
    group_labels: Mapping[Any, str] | None = None,
    group_order: Sequence[str] | None = None,
    screened_text: str = "{n} assessed for eligibility",
    excluded_text: str = "{n} excluded",
    included_text: str = "{n} included in the analysis",
    arm_text: str | Mapping[str, str] = "{n} {label}",
    footnote_text: str = "Criteria that excluded no patients: {reasons}.",
    style: FlowchartStyle | None = None,
    **style_overrides: Any,
) -> Figure:
    """Draw a trial profile: screened → excluded (all reasons) → included → arms.

    Step-level counts only: substeps are not listed, since their matches may overlap.
    Steps that removed nobody are named in a footnote.

    Args:
        tracker: The change tracker to draw.
        obs: Observations of the final cohort; required when ``groupby`` is given.
        title: Optional title above the chart.
        output_file: If given, save the figure there; the format follows the extension.
        screen_step: Substring of the step description whose result is the screened
            population. Default: the tracker's initial snapshot.
        labels: Description substring → display text for the exclusion reasons. Steps
            without a match show their description without tracker scaffolding.
        groupby: Column of ``obs`` that splits the included cohort into arms.
        group_labels: Group value → arm label.
        group_order: Arm labels in display order; unnamed arms follow, sorted.
        screened_text: Template of the top box; ``{n}``.
        excluded_text: Header of the exclusion box; ``{n}``.
        included_text: Template of the included box; ``{n}``.
        arm_text: Template of an arm box (``{n}``, ``{label}``), or a mapping from arm
            label to such a template.
        footnote_text: Template of the footnote; ``{reasons}``.
        style: Base style; ``style_overrides`` replace single fields of it.

    Returns:
        matplotlib.figure.Figure: The flowchart.

    Raises:
        ValueError: See :func:`profile_counts`.
    """
    st = _resolve_style(style, style_overrides)
    counts = profile_counts(
        tracker,
        obs,
        screen_step=screen_step,
        labels=labels,
        groupby=groupby,
        group_labels=group_labels,
        group_order=group_order,
    )

    def wrap(text: str, width: float, size: float | None = None) -> list[str]:
        return wrap_text(
            text, width, st.font_size if size is None else size, st.font_family
        )

    # --- wrap everything first: box heights follow from the line counts -----
    main_text_w = st.main_box_width - 2 * st.pad_x
    excl_w = st.page_width - st.margin - st.side_x
    reason_counts = [st.fmt(n) for _, n in counts.reasons]
    count_w = _count_column_width(reason_counts, st)

    screened_lines = wrap(screened_text.format(n=st.fmt(counts.screened)), main_text_w)
    included_lines = wrap(included_text.format(n=st.fmt(counts.included)), main_text_w)
    excl_header = wrap(
        excluded_text.format(n=st.fmt(counts.excluded)), excl_w - 2 * st.pad_x
    )
    wrapped_reasons = _wrap_items(
        [(c, label) for c, (label, _) in zip(reason_counts, counts.reasons)],
        excl_w - 2 * st.pad_x - count_w - st.count_gap,
        st,
    )

    # Arms share one width, shrunk if needed so they fit between the margins.
    n_arms = len(counts.arms)
    arm_w = st.arm_box_width
    if n_arms:
        avail = st.page_width - 2 * st.margin
        arm_w = min(arm_w, (avail - (n_arms - 1) * st.arm_gap) / n_arms)
    arm_lines = []
    for label, n in counts.arms:
        template = (
            arm_text.get(label, "{n} {label}")
            if isinstance(arm_text, Mapping)
            else arm_text
        )
        arm_lines.append(
            wrap(template.format(n=st.fmt(n), label=label), arm_w - 2 * st.pad_x)
        )

    # --- vertical layout, top down -------------------------------------------
    title_lines, title_h = _title_height(title, st)
    main_x = st.spine_x - st.main_box_width / 2
    screened = Box(
        main_x,
        st.top_margin + title_h,
        st.main_box_width,
        _box_height(len(screened_lines), st),
    )
    excl_extra = (
        (st.header_gap + _items_height(wrapped_reasons, st)) if wrapped_reasons else 0.0
    )
    excluded = Box(
        st.side_x,
        screened.bottom + st.side_top_offset,
        excl_w,
        _box_height(len(excl_header), st, extra=excl_extra),
    )
    included = Box(
        main_x,
        max(excluded.bottom + st.drop_after_side, screened.bottom + st.min_drop),
        st.main_box_width,
        _box_height(len(included_lines), st),
    )

    arms: list[Box] = []
    if n_arms:
        arm_top = included.bottom + st.split_gap
        arm_h = _box_height(max(len(lines) for lines in arm_lines), st)
        span = n_arms * arm_w + (n_arms - 1) * st.arm_gap
        x0 = st.spine_x - span / 2
        x0 = max(st.margin, min(x0, st.page_width - st.margin - span))
        arms = [
            Box(x0 + i * (arm_w + st.arm_gap), arm_top, arm_w, arm_h)
            for i in range(n_arms)
        ]

    content_bottom = max([included.bottom, excluded.bottom] + [b.bottom for b in arms])
    footnote_lines: list[str] = []
    if counts.zero_reasons:
        footnote_lines = wrap(
            footnote_text.format(reasons="; ".join(counts.zero_reasons)),
            st.page_width - 2 * st.margin,
            st.footnote_size,
        )
    footnote_h = (
        st.footnote_gap + len(footnote_lines) * st.line_height(st.footnote_size)
        if footnote_lines
        else 0.0
    )
    page_h = content_bottom + footnote_h + st.bottom_margin

    # --- paint -----------------------------------------------------------------
    fig, ax = _new_figure(st, page_h)
    _draw_title(ax, title_lines, st)
    for box, lines in [(screened, screened_lines), (included, included_lines)] + list(
        zip(arms, arm_lines)
    ):
        _draw_box(ax, box, st)
        _draw_lines(ax, lines, box.x + st.pad_x, box.y + st.pad_y, st)

    _draw_box(ax, excluded, st)
    y = _draw_lines(ax, excl_header, excluded.x + st.pad_x, excluded.y + st.pad_y, st)
    count_right = excluded.x + st.pad_x + count_w
    _draw_items(
        ax,
        wrapped_reasons,
        count_right,
        count_right + st.count_gap,
        y + st.header_gap,
        st,
    )

    _arrow(ax, (st.spine_x, excluded.cy), (excluded.x, excluded.cy), st)
    _arrow(ax, (st.spine_x, screened.bottom), (st.spine_x, included.y), st)
    if arms:
        _split(
            ax,
            st.spine_x,
            included.bottom,
            included.bottom + st.split_gap * st.split_bar_frac,
            [(box.cx, box.y) for box in arms],
            st,
        )

    if footnote_lines:
        _draw_lines(
            ax,
            footnote_lines,
            st.margin,
            content_bottom + st.footnote_gap,
            st,
            size=st.footnote_size,
        )

    return _finish(fig, output_file, st)
