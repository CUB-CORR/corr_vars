from __future__ import annotations

import hashlib

import numpy as np
import polars as pl
from graphviz import Digraph
from polars._utils.parse import parse_into_list_of_expressions
from polars._utils.wrap import wrap_expr

from corr_vars.utils import convert_to_polars_lf

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    from os import PathLike

    import pandas as pd

    from corr_vars.definitions import Cohort, IntoExprColumn

    from collections.abc import Iterable
    from types import TracebackType


def _hash_ids(arr: np.ndarray) -> str:
    if arr.size == 0:
        return hashlib.sha256(b"").hexdigest()
    # sort stable copy so hash is order-independent
    b = np.sort(arr).tobytes()
    return hashlib.sha256(b).hexdigest()


@dataclass
class ChangeTrackerSnapshot:
    rows: int
    nuniques: int
    unique_ids: np.ndarray = field(repr=False)
    unique_hash: str

    @classmethod
    def from_df(
        cls,
        df: pl.DataFrame | pl.LazyFrame | pd.DataFrame,
        primary_key: str,
        use_hash: bool = True,
    ) -> ChangeTrackerSnapshot:
        lf = convert_to_polars_lf(df)
        rows = cast("int", lf.select(pl.count()).collect().item())
        nuniques = cast(
            "int", lf.select(pl.col(primary_key).n_unique()).collect().item()
        )
        unique_series = lf.select(pl.col(primary_key)).unique().collect()[primary_key]
        unique_ids = unique_series.to_numpy()
        unique_hash = _hash_ids(unique_ids) if use_hash else ""
        return cls(
            rows=rows, nuniques=nuniques, unique_ids=unique_ids, unique_hash=unique_hash
        )


@dataclass
class ChangeTrackerStep:
    description: str
    before: ChangeTrackerSnapshot
    after: ChangeTrackerSnapshot
    substeps: list[ChangeTrackerStep] = field(default_factory=list)
    group: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def delta_rows(self) -> int:
        return self.after.rows - self.before.rows

    @property
    def delta_uniques(self) -> int:
        return self.after.nuniques - self.before.nuniques

    def add_substep(
        self,
        after_df: pl.DataFrame | pl.LazyFrame | pd.DataFrame,
        description: str,
        primary_key: str,
        use_hash: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> ChangeTrackerStep:
        after_snap = ChangeTrackerSnapshot.from_df(after_df, primary_key, use_hash)
        sub = ChangeTrackerStep(
            description=description,
            before=self.before,
            after=after_snap,
            group=self.group,
            metadata=metadata or {},
        )
        self.substeps.append(sub)
        return sub


@dataclass
class ChangeTrackerGroup:
    name: str
    steps: list[ChangeTrackerStep] = field(default_factory=list)

    def add_step(self, step: ChangeTrackerStep) -> None:
        self.steps.append(step)


class ChangeTrackerPipeline:
    """This pipeline logs successive changes to a dataset, allowing for a step-by-step
    review of how the dataset was filtered. It can create branches (arms) that are
    independent pipelines whose initial state is the pipeline's current snapshot.

    The tracker maintains a list of steps (optionally grouped), where each step represents
    the change of the DataFrame at a specific point in time. It automatically calculates
    the number of rows and unique IDs removed between each step, which is useful for
    generating PRISMA-style flowcharts.
    """

    def __init__(
        self,
        primary_key: str,
        initial_df: pl.DataFrame | pl.LazyFrame | pd.DataFrame,
        initial_description: str = "Initial state",
        use_hash: bool = True,
    ):
        self.primary_key = primary_key
        self.use_hash = use_hash
        self.initial_snapshot = ChangeTrackerSnapshot.from_df(
            initial_df, primary_key, use_hash
        )
        self.initial_description = initial_description
        self.steps: list[ChangeTrackerStep] = []
        self.groups: dict[str, ChangeTrackerGroup] = {}
        self._current_df = convert_to_polars_lf(
            initial_df
        )  # lazyframe representing current state

    @property
    def current_step(self) -> ChangeTrackerStep | None:
        return self.steps[-1] if self.steps else None

    @property
    def current_snapshot(self) -> ChangeTrackerSnapshot:
        return ChangeTrackerSnapshot.from_df(
            self._current_df, self.primary_key, self.use_hash
        )

    @property
    def current_rows(self) -> int:
        return self.current_snapshot.rows

    @property
    def current_uniques(self) -> int:
        return self.current_snapshot.nuniques

    @property
    def delta_rows(self) -> int:
        return self.initial_snapshot.rows - self.current_rows

    @property
    def delta_uniques(self) -> int:
        return self.initial_snapshot.nuniques - self.current_uniques

    def add_step(
        self,
        after_df: pl.DataFrame | pl.LazyFrame | pd.DataFrame,
        description: str,
        group: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ChangeTrackerStep:
        before_snap = self.current_snapshot
        after_snap = ChangeTrackerSnapshot.from_df(
            after_df, self.primary_key, self.use_hash
        )
        step = ChangeTrackerStep(
            description=description,
            before=before_snap,
            after=after_snap,
            group=group,
            metadata=metadata or {},
        )
        self.steps.append(step)
        if group:
            self.groups.setdefault(group, ChangeTrackerGroup(group)).add_step(step)
        # update pipeline current state to after_df
        self._current_df = convert_to_polars_lf(after_df)
        return step

    def add_step_from_filter(
        self,
        filter_expr: pl.Expr,
        description: str,
        group: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ChangeTrackerStep:
        # apply a polars expression to current df (include rule)
        after_lf = self._current_df.filter(filter_expr)
        return self.add_step(after_lf, description, group, metadata)

    def create_branch(
        self,
        name: str,
        start_df: pl.DataFrame | pl.LazyFrame | pd.DataFrame | None = None,
    ) -> ChangeTrackerPipeline:
        # If start_df is provided, branch from there, otherwise use current pipeline snapshot
        start = start_df if start_df is not None else self._current_df
        return ChangeTrackerPipeline(
            primary_key=self.primary_key,
            initial_df=start,
            initial_description=name,
            use_hash=self.use_hash,
        )

    def summarize(self, show_hash: bool = False) -> str:
        lines = []
        lines.append(
            f"Initial: rows={self.initial_snapshot.rows}, "
            f"uniques={self.initial_snapshot.nuniques}"
            + (
                f", description='{self.initial_description}'"
                if self.initial_description
                else ""
            )
            + (f", hash={self.initial_snapshot.unique_hash}" if show_hash else "")
        )
        for i, s in enumerate(self.steps):
            lines.append(
                f"Step {i}: '{s.description}' (group={s.group}) "
                f"rows: {s.before.rows} -> {s.after.rows} (Δ={s.delta_rows}), "
                f"uniques: {s.before.nuniques} -> {s.after.nuniques} (Δ={s.delta_uniques})"
                + (f", hash={s.after.unique_hash}" if show_hash else "")
            )
            for j, sub in enumerate(s.substeps):
                lines.append(
                    f"  Sub {i}.{j}: '{sub.description}' "
                    f"rows: {sub.before.rows} -> {sub.after.rows} (Δ={sub.delta_rows}), "
                    f"uniques: {sub.before.nuniques} -> {sub.after.nuniques} (Δ={sub.delta_uniques})"
                )
        return "\n".join(lines)

    def print_summary(self, **kwargs) -> None:
        """Prints the summarized states to the console."""
        print(self.summarize(**kwargs))

    def create_flowchart(
        self,
        title: str = "Study Inclusion Flowchart",
        output_file: PathLike | str | None = None,
    ) -> Digraph:
        """Creates a PRISMA-style flowchart of the tracked changes using the
        ChangeTrackerPipeline dataclasses (initial snapshot, steps and substeps).

        Returns a graphviz.Digraph.
        """
        dot = Digraph(comment=title)
        dot.attr(rankdir="TB", newrank="true")
        dot.attr("node", shape="box", style="rounded")

        # initial node
        init_id = "init"
        init_label = f"{self.initial_description}\n(n={self.initial_snapshot.nuniques})"
        dot.node(init_id, init_label)

        # build nodes/edges and collect ranks
        ranks: dict[int, list[str]] = {0: [init_id]}
        prev_id = init_id

        for i, step in enumerate(self.steps):
            step_id = f"s{i}"
            step_label = f"{step.description}\n(n={step.after.nuniques})"
            dot.node(step_id, step_label)
            dot.edge(prev_id, step_id)

            # put main step and its substeps/dropped node on the same rank (horizontal alignment)
            rank = i + 1
            ranks.setdefault(rank, []).append(step_id)

            dropped_uniques = max(0, step.before.nuniques - step.after.nuniques)
            drop_id = f"s{i}_dropped"
            drop_label = f"Dropped (n={dropped_uniques})"
            if step.substeps:
                sub_lines = []
                for sub in step.substeps:
                    sub_lines.append(f"- {sub.description} (n={sub.after.nuniques})")
                sub_label = "\n".join(sub_lines)
                drop_label += "\n" + sub_label
            dot.node(drop_id, drop_label, style="dashed")
            dot.edge(step_id, drop_id)
            ranks.setdefault(rank, []).append(drop_id)

            prev_id = step_id

        # enforce same-rank groups
        for rank in sorted(ranks.keys()):
            nodes_at_rank = ranks[rank]
            if nodes_at_rank:
                dot.body.append(f"{{rank = same; {' '.join(nodes_at_rank)}}}")

        if output_file:
            dot.render(output_file, format="svg", view=False, cleanup=True)

        return dot

    def __repr__(self) -> str:
        current_step = self.current_step
        current_info = ""
        if current_step:
            current_info = (
                f", current_rows={current_step.after.rows}, "
                f"current_uniques={current_step.after.nuniques}"
                f"current_description={current_step.description}"
            )

        return (
            f"ChangeTrackerPipeline(primary_key='{self.primary_key}', "
            f"steps={len(self.steps)}"
            f"{current_info})"
        )


class ChangeTrackerContext:
    """Context manager to group cohort edits and record one ChangeTracker state on exit.

    Usage:
        with cohort.change_tracker("Adult patients", mode="include") as track:
            # either apply helper
            track.filter(pl.col("age_on_admission") >= 18)
            # or perform direct cohort edits:
            cohort.obs = cohort.obs.filter(pl.col("age_on_admission") >= 18)
    """

    _before_ids: pl.Series
    _filters: list[tuple[pl.Expr, dict[str, pl.Expr]]]

    def __init__(
        self,
        pipeline: ChangeTrackerPipeline,
        cohort: Cohort,
        description: str,
        group: str | None = None,
        mode: Literal["include", "exclude"] = "include",
    ) -> None:
        self.pipeline = pipeline
        self.cohort = cohort
        self.description = description
        self.group = group
        self.mode = mode

    def __enter__(self) -> ChangeTrackerContext:
        # snapshot unique ids before change
        df = self.cohort._obs
        self._before_ids = df[self.pipeline.primary_key].unique().sort()
        self._filters = []
        return self

    def _parse_as_expr(
        self,
        *predicates: (
            IntoExprColumn
            | Iterable[IntoExprColumn]
            | bool
            | list[bool]
            | np.ndarray[Any, Any]
        ),
        **constraints: Any,
    ) -> list[pl.Expr]:
        exprs = [wrap_expr(x) for x in parse_into_list_of_expressions(*predicates)]
        for column, value in constraints.items():
            exprs.append(pl.col(column) == value)
        return exprs

    def _apply_expr(self, df: pl.DataFrame, expr: pl.Expr) -> pl.DataFrame:
        if self.mode == "include":
            return df.filter(expr)
        if self.mode == "exclude":
            return df.remove(expr)
        raise ValueError(f"Invalid mode: {self.mode}")

    def filter(
        self,
        *predicates: (
            IntoExprColumn
            | Iterable[IntoExprColumn]
            | bool
            | list[bool]
            | np.ndarray[Any, Any]
        ),
        **constraints: Any,
    ) -> None:
        """Apply a boolean pl.Expr filter inside the context.

        If mode == "include", keep rows matching expr.
        If mode == "exclude", drop rows matching expr.
        """
        parsed_predicates = self._parse_as_expr(*predicates, **constraints)

        # combine predicates via AND to get union of matches
        combined_predicate = parsed_predicates[0]
        for predicate in parsed_predicates[1:]:
            combined_predicate = combined_predicate & predicate

        self._filters.append((combined_predicate, {}))

    def multiple_can_apply_filter(
        self,
        *predicates: (
            IntoExprColumn
            | Iterable[IntoExprColumn]
            | bool
            | list[bool]
            | np.ndarray[Any, Any]
        ),
        descriptions: list[str] | None = None,
        **constraints: Any,
    ) -> None:
        """Apply a boolean pl.Expr filter inside the context.

        If mode == "include", keep rows matching expr.
        If mode == "exclude", drop rows matching expr.
        """
        parsed_predicates = self._parse_as_expr(*predicates, **constraints)

        if descriptions and len(descriptions) != len(parsed_predicates):
            raise ValueError(
                "Number of descriptions must match the number of predicates."
            )

        sub_predicates_mapping: dict[str, pl.Expr] = {}
        for i, predicate in enumerate(parsed_predicates):
            if descriptions:
                description = descriptions[i]
            else:
                description = predicate.meta.output_name(
                    raise_if_undetermined=False
                ) or " + ".join(predicate.meta.root_names())

            sub_predicates_mapping[description] = predicate

        # combine predicates via OR to get union of matches
        combined_predicate = parsed_predicates[0]
        for predicate in parsed_predicates[1:]:
            combined_predicate = combined_predicate | predicate

        self._filters.append((combined_predicate, sub_predicates_mapping))

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        # If an exception occurred, do not add a state (let the exception propagate)
        if exc_type is not None:
            return False

        df = self.cohort._obs
        after_ids = df[self.pipeline.primary_key].unique().sort()

        # User filtered obs manually and didn't use single_filter or multiple_filter
        if not after_ids.equals(self._before_ids) and not self._filters:
            desc = f"{self.mode.capitalize()} {self.description}"
            self.pipeline.add_step(df, desc, self.group)
            return False

        # User did not filter manually and did use single_filter or multiple_filter only
        if after_ids.equals(self._before_ids) and self._filters:
            # use the temporary pipeline to compute snapshots and final dfs
            before_df = df
            for predicate, sub_predicates_mapping in self._filters:
                # Parent (the overall effect of combined predicate)
                final_df = self._apply_expr(before_df, predicate)
                desc = f"{self.mode.capitalize()} {self.description}"
                # add parent state for the overall union/combined predicate
                parent = self.pipeline.add_step(final_df, desc, self.group)

                # If there are sub-predicates, record each as a substate computed from the same before_df
                if sub_predicates_mapping:
                    for (
                        sub_description,
                        sub_predicate,
                    ) in sub_predicates_mapping.items():
                        sub_after = self._apply_expr(before_df, sub_predicate)
                        # substate: computed relative to parent (parent provided to keep hierarchical relation)
                        parent.add_substep(
                            after_df=sub_after,
                            description=sub_description,
                            primary_key=self.pipeline.primary_key,
                        )

                # update working df and cohort obs
                before_df = final_df
                self.cohort._obs = final_df

            return False

        return False
