from __future__ import annotations

from datetime import date

import pytest

from corr_vars.concepts.spec import (
    LATEST_SELECTOR,
    VariableSpec,
    VersionSelector,
    parse_variable_spec,
    parse_version_selector,
    resolve_default_version,
)


class TestParseVersionSelector:
    def test_latest(self) -> None:
        assert parse_version_selector("latest") == LATEST_SELECTOR
        assert parse_version_selector("latest").query_params == {}

    @pytest.mark.parametrize("text,number", [("v1", 1), ("v3", 3), ("v42", 42)])
    def test_version_number(self, text: str, number: int) -> None:
        selector = parse_version_selector(text)
        assert selector == VersionSelector(kind="version", value=number)
        assert selector.query_params == {"v": str(number)}

    def test_iso_date(self) -> None:
        selector = parse_version_selector("2024-05-06")
        assert selector == VersionSelector(kind="date", value=date(2024, 5, 6))
        assert selector.query_params == {"date": "2024-05-06"}

    def test_draft(self) -> None:
        selector = parse_version_selector("draft1234")
        assert selector == VersionSelector(kind="draft", value=1234)
        assert selector.query_params == {"draft": "1234"}

    def test_date_uses_date_param_not_vd(self) -> None:
        assert "vd" not in parse_version_selector("2024-05-06").query_params

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "   ",
            "V3",
            "v",
            "v0",
            "v-1",
            "3",
            "draft",
            "draft0",
            "newest",
            "2024-5-6",
            "2024-13-01",
            "2024-02-30",
            "20240506",
            " v3",
            "v3 ",
        ],
    )
    def test_malformed(self, text: str) -> None:
        with pytest.raises(ValueError):
            parse_version_selector(text)

    def test_error_message_lists_accepted_forms(self) -> None:
        with pytest.raises(ValueError, match="draftNNNN"):
            parse_version_selector("nonsense")

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("latest", "latest"),
            ("v3", "v3"),
            ("draft77", "draft77"),
            ("2024-05-06", "2024-05-06"),
        ],
    )
    def test_roundtrips_through_str(self, text: str, expected: str) -> None:
        assert str(parse_version_selector(text)) == expected


class TestParseVariableSpec:
    def test_bare_name_uses_both_defaults(self) -> None:
        spec = parse_variable_spec("blood_sodium", default_taxonomy="corr_v1")
        assert spec == VariableSpec("blood_sodium", "corr_v1", LATEST_SELECTOR)

    def test_taxonomy_qualified(self) -> None:
        spec = parse_variable_spec("other_tax/blood_sodium", default_taxonomy="corr_v1")
        assert spec.taxonomy == "other_tax"
        assert spec.name == "blood_sodium"
        assert spec.version == LATEST_SELECTOR

    def test_version_pinned(self) -> None:
        spec = parse_variable_spec("blood_sodium::v3", default_taxonomy="corr_v1")
        assert spec.taxonomy == "corr_v1"
        assert spec.name == "blood_sodium"
        assert spec.version == VersionSelector(kind="version", value=3)

    def test_fully_qualified(self) -> None:
        spec = parse_variable_spec("tax/name::draft9", default_taxonomy="corr_v1")
        assert spec == VariableSpec("name", "tax", VersionSelector("draft", 9))

    def test_default_version_is_used_when_absent(self) -> None:
        default = VersionSelector(kind="date", value=date(2023, 1, 1))
        spec = parse_variable_spec(
            "blood_sodium", default_taxonomy="corr_v1", default_version=default
        )
        assert spec.version == default

    def test_explicit_version_beats_default_version(self) -> None:
        default = VersionSelector(kind="date", value=date(2023, 1, 1))
        spec = parse_variable_spec(
            "blood_sodium::v2", default_taxonomy="corr_v1", default_version=default
        )
        assert spec.version == VersionSelector(kind="version", value=2)

    def test_explicit_taxonomy_beats_default_taxonomy(self) -> None:
        spec = parse_variable_spec("mine/x", default_taxonomy="corr_v1")
        assert spec.taxonomy == "mine"

    @pytest.mark.parametrize(
        "reference",
        [
            "",
            "   ",
            "::v3",
            "blood_sodium::",
            "blood_sodium::v3::v4",
            "a/b/c",
            "/blood_sodium",
            "tax/",
            "blood sodium",
            "blood/sodium::bogus",
            "blood_sodium::V3",
        ],
    )
    def test_malformed(self, reference: str) -> None:
        with pytest.raises(ValueError):
            parse_variable_spec(reference, default_taxonomy="corr_v1")

    def test_non_string_rejected(self) -> None:
        with pytest.raises(ValueError):
            parse_variable_spec(3, default_taxonomy="corr_v1")  # type: ignore[arg-type]

    def test_str_shows_all_three_parts(self) -> None:
        spec = parse_variable_spec("blood_sodium::v3", default_taxonomy="corr_v1")
        assert str(spec) == "corr_v1/blood_sodium::v3"


class TestResolveDefaultVersion:
    def test_none_is_latest(self) -> None:
        assert resolve_default_version(None, None) == LATEST_SELECTOR

    def test_latest_string(self) -> None:
        assert resolve_default_version("latest", None) == LATEST_SELECTOR

    def test_explicit_version(self) -> None:
        assert resolve_default_version("v7", None) == VersionSelector("version", 7)

    def test_date_string(self) -> None:
        assert resolve_default_version("latest", "2024-05-06") == VersionSelector(
            "date", date(2024, 5, 6)
        )

    def test_date_object(self) -> None:
        assert resolve_default_version(None, date(2024, 5, 6)) == VersionSelector(
            "date", date(2024, 5, 6)
        )

    def test_version_and_date_together_are_ambiguous(self) -> None:
        with pytest.raises(ValueError, match="only one"):
            resolve_default_version("v3", "2024-05-06")

    def test_bad_date_rejected(self) -> None:
        with pytest.raises(ValueError):
            resolve_default_version(None, "not-a-date")
