import polars as pl
import pytest

from corr_vars.definitions.typing import ExtractedVariable, ObsLevel


class TestObsLevel:
    def test_enum_members(self) -> None:
        assert ObsLevel.PATIENT.value == (
            "patient_id",
            "birthdate",
            "censordate",
            "birthdate",
            "censordate",
        )
        assert ObsLevel.HOSPITAL_STAY.value == (
            "case_id",
            "hospital_admission",
            "hospital_discharge",
            "hospital_admission",
            "hospital_discharge",
        )
        assert ObsLevel.ICU_STAY.value == (
            "icu_stay_id",
            "icu_admission",
            "icu_discharge",
            "icu_admission",
            "icu_discharge",
        )
        assert ObsLevel.PROCEDURE.value == (
            "procedure_id",
            "or_time_begin",
            "or_time_end",
            "or_time_begin",
            "hospital_discharge",
        )

    def test_lower_name(self) -> None:
        assert ObsLevel.PATIENT.lower_name == "patient"
        assert ObsLevel.HOSPITAL_STAY.lower_name == "hospital_stay"
        assert ObsLevel.ICU_STAY.lower_name == "icu_stay"
        assert ObsLevel.PROCEDURE.lower_name == "procedure"

    def test_primary_keys(self) -> None:
        assert ObsLevel.primary_keys() == [
            "patient_id",
            "case_id",
            "icu_stay_id",
            "procedure_id",
        ]

    @pytest.mark.parametrize(
        "value, expected",
        [
            ("patient", ObsLevel.PATIENT),
            ("patient_stay", ObsLevel.PATIENT),
            ("hospital", ObsLevel.HOSPITAL_STAY),
            ("HOSPITAL_STAY", ObsLevel.HOSPITAL_STAY),
            ("icu", ObsLevel.ICU_STAY),
            ("icu_stay", ObsLevel.ICU_STAY),
            ("procedure", ObsLevel.PROCEDURE),
            ("non_existent", None),
        ],
    )
    def test_missing(self, value: str, expected: ObsLevel | None) -> None:
        if expected:
            assert ObsLevel(value) == expected
        else:
            with pytest.raises(ValueError):
                ObsLevel(value)


class TestExtractedVariable:
    def test_init(self) -> None:
        var = ExtractedVariable(var_name="test_var")
        assert var.var_name == "test_var"
        assert var.dynamic is True
        assert isinstance(var.data, pl.DataFrame)
        assert var.data.is_empty()

    def test_init_with_data(self) -> None:
        df = pl.DataFrame({"a": [1, 2, 3]})
        var = ExtractedVariable(var_name="test_var", dynamic=False, data=df)
        assert var.var_name == "test_var"
        assert var.dynamic is False
        assert var.data is not None
        assert var.data.equals(df)

    def test_repr_str_no_data(self) -> None:
        var = ExtractedVariable(var_name="test_var")
        repr_str = repr(var)
        assert "Variable: test_var" in repr_str
        assert "ExtractedVariable" in repr_str
        assert "dynamic" in repr_str
        assert "Nothing extracted" in repr_str
        assert repr_str == str(var)

    def test_repr_str_with_data(self) -> None:
        df = pl.DataFrame({"a": [1, 2, 3]})
        var = ExtractedVariable(var_name="test_var", data=df)
        repr_str = repr(var)
        assert "Variable: test_var" in repr_str
        assert "ExtractedVariable" in repr_str
        assert "dynamic" in repr_str
        assert "data: (3, 1)" in repr_str
        assert "Not extracted" not in repr_str
        assert repr_str == str(var)
