import pytest


@pytest.mark.parametrize(
    "test_config",
    [  # One per variable type
        {
            "var_name": "heart_rate",
            "cleaning": {"value": {"low": 70, "high": 100}},
        },  # Simply non struct variable
        {
            "var_name": "blood_sodium",
            "cleaning": {"value": {"low": 0, "high": 300}},
        },  # Simple struct variables
        {
            "var_name": "systolic_bp_combined",
            "cleaning": {"value": {"low": 10, "high": 500}},
        },  # Calculation variable
        {
            "var_name": "sofa_score",
            "cleaning": {"value": {"low": 3, "high": 15}},
        },  # Static precalculated variable
    ],
    ids=lambda param: f"var={param['var_name']}",
)
def test_extract_custom_vars_with_local_definition(sample_cohort, test_config):
    sample_cohort.add_variable_definition(
        test_config["var_name"], {"cleaning": test_config["cleaning"]}
    )
    var = sample_cohort.add_variable(test_config["var_name"])

    # Get min and max values as scalars
    if var.dynamic:
        min_value = (
            sample_cohort.obsm[test_config["var_name"]].select("value").min().item()
        )
        max_value = (
            sample_cohort.obsm[test_config["var_name"]].select("value").max().item()
        )
    else:
        min_value = sample_cohort.obs.select(test_config["var_name"]).min().item()
        max_value = sample_cohort.obs.select(test_config["var_name"]).max().item()

    assert min_value >= test_config["cleaning"]["value"]["low"]
    assert max_value <= test_config["cleaning"]["value"]["high"]
