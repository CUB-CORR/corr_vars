import polars as pl

from corr_vars import __, logger
from corr_vars.definitions import ObsLevel, ObsLevelNotSupportedError
from corr_vars.sources.reprodicu import helpers
from corr_vars.sources.reprodicu.config import SourceDict


def load_data(obs_level, source_kwargs: SourceDict):
    """Load data from reprodicu parquet files.

    Args:
        obs_level (ObsLevel): Level of observation ("icu_stay", "hospital_stay", "procedure")
        source_kwargs: Dictionary containing source configuration
            - path: Path to reprodicu parquet files
            - exclude_datasets: List of datasets to exclude (optional)
            - include_datasets: List of datasets to include (optional)

    Returns:
        pl.DataFrame: Loaded observation data
    """
    path = source_kwargs["path"]
    exclude_datasets = source_kwargs["exclude_datasets"]
    include_datasets = source_kwargs["include_datasets"]

    if not path:
        raise ValueError(
            "Path to reprodicu data files must be specified in source_kwargs['path']"
        )

    # Load patient information
    logger.info("Loading reprodicu patient information")
    ov = helpers.read_parquet(path, "patient_information")

    # Handle different observation levels
    match obs_level:
        case ObsLevel.HOSPITAL_STAY:
            raise NotImplementedError(
                "Hospital stay level not implemented for reprodicu"
            )
        case ObsLevel.PROCEDURE:
            raise NotImplementedError("Procedure level not implemented for reprodicu")
        case ObsLevel.ICU_STAY:
            # Default case for ICU stay level
            pass
        case _:
            raise ObsLevelNotSupportedError(
                f"Unsupported observation level: {obs_level}"
            )

    # Filter datasets if specified
    if include_datasets:
        logger.info(
            __(
                "Including only datasets: {include_datasets}",
                include_datasets=lambda: ", ".join(include_datasets),
            )
        )
        ov = ov.filter(pl.col("Source Dataset").is_in(include_datasets))

    if exclude_datasets:
        logger.info(
            __(
                "Excluding datasets: {exclude_datasets}",
                exclude_datasets=lambda: ", ".join(exclude_datasets),
            )
        )
        ov = ov.filter(pl.col("Source Dataset").is_in(exclude_datasets).not_())

    # Column mapping from reprodicu to corr-vars standard names
    column_mapping = {
        "Global Person ID": "patient_id",
        "Global Hospital Stay ID": "case_id",
        "Global ICU Stay ID": "icu_stay_id",
        "Admission Age (years)": "age_on_admission",
        "Admission Height (cm)": "height_on_admission",
        "Admission Weight (kg)": "weight_on_admission",
        "Gender": "sex",
        "Ethnicity": "ethnicity",
        "Primary (Admission) Diagnosis (ICD)": "primary_diagnosis",
        "Care Site": "icu_id",
        "Mortality in Hospital": "inhospital_death",
        "Source Dataset": "reprodicu_dataset",
        "ICU Length of Stay (days)": "icu_length_of_stay",
        "Hospital Length of Stay (days)": "hospital_length_of_stay",
        # "icu_admission" is added later
        # "icu_discharge" is added later
    }

    # Select and rename columns that exist in the dataset
    available_columns = ov.columns
    filtered_column_mapping = {
        k: v for k, v in column_mapping.items() if k in available_columns
    }

    missing_columns = set(column_mapping.keys()) - set(ov.columns)
    unused_columns = set(ov.columns) - set(column_mapping.keys())

    logger.info(
        __(
            "Available columns:\n{available_columns}",
            available_columns=lambda: helpers.pretty_join(available_columns),
        )
    )
    if not filtered_column_mapping:
        raise ValueError("No expected columns found in patient_information.")
    if missing_columns:
        raise ValueError(
            __(
                "Some columns were not not found in patient_information.\nUnused:\n{unused_columns}\nMissing:\n{missing_columns}",
                unused_columns=lambda: helpers.pretty_join(unused_columns),
                missing_columns=lambda: helpers.pretty_join(missing_columns),
            )
        )

    ov = ov.select(filtered_column_mapping.keys()).rename(filtered_column_mapping)

    # Apply standard transformations
    first_transformations = []
    last_transformations = []

    # Sex transformation
    if "sex" in ov.columns:
        first_transformations.append(
            pl.col("sex")
            .cast(pl.Utf8)
            .replace({"Male": "M", "Female": "F"})
            .alias("sex")
        )

    # Boolean transformations
    if "inhospital_death" in ov.columns:
        first_transformations.append(
            pl.col("inhospital_death").cast(pl.Boolean).alias("inhospital_death")
        )

    # Date/time transformations (if columns exist)
    # datetime_columns = [
    #     "icu_admission",
    #     "icu_discharge",
    #     "hospital_admission",
    #     "hospital_discharge",
    # ]
    # for col in datetime_columns:
    #     if col in ov.columns:
    #         first_transformations.append(
    #             pl.col(col)
    #             .str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False)
    #             .alias(col)
    #         )

    # For reprodicu, if we don't have actual admission/discharge times,
    # we need to create them based on the relative time structure
    if "icu_admission" not in ov.columns:
        # Create a dummy ICU admission time - this will be the reference point (time 0)
        # for the relative times in the timeseries data
        first_transformations.append(
            pl.datetime(1900, 1, 1, 0, 0, 0).alias("icu_admission")
        )

    if "hospital_admission" not in ov.columns:
        # Set hospital admission to be the same as ICU admission for simplicity
        if "icu_admission" not in ov.columns:
            # Both are missing, create dummy times
            first_transformations.append(
                pl.datetime(1900, 1, 1, 0, 0, 0).alias("hospital_admission")
            )
        else:
            # ICU admission exists (either original or will be created), reference it
            first_transformations.append(
                pl.col("icu_admission").alias("hospital_admission")
            )

    if "icu_discharge" not in ov.columns:
        last_transformations.append(
            (
                pl.col("icu_admission")
                + pl.duration(seconds=pl.col("icu_length_of_stay") * 86400)
            ).alias("icu_discharge")
        )
    if "hospital_discharge" not in ov.columns:
        last_transformations.append(
            (
                pl.col("hospital_admission")
                + pl.duration(seconds=pl.col("hospital_length_of_stay") * 86400)
            ).alias("hospital_discharge")
        )

    if first_transformations:
        ov = ov.with_columns(first_transformations)
    if last_transformations:
        ov = ov.with_columns(last_transformations)

    logger.info(
        __(
            "Loaded {n_ov} {obs_level} observations from reprodicu",
            n_ov=len(ov),
            obs_level=obs_level,
        )
    )
    logger.info(
        __(
            "Datasets included: {data_sources}",
            data_sources=lambda: (
                ",".join(ov.select("data_source").unique().to_series().to_list())
                if "data_source" in ov.columns
                else "Unknown"
            ),
        ),
    )

    return ov
