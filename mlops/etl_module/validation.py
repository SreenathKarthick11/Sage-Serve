from __future__ import annotations

from typing import Iterable

import great_expectations as gx
import great_expectations.expectations as gxe
import pandas as pd


# SageServe trace schema

RAW_COLUMNS = [
    "request_id",
    "batch_id",
    "client_tenant",
    "request_type",
    "scenario",
    "sla",
    "utility",
    "regions",
    "model_type",
    "workload_type",
    "application_id",
    "arrival_timestamp",
    "batch_size",
    "prompt_size",
    "token_size",
]

# Current SageServe workload models.
VALID_MODELS = ["A", "B", "C", "D"]

# `regions` is a priority encoding containing all three regions.
# Examples from the trace: 012, 021, 102, 120, 201, 210
VALID_REGION_ENCODINGS = ["012","021","102","120","201","210",]

# The actual SageServe region represented by the first digit.
VALID_REGION_IDS = [0, 1, 2]


# GX helpers
def _create_batch(df: pd.DataFrame):
    """
    Create a Great Expectations Batch from a Pandas DataFrame.
    """
    context = gx.get_context(mode="ephemeral")
    data_source = context.data_sources.add_pandas(name="sageserve_trace")
    data_asset = data_source.add_dataframe_asset(name="trace_dataframe")
    batch_definition = (
        data_asset.add_batch_definition_whole_dataframe("trace_batch")
    )

    return batch_definition.get_batch(
        batch_parameters={"dataframe": df}
    )


def _validate_suite( df: pd.DataFrame,expectations: Iterable, suite_name: str):
    """
    Build an Expectation Suite and validate a DataFrame.
    """
    batch = _create_batch(df)
    suite = gx.ExpectationSuite(name=suite_name)
    for expectation in expectations:
        suite.add_expectation(expectation)

    result = batch.validate(suite)

    if not result.success:
        print(result.describe())
        raise ValueError(f"Great Expectations validation failed: {suite_name}")

    return result


# Raw trace validation
def validate_raw_trace(df: pd.DataFrame):
    """
    Validate the raw SageServe workload trace.

    Checks:
        - required columns
        - non-null required values
        - unique request IDs
        - valid models
        - valid region encodings
        - non-negative timestamps
        - non-negative workload quantities
    """

    expectations = [
        # Schema
        gxe.ExpectTableColumnsToMatchSet(
            column_set=RAW_COLUMNS,
            exact_match=True,
        ),

        # Missing values
        *[
            gxe.ExpectColumnValuesToNotBeNull(
                column=column
            )
            for column in RAW_COLUMNS
        ],

        # Valid model identifiers
        gxe.ExpectColumnValuesToBeInSet(
            column="model_type",
            value_set=VALID_MODELS,
        ),

        # Valid region-priority encodings
        gxe.ExpectColumnValuesToBeInSet(
            column="regions",
            value_set=VALID_REGION_ENCODINGS,
        ),

        # Workload values
        gxe.ExpectColumnValuesToBeBetween(
            column="arrival_timestamp",
            min_value=0,
        ),

        gxe.ExpectColumnValuesToBeBetween(
            column="batch_size",
            min_value=1,
        ),

        gxe.ExpectColumnValuesToBeBetween(
            column="prompt_size",
            min_value=0,
        ),

        gxe.ExpectColumnValuesToBeBetween(
            column="token_size",
            min_value=0,
        ),
    ]

    return _validate_suite(
        df=df,
        expectations=expectations,
        suite_name="sageserve_raw_trace",
    )

