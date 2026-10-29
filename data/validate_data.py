"""Validate the training dataset with Great Expectations.

Checks (project contract):
- the expected columns exist
- review text is never empty
- star rating is between 1 and 5
- label is one of: negative, neutral, positive
- the dataset is not suspiciously small

The validate_reviews() function can be reused later for crawled reviews.

Usage:
    python data/validate_data.py
Exit code is 0 when all checks pass and 1 otherwise, so CI and the
auto-retrain loop can stop when the data is broken.
"""

import sys
from pathlib import Path

import great_expectations as gx
import pandas as pd

DATA_PATH = Path(__file__).resolve().parent / "raw" / "amazon_reviews.csv"
LABELS = ["negative", "neutral", "positive"]
REQUIRED_COLUMNS = ["text", "rating", "label"]
MIN_ROWS = 1_000


def build_suite() -> gx.ExpectationSuite:
    """Create the list of rules the data must follow."""
    suite = gx.ExpectationSuite(name="reviews_suite")
    for column in REQUIRED_COLUMNS:
        suite.add_expectation(gx.expectations.ExpectColumnToExist(column=column))
        suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column=column))
    suite.add_expectation(
        gx.expectations.ExpectColumnValueLengthsToBeBetween(column="text", min_value=1)
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(column="rating", min_value=1, max_value=5)
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(column="label", value_set=LABELS)
    )
    suite.add_expectation(gx.expectations.ExpectTableRowCountToBeBetween(min_value=MIN_ROWS))
    return suite


def validate_reviews(df: pd.DataFrame) -> bool:
    """Run all checks on a DataFrame, print a report, return True if all pass."""
    context = gx.get_context(mode="ephemeral")
    data_source = context.data_sources.add_pandas(name="reviews_source")
    data_asset = data_source.add_dataframe_asset(name="reviews")
    batch_definition = data_asset.add_batch_definition_whole_dataframe("all_rows")
    suite = context.suites.add(build_suite())

    validation = context.validation_definitions.add(
        gx.ValidationDefinition(name="reviews_validation", data=batch_definition, suite=suite)
    )
    results = validation.run(batch_parameters={"dataframe": df})

    for result in results.results:
        config = result.expectation_config
        column = config.kwargs.get("column", "table")
        status = "PASS" if result.success else "FAIL"
        print(f"[{status}] {config.type} ({column})")

    return results.success


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    print(f"Validating {len(df)} rows from {DATA_PATH}\n")
    if validate_reviews(df):
        print("\nAll checks passed.")
        sys.exit(0)
    print("\nSome checks FAILED. Fix the data before training.")
    sys.exit(1)


if __name__ == "__main__":
    main()
