import textwrap
import warnings
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd


@dataclass
class ImplicationRule:
    """
    Asserted implication rule between categorical variables.

    If all key–value conditions in implies hold for a row (lists meaning any listed value is acceptable), then the
    corresponding conditions in is_implied must also hold (lists defining the allowed values, with the last value used
    for imputation of missing entries).
    """

    implies: dict[str, int | list[int]]
    is_implied: dict[str, int | list[int]]


@dataclass
class EquivalenceRule:
    """
    Asserted equivalence rule between categorical variables.

    All key–value pairs in `equivalent` are treated as mutually implying: if any single condition holds, the others must
    also hold (and may be imputed if missing).
    """

    equivalent: dict[str, int | list[int]]

    def to_implication_rules(self) -> list[ImplicationRule]:
        return [
            ImplicationRule(
                implies={var_implies: value_implies},
                is_implied={var: value for var, value in self.equivalent.items() if var != var_implies},
            )
            for var_implies, value_implies in self.equivalent.items()
        ]


type Rule = ImplicationRule | EquivalenceRule
type MaskType = pd.Series[bool] | npt.NDArray[np.bool_] | list[bool]


def impute_value(data: pd.DataFrame, mask: MaskType, var: str, value: Any) -> pd.DataFrame:
    """
    Impute variable var in data with value.

    Note: If the variable is categorical and the category `value` doesn't exist it is created with a warning.
    """
    series = data[var]
    if isinstance(series.dtype, pd.CategoricalDtype):
        if value not in series.cat.categories:
            warnings.warn(f"Created new category {value} for {var}")
            data[var] = series.cat.add_categories([value])
    data.loc[mask, var] = value
    return data


def _check_and_impute_rule(
    data: pd.DataFrame, rule: Rule, failure_strategy: Literal["error", "drop"] | None = None
) -> tuple[pd.DataFrame, int, int]:
    """Enforces one rule by imputing and possibly removing observations."""

    # Make counters
    imputation_count: int = 0
    failure_count: int = 0

    for implication_rule in rule.to_implication_rules() if isinstance(rule, EquivalenceRule) else [rule]:
        # Build condition
        condition = pd.Series(True, index=data.index)
        for var, value in implication_rule.implies.items():
            condition &= data[var].isin(value) if isinstance(value, list) else data[var].eq(value)

        # Check and impute
        for var, value in implication_rule.is_implied.items():
            # Impute
            na_mask = condition & data[var].isna()
            if (na_sum := na_mask.sum()) != 0:
                data = impute_value(data, na_mask, var, value[-1] if isinstance(value, list) else value)
                imputation_count += na_sum

            # Failures
            failure_mask = condition & ~(data[var].isin(value) if isinstance(value, list) else data[var].eq(value))
            if (failure_sum := failure_mask.sum()) != 0:
                if failure_strategy == "drop":
                    data = data.loc[~failure_mask]
                failure_count += failure_sum

    return data, imputation_count, failure_count


def check_and_impute(
    data: pd.DataFrame, rules: list[Rule], failure_strategy: Literal["error", "drop"] | None = None
) -> pd.DataFrame:
    """
    Enforces conditions by imputing and possibly raising an error or removing observations.

    data:
        The data to compare and impute.
    rules:
        List of rules to check and impute by. Prints a message for every element of rules, group them for less printing.
    failure_strategy:
        What to do when an observation does not conform to a rule.
    """
    data = data.copy()
    if failure_strategy == "drop":
        initial_len = len(data)

    for rule in rules:
        data, imputation_count, failure_count = _check_and_impute_rule(data, rule, failure_strategy)

        # Print message
        shortened_rule = textwrap.shorten(str(rule), width=85, placeholder="...")
        if failure_strategy == "error" and failure_count != 0:
            raise AssertionError(f"Rule {shortened_rule} failed {failure_count} times.")
        label = "dropped" if failure_strategy == "drop" else "failed"
        print(f"{imputation_count} imputed, {failure_count} {label}, used rule {shortened_rule}")

    if failure_strategy == "drop":
        print(f"\nTOTAL: {initial_len - len(data)} dropped")
    return data
