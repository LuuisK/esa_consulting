import textwrap
import warnings
from collections import Counter
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

pd.options.mode.copy_on_write = True
pd.options.future.infer_string = True


type MaskType = pd.Series[bool] | npt.NDArray[np.bool_] | list[bool]


@dataclass
class ImplicationRule:
    """
    Asserted implication rule between categorical variables.

    If all key–value conditions in implies hold for a row (lists meaning any listed value is acceptable), then the
    corresponding conditions in is_implied must also hold (lists defining the allowed values).
    """

    implies: dict[str, Any | list[Any]]
    is_implied: dict[str, Any | list[Any]]


@dataclass
class EquivalenceRule:
    """
    Asserted equivalence rule between categorical variables.

    All key–value pairs in `equivalent` are treated as mutually implying: if any single condition holds, the others are
    asserted to also hold.
    """

    equivalent: dict[str, Any | list[Any]]

    def to_implication_rules(self) -> list[ImplicationRule]:
        return [
            ImplicationRule(
                implies={var_implies: value_implies},
                is_implied={var: value for var, value in self.equivalent.items() if var != var_implies},
            )
            for var_implies, value_implies in self.equivalent.items()
        ]


type Rule = ImplicationRule | EquivalenceRule


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


def _enforce_rule(
    data: pd.DataFrame,
    rule: Rule,
    na_strategy: Literal["impute", "error"] | None = "impute",
    failure_strategy: Literal["drop", "error"] | None = None,
) -> tuple[pd.DataFrame, int, int]:
    """
    Enforces one rule by possibly imputing, dropping observations, or raising errors.

    Note: Ignores cases where there are several possible imputation values with a warning.
    """
    # Make counters
    na_count: int = 0
    failure_count: int = 0
    imputation_warning = Counter()

    for implication_rule in rule.to_implication_rules() if isinstance(rule, EquivalenceRule) else [rule]:
        # Build condition under which rule is asserted
        condition = pd.Series(True, index=data.index)
        for var, value in implication_rule.implies.items():
            # Ensure value is a list
            value = value if isinstance(value, list) else [value]
            condition &= data[var].isin(value)

        for var, value in implication_rule.is_implied.items():
            # Ensure value is a list
            value = value if isinstance(value, list) else [value]

            # Deal with NAs
            na_mask = condition & data[var].isna()
            if (na_sum := na_mask.sum()) != 0:
                na_count += na_sum
                if na_strategy == "impute":
                    if len(value) != 1:
                        imputation_warning.update({var: na_sum})
                        na_count -= na_sum
                    else:
                        data = impute_value(data, na_mask, var, value[0])

            # Deal with failures
            failure_mask = condition & ~data[var].isna() & ~data[var].isin(value)
            if (failure_sum := failure_mask.sum()) != 0:
                failure_count += failure_sum
                if failure_strategy == "drop":
                    data = data.loc[~failure_mask]

    # Warn if imputation is ignored because of several possible imputation values
    if len(imputation_warning) != 0:
        for var, count in imputation_warning.items():
            value = rule.equivalent[var] if isinstance(rule, EquivalenceRule) else rule.is_implied[var]
            warnings.warn(
                f"Failed {count} times to impute {var}, because there are several possible imputation values {value}."
            )

    return data, na_count, failure_count


def enforce_rules(
    data: pd.DataFrame,
    rules: list[Rule],
    na_strategy: Literal["impute", "error"] | None = "impute",
    failure_strategy: Literal["drop", "error"] | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Enforces rules by possibly imputing, dropping observations, or raising errors.

    data:
        The data to compare and impute.
    rules:
        List of rules to enforce.
    na_strategy:
        What to do when a variable in an observation is missing and can be imputed according to a rule.
    failure_strategy:
        What to do when an observation does not conform to a rule.
    verbose:
        If True prints a message for every element of rules, additionally to the final one.
    """
    data = data.copy()

    na_label = "imputed" if na_strategy == "impute" else "left NA"
    failure_label = "dropped" if failure_strategy == "drop" else "failed"

    na_count: int = 0
    failure_count: int = 0
    for rule in rules:
        # Enforce rule
        data, na_sum, failure_sum = _enforce_rule(data, rule, na_strategy, failure_strategy)
        na_count += na_sum
        failure_count += failure_sum

        # Print message or error for rule if necessary
        shortened_rule = textwrap.shorten(str(rule), width=85, placeholder="...")
        if verbose:
            print(f"{na_sum} {na_label}, {failure_sum} {failure_label}, used rule {shortened_rule}")
        error_sum = 0
        if failure_strategy == "error":
            error_sum += failure_sum
        if na_strategy == "error":
            error_sum += na_sum
        if error_sum != 0:
            raise AssertionError(f"Rule {shortened_rule} failed {error_sum} times.")

    # Print total message
    print(("\n" if verbose else "") + f"TOTAL: {na_count} {na_label}, {failure_count} {failure_label}")

    return data
