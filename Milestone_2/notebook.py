#1


target = "price_tier"
identifier = "space_id"

date_cols = [
    "snapshot_start_date",
    "snapshot_end_date",
    "first_review_date",
    "last_review_date"
]

special_cols = ["facilities", "operator_ref"]

excluded_cols = [target, identifier, "facilities", "operator_ref"]

feature_cols = [
    c for c in train.columns
    if c not in excluded_cols + date_cols
]

numerical_cols = train[feature_cols].select_dtypes(
    include=["number"]
).columns.tolist()

categorical_cols = train[feature_cols].select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

low_cardinality_cols = [
    c for c in categorical_cols
    if train[c].nunique(dropna=True) <= 20
]

high_cardinality_cols = [
    c for c in categorical_cols
    if train[c].nunique(dropna=True) > 20
]

numeric_missing_cols = [
    c for c in numerical_cols
    if train[c].isna().any()
]

print("Numerical columns:", numerical_cols)
print("\nNumerical columns with missing values:",
      numeric_missing_cols)
print("\nLow-cardinality categorical columns:",
      low_cardinality_cols)
print("\nHigh-cardinality categorical columns:",
      high_cardinality_cols)
print("\nDate columns:", date_cols)
print("\nSpecial columns:", special_cols)
print("\nExcluded from baseline:", excluded_cols)

#3
#example for just 1 column to show how imputation works
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

col = "operator_portfolio_size"

original = train[col].copy()

imputer = SimpleImputer(strategy="median")
imputed = imputer.fit_transform(
    train[[col]]
).ravel()

print("Missing before:", original.isna().sum())
print("Missing after:", pd.isna(imputed).sum())
print("Original median:", original.median())
print("Imputed median:", pd.Series(imputed).median())
print("Original standard deviation:", original.std())
print("Imputed standard deviation:", pd.Series(imputed).std())

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

original.dropna().hist(bins=40, ax=axes[0])
axes[0].set_title("Before Imputation")

pd.Series(imputed).hist(bins=40, ax=axes[1])
axes[1].set_title("After Median Imputation")

plt.tight_layout()
plt.show()

#4

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
outlier_cols = [
    "operator_portfolio_size",
    "review_count",
    "booked_load",
    "max_term_days"
]

comparison_rows = []

for col in outlier_cols:
    s = train[col].dropna()

    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    comparison_rows.append({
        "feature": col,
        "original_min": s.min(),
        "original_median": s.median(),
        "original_max": s.max(),
        "IQR_lower": lower,
        "IQR_upper": upper,
        "IQR_outlier_count": ((s < lower) | (s > upper)).sum()
    })

display(pd.DataFrame(comparison_rows).set_index("feature"))
fig, axes = plt.subplots(
    len(outlier_cols), 2, figsize=(12, 4 * len(outlier_cols))
)

for i, col in enumerate(outlier_cols):
    s = train[col].dropna()

    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1

    lower = max(0, q1 - 1.5 * iqr) if s.min() >= 0 else q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    capped = s.clip(lower=lower, upper=upper)

    axes[i, 0].boxplot(s)
    axes[i, 0].set_title(f"{col}: Original")

    axes[i, 1].boxplot(capped)
    axes[i, 1].set_title(f"{col}: Diagnostic IQR Capping")

plt.tight_layout()
plt.show()

#5

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numerical_cols),
], remainder="drop")
