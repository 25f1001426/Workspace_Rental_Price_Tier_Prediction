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

