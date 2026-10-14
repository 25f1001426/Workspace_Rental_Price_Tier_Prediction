import numpy as np
import pandas as pd
train = pd.read_csv('/kaggle/input/competitions/workstation-rental-price-tier-prediction/kaggle_ready_dataset/train.csv')
test = pd.read_csv('/kaggle/input/competitions/workstation-rental-price-tier-prediction/kaggle_ready_dataset/test.csv')
print(train.shape)
print(test.shape)
train.head()
print(train.dtypes)
identifier = 'space_id'
target = "price_tier"
numerical_cols = ['operator_platform_age_years','operator_platform_age_months','operator_active_years','operator_active_months','operator_portfolio_size','geo_lat',
    'geo_lon','utility_rooms','zones','workstations','min_term_days','max_term_days','min_term_floor','min_term_ceiling','max_term_floor', 'max_term_ceiling',
    'min_term_avg','max_term_avg','open_slots_a','open_slots_b','open_slots_c','open_slots_d','open_slots_e','review_count','review_count_12m','review_count_30d',
    'review_count_prev_yr','booked_load','score_overall','score_accuracy','score_condition','score_onboarding','score_support','score_access','score_value',
    'operator_metro_units','operator_metro_whole','operator_metro_private','operator_metro_shared','reviews_per_month','max_seats']
categorical_cols = ['operator_ref','operator_class', 'operator_certified','operator_has_avatar','operator_id_verified','zone','borough','space_class',
    'access_type','facilities','bookable','compliance_status','metro']
datetime_cols = ['snapshot_start_date','snapshot_end_date','first_review_date','last_review_date']

# missing values in traing set
missing = train.isnull().sum()

missing_summary = pd.DataFrame({
    "missing_count": missing,
    "missing_percentage": (missing / len(train)) * 100
})

#columns that contain missing values
missing_summary = missing_summary[
    missing_summary["missing_count"] > 0
].sort_values("missing_percentage", ascending=False)

display(missing_summary)

# missing values in test dataset

test_missing = test.isnull().sum()

test_missing_summary = pd.DataFrame({
    "missing_count": test_missing,
    "missing_percentage": (test_missing / len(test)) * 100
})

test_missing_summary = test_missing_summary[
    test_missing_summary["missing_count"] > 0
].sort_values("missing_percentage", ascending=False)

display(test_missing_summary)

#  duplicate rows
print("Duplicate rows in train:", train.duplicated().sum())
print("Duplicate rows in test:", test.duplicated().sum())

# duplicate workspace IDs
print("Duplicate space_id in train:", train["space_id"].duplicated().sum())
print("Duplicate space_id in test:", test["space_id"].duplicated().sum())


import matplotlib.pyplot as plt
import seaborn as sns
print(train["price_tier"].value_counts().sort_index())
plt.figure(figsize=(8, 5))
sns.countplot(data=train, x="price_tier", hue="price_tier", legend=False)
plt.title("Distribution of Workspace Price Tiers")
plt.xlabel("Price Tier")
plt.ylabel("Number of Listings")
plt.show()
print("\nPercentage distribution:")
print((train["price_tier"].value_counts(normalize=True)
       .sort_index() * 100).round(2))
