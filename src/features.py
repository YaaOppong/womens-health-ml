"""
features.py — Build the per-day base feature matrix for menstrual cycle phase
prediction from the mcPHASES dataset.

Usage (from the project root):
    python src/features.py

This is the single owner of all raw-table extraction. It produces one row per
participant-day combining:
  - wearable physiological signals (high-frequency signals aggregated to daily
    mean/min/max/std; already-daily tables collapsed to a daily mean),
  - nightly wrist temperature (from the computed_temperature table),
  - self-reported symptoms (2022 only) and bleed flow,
  - the hormonally-derived cycle-phase label and the study interval.

The physiological signals are the modelling features; temperature, symptoms, and
flow are carried as NULLABLE extras. The complete-case filter is applied to the
physiological features and the phase label ONLY, so the symptom and temperature
columns (sparse / 2022-only) do not shrink the matrix. All model-level
transforms (per-person z-scoring, rolling features, days_since_bleed, symptom
encoding) live downstream in model.py, which reads only this matrix and never
re-opens the raw tables.
"""

import os
import pandas as pd


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
DATA_DIR = ('data/mcphases-a-dataset-of-physiological-hormonal-and-self-'
            'reported-events-and-symptoms-for-menstrual-health-tracking-'
            'with-wearables-1.0.0/')

OUTPUT_PATH = 'data/feature_matrix.csv'   # derived participant-level data; gitignored

KEYS = ['id', 'day_in_study']
LABELS = ['phase', 'study_interval']

# High-frequency signals aggregated to daily summary statistics:
#   {output_prefix: (source_table, source_column)}
SIGNALS = {
    'temperature_diff_from_baseline': ('wrist_temperature', 'temperature_diff_from_baseline'),
    'rmssd':          ('heart_rate_variability_details', 'rmssd'),
    'high_frequency': ('heart_rate_variability_details', 'high_frequency'),
    'low_frequency':  ('heart_rate_variability_details', 'low_frequency'),
    'altitude':       ('altitude', 'altitude'),
}

# Already-daily tables merged directly after a daily-mean collapse:
#   {source_table: [columns_to_take]}
DAILY_TABLES = {
    'demographic_vo2_max': ['filtered_demographic_vo2_max'],
    'sleep_score': ['overall_score', 'revitalization_score',
                    'deep_sleep_in_minutes', 'resting_heart_rate', 'restlessness'],
    'respiratory_rate_summary': ['full_sleep_breathing_rate', 'deep_sleep_breathing_rate',
                                 'light_sleep_breathing_rate', 'rem_sleep_breathing_rate'],
}

# Nightly temperature. This table has no day_in_study; each night is attributed
# to the wake day (basal-temperature convention) and collapsed to a daily mean.
TEMP_TABLE = 'computed_temperature'
TEMP_DAY_FROM = 'sleep_end_day_in_study'
TEMP_RAW = ['nightly_temperature', 'baseline_relative_nightly_standard_deviation']

# Self-reported symptoms (2022 only); kept as raw text, encoded downstream.
SYMPTOMS = ['appetite', 'exerciselevel', 'headaches', 'cramps', 'sorebreasts', 'fatigue',
            'sleepissue', 'moodswing', 'stress', 'foodcravings', 'indigestion', 'bloating']

# Bleed flow for the downstream days_since_bleed feature.
# VERIFY: the project notes define days_since_bleed from `flow_volume`; if your
# column is named differently (e.g. flow_numeric) or lives in another table, change these.
FLOW_TABLE = 'hormones_and_selfreport'
FLOW_COL = 'flow_volume'

# Final physiological feature columns (the modelling features; complete-case applies here).
FINAL_FEATURES = [
    # Temperature (skin temperature deviation from personal baseline)
    'temperature_diff_from_baseline_mean', 'temperature_diff_from_baseline_min',
    'temperature_diff_from_baseline_max', 'temperature_diff_from_baseline_std',
    # Resting heart rate (cross-cohort; replaces cohort-blocked beat-level bpm)
    'resting_heart_rate',
    # Heart rate variability
    'rmssd_mean', 'rmssd_min', 'rmssd_max', 'rmssd_std',
    'high_frequency_mean', 'low_frequency_mean',
    # Sleep / respiratory
    'deep_sleep_in_minutes', 'full_sleep_breathing_rate', 'deep_sleep_breathing_rate',
    'rem_sleep_breathing_rate', 'light_sleep_breathing_rate', 'restlessness',
    'overall_score', 'revitalization_score',
    # Context covariates (used in with/without comparison)
    'altitude_mean', 'filtered_demographic_vo2_max',
]

# Nullable extras carried alongside the modelling features (NOT complete-cased).
EXTRAS = TEMP_RAW + SYMPTOMS + [FLOW_COL]


# ----------------------------------------------------------------------
# Build functions
# ----------------------------------------------------------------------
def load_tables(data_dir):
    """Load all CSV tables from the dataset directory into a dict."""
    tables = {}
    for filename in os.listdir(data_dir):
        if filename.endswith('.csv'):
            name = filename[:-4]
            print(f"  loading {filename}...", flush=True)
            tables[name] = pd.read_csv(os.path.join(data_dir, filename))
    if not tables:
        raise FileNotFoundError(
            f"No CSV files found in {data_dir!r}. "
            "Check the path (PhysioNet downloads may nest files in a subfolder)."
        )
    return tables


def aggregate_daily(df, signal_col):
    """Aggregate a high-frequency signal to per-day mean/min/max/std."""
    agg = df.groupby(KEYS)[signal_col].agg(['mean', 'min', 'max', 'std']).reset_index()
    agg.columns = KEYS + [f'{signal_col}_{s}' for s in ['mean', 'min', 'max', 'std']]
    return agg


def aggregate_daily_mean(df, cols):
    """Collapse multiple per-day rows (timestamped sleep sessions, repeated VO2 max
    estimates) to a single daily mean for each requested column.

    Several nominally-daily tables in fact contain multiple rows per participant-day.
    Merging them directly multiplies rows, so they must be collapsed to one row per
    (id, day_in_study) first.
    """
    return df.groupby(KEYS)[cols].mean().reset_index()


def aggregate_temperature(df, day_from, cols):
    """Aggregate the computed_temperature table.

    This table has no day_in_study column; each night is attributed to the wake day
    (sleep_end_day_in_study) and then collapsed to one row per (id, day_in_study).
    day_in_study is unique per participant across intervals, so KEYS suffices.
    """
    t = df.copy()
    t['day_in_study'] = t[day_from]
    return t.groupby(KEYS)[cols].mean().reset_index()


def build_feature_matrix(data_tables):
    """Merge daily physiological aggregates, nightly temperature, labels, symptoms, and flow."""
    matrix = None

    # High-frequency signals -> daily aggregates
    for name, (table, col) in SIGNALS.items():
        print(f"  aggregating {name} from {table} "
              f"({len(data_tables[table]):,} rows)...", flush=True)
        daily = aggregate_daily(data_tables[table], col)
        matrix = daily if matrix is None else matrix.merge(daily, on=KEYS, how='outer')

    # Already-daily tables -> daily mean (these contain multiple rows per id-day)
    for table, cols in DAILY_TABLES.items():
        print(f"  aggregating {table} to daily mean...", flush=True)
        daily = aggregate_daily_mean(data_tables[table], cols)
        matrix = matrix.merge(daily, on=KEYS, how='left')

    # Nightly temperature -> wake-day key -> daily mean
    print("  aggregating nightly temperature...", flush=True)
    temp = aggregate_temperature(data_tables[TEMP_TABLE], TEMP_DAY_FROM, TEMP_RAW)
    matrix = matrix.merge(temp, on=KEYS, how='left')

    # Hormonally-derived phase labels, study interval, and self-reported symptoms
    print("  merging phase labels and symptoms...", flush=True)
    selfreport = data_tables['hormones_and_selfreport'][KEYS + LABELS + SYMPTOMS]
    matrix = matrix.merge(selfreport, on=KEYS, how='left')

    # Bleed flow (for downstream days_since_bleed)
    print("  merging bleed flow...", flush=True)
    flow = data_tables[FLOW_TABLE][KEYS + [FLOW_COL]]
    matrix = matrix.merge(flow, on=KEYS, how='left')

    return matrix[KEYS + FINAL_FEATURES + EXTRAS + LABELS]


def main():
    print("Loading tables...", flush=True)
    data_tables = load_tables(DATA_DIR)

    print("Building feature matrix...", flush=True)
    features = build_feature_matrix(data_tables)

    # Guard against row-multiplication from merges: each (id, day_in_study) must
    # appear exactly once. (A multi-row-per-day source table merged without
    # aggregation previously inflated this and scrambled feature/label alignment.)
    max_per_day = features.groupby(KEYS).size().max()
    assert max_per_day == 1, (
        f"Feature matrix has up to {max_per_day} rows per (id, day_in_study); "
        "a source table was merged without daily aggregation."
    )

    print(f"Full matrix:  {features.shape[0]} participant-days, "
          f"{features['id'].nunique()} participants")

    # Keep every participant-day that has a phase label. Feature-completeness is handled
    # downstream per experiment in model.py (each model drops rows missing its own features),
    # so the base matrix imposes no single complete-case rule and the leaner model is not
    # restricted by missingness in signals it does not use. Both intervals are retained.
    model_ready = features.dropna(subset=['phase'])
    print(f"Labelled rows: {model_ready.shape[0]} participant-days, "
          f"{model_ready['id'].nunique()} participants")
    print(model_ready['phase'].value_counts())

    # Coverage of the nullable extras, for transparency.
    print("\nNullable-extra coverage (fraction of model-ready rows):")
    print((model_ready[EXTRAS].notna().mean()).round(2))

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    model_ready.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved -> {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
