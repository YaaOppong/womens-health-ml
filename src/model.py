"""
model.py — Menstrual cycle-phase classification on the mcPHASES dataset.

Runs from the project root, reading ONLY data/feature_matrix.csv (no raw-table access):

    python src/model.py

Two experiments, both leave-one-subject-out (LOSO) with fixed hyperparameters:

  [A] CHANNEL FACTORIAL (2022 only, where all channels exist; ~Specht's cohort)
      Full factorial over three channels on identical rows:
        cycle day (days_since_bleed) | physiology (leaner wearable set) | self-report (symptom variability)
      Seven cells: each channel alone, each pair, all three. Answers the research question:
      the marginal value of each channel and which combination predicts phase best.

  [B] PHYSIOLOGY GENERALISATION (both intervals)
      Physiology-only across 2022, 2024, and both. Flow and symptoms exist only in 2022, so
      passive physiology is the only signal 2024 can contribute; this checks whether it
      replicates in the independent 2024 period.

WHY 2022-ONLY FOR THE FACTORIAL
  Flow (the bleed anchor) and symptoms are logged only in 2022, so cycle-day and self-report
  cannot be computed for 2024. days_since_bleed is reset per interval, so 2024 (no logged
  bleeding) correctly has no anchor. The factorial therefore runs on 2022.

GOVERNANCE
  Writes ONLY aggregate, non-identifying tables to results/, each through a guard that refuses
  participant-level objects. Figures are produced by results.py. The serialised model lands in
  gitignored data/models/, never committed.

VERIFY: FLOW_COL / FLOW_THRESHOLD define a bleeding day ("more than spotting" -> >= 2).
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
FEATURE_MATRIX = 'data/feature_matrix.csv'
RESULTS_DIR    = 'results/'
MODEL_DIR      = 'data/models/'

LABEL  = 'phase'
GROUP  = 'id'
PHASES = ['Menstrual', 'Follicular', 'Fertility', 'Luteal']

# bleeding-day definition for days_since_bleed (the cycle-day channel)
FLOW_COL       = 'flow_volume'
FLOW_THRESHOLD = 2                      # >= 2 -> Somewhat Light and up (more than spotting)
FLOW_ORD = {
    'not at all': 0, 'none': 0, 'no': 0, 'no flow': 0,
    'spotting / very light': 1, 'spotting': 1, 'very light': 1, 'trace': 1,
    'somewhat light': 2, 'light': 2, 'mild': 2,
    'moderate': 3, 'medium': 3, 'somewhat heavy': 3, 'normal': 3,
    'heavy': 4,
    'very heavy': 5,
}

# rolling window for trend features (~one cycle phase)
ROLL_WINDOWS = [7]

# leaner physiology: physiologically meaningful signals, standardised per participant,
# plus a 7-day rolling mean and std to capture the cycle trajectory and its variability.
PHYSIO_BASE = [
    'full_sleep_breathing_rate',                      # respiratory rate (rises in luteal)
    'resting_heart_rate',                             # RHR (rises in luteal)
    'nightly_temperature',                            # post-ovulatory temperature shift
    'baseline_relative_nightly_standard_deviation',   # nightly temperature variability
    'rmssd_mean',                                     # HRV (parasympathetic tone)
    'overall_score',                                  # sleep quality
]

# self-reported symptoms (2022 only); ordinal text -> number -> per-person rolling std
SYMPTOMS = ['appetite', 'exerciselevel', 'headaches', 'cramps', 'sorebreasts', 'fatigue',
            'sleepissue', 'moodswing', 'stress', 'foodcravings', 'indigestion', 'bloating']
SYMPTOM_WINDOW = 5
ORD = {
    'not at all': 0, 'none': 0, 'no': 0, 'never': 0,
    'very low': 1, 'very low/little': 1, 'very light': 1,
    'low': 2, 'mild': 2, 'light': 2, 'a little': 2,
    'moderate': 3, 'medium': 3, 'somewhat': 3, 'normal': 3,
    'high': 4, 'a lot': 4,
    'very high': 5, 'severe': 5, 'extreme': 5,
}


def make_clf():
    """Fixed hyperparameters (no tuning -> unbiased), matching the notebook."""
    return RandomForestClassifier(n_estimators=200, max_depth=None, max_features='sqrt',
                                  class_weight='balanced', random_state=42)


# ----------------------------------------------------------------------
# Feature engineering (operates only on columns already in the matrix)
# ----------------------------------------------------------------------
def _coerce_ordinal(series, mapping, label=''):
    """Coerce a numeric-or-ordinal-text column to numbers; report unmapped text."""
    num = pd.to_numeric(series, errors='coerce')
    text = series.astype(str).str.strip().str.lower().map(mapping)
    combined = num.where(num.notna(), text)
    unresolved = series.notna() & combined.isna()
    if unresolved.any():
        bad = sorted(series[unresolved].astype(str).str.strip().str.lower().unique())
        print(f'  WARNING unmapped {label} values (extend mapping):', bad[:12])
    return combined


def add_days_since_bleed(df):
    """Days since the most recent bleeding day (flow >= threshold), per id and interval.

    Derived from self-reported flow, so it is a self-report feature (the cycle-day channel),
    not a passive wearable one. Reset per interval: 2024 has no logged bleeding, so its rows
    get NaN and are excluded from any cycle-day model.
    """
    df = df.sort_values(['id', 'study_interval', 'day_in_study'])
    df['_flow'] = _coerce_ordinal(df[FLOW_COL], FLOW_ORD, 'flow')
    print('  bleeding days by interval:',
          df.assign(b=df['_flow'] >= FLOW_THRESHOLD).groupby('study_interval')['b'].sum().to_dict())

    def per_group(g):
        last = np.nan
        out = []
        for day, fl in zip(g['day_in_study'], g['_flow']):
            if pd.notna(fl) and fl >= FLOW_THRESHOLD:
                last = day
            out.append(np.nan if pd.isna(last) else day - last)
        return pd.Series(out, index=g.index)

    df['days_since_bleed'] = (df.groupby(['id', 'study_interval'], group_keys=False)
                                .apply(per_group))
    return df.drop(columns=['_flow'])


def add_zscores(df, cols):
    """Per-person standardisation (within-participant baseline)."""
    for c in cols:
        df[c + '_z'] = df.groupby('id')[c].transform(lambda s: (s - s.mean()) / s.std(ddof=0))
    return df


def add_rolling(df, cols, windows):
    """Per-person rolling mean and std over the given day windows."""
    df = df.sort_values(['id', 'study_interval', 'day_in_study'])
    for c in cols:
        grp = df.groupby(['id', 'study_interval'])[c]   # never roll across the interval boundary
        for w in windows:
            df[f'{c}_roll{w}_mean'] = grp.transform(lambda s: s.rolling(w, min_periods=2).mean())
            df[f'{c}_roll{w}_std'] = grp.transform(lambda s: s.rolling(w, min_periods=2).std())
    return df


def add_symptoms(df):
    """Ordinal symptom text -> number -> per-person rolling std (Specht's variability signal)."""
    df = df.sort_values(['id', 'study_interval', 'day_in_study'])
    for c in SYMPTOMS:
        df[c + '_num'] = _coerce_ordinal(df[c], ORD, c)
    for c in SYMPTOMS:
        df[c + '_rstd'] = df.groupby('id')[c + '_num'].transform(
            lambda s: s.rolling(SYMPTOM_WINDOW, min_periods=2).std())
    return df


def build():
    """Read the base matrix and apply all model-level transforms. No raw-table access."""
    print('Reading base feature matrix...', flush=True)
    df = pd.read_csv(FEATURE_MATRIX)
    print('Engineering model features...', flush=True)
    df = add_days_since_bleed(df)
    df = add_zscores(df, PHYSIO_BASE)
    df = add_rolling(df, PHYSIO_BASE, ROLL_WINDOWS)
    df = add_symptoms(df)
    return df.copy()


# ----------------------------------------------------------------------
# Channel / feature-set definitions
# ----------------------------------------------------------------------
def physiology_features():
    feats = [c + '_z' for c in PHYSIO_BASE]
    for c in PHYSIO_BASE:
        for w in ROLL_WINDOWS:
            feats += [f'{c}_roll{w}_mean', f'{c}_roll{w}_std']
    return feats


CYCLE_DAY        = ['days_since_bleed']
PHYSIOLOGY       = physiology_features()
SYMPTOM_FEATURES = [c + '_rstd' for c in SYMPTOMS]


def factorial_sets():
    """All seven non-empty combinations of the three channels."""
    return {
        'cycle_day':           CYCLE_DAY,
        'physiology':          PHYSIOLOGY,
        'symptoms':            SYMPTOM_FEATURES,
        'cycle+physiology':    CYCLE_DAY + PHYSIOLOGY,
        'cycle+symptoms':      CYCLE_DAY + SYMPTOM_FEATURES,
        'physiology+symptoms': PHYSIOLOGY + SYMPTOM_FEATURES,
        'all_three':           CYCLE_DAY + PHYSIOLOGY + SYMPTOM_FEATURES,
    }


# ----------------------------------------------------------------------
# LOSO evaluation + guarded output
# ----------------------------------------------------------------------
def loso(df, feats):
    data = df.dropna(subset=feats + [LABEL]).copy()
    X = data[feats].to_numpy()
    y = data[LABEL].to_numpy()
    g = data[GROUP].to_numpy()
    logo = LeaveOneGroupOut()
    y_true, y_pred, imp = [], [], []
    for tr, te in logo.split(X, y, groups=g):
        clf = make_clf()
        clf.fit(X[tr], y[tr])
        y_true.extend(y[te])
        y_pred.extend(clf.predict(X[te]))
        imp.append(clf.feature_importances_)
    importances = pd.Series(np.mean(imp, axis=0), index=feats).sort_values(ascending=False)
    return np.array(y_true), np.array(y_pred), importances, len(data)


def summarise(y_true, y_pred):
    rep = classification_report(y_true, y_pred, labels=PHASES, digits=3, output_dict=True)
    return {'accuracy': (y_true == y_pred).mean(),
            'macro_f1': f1_score(y_true, y_pred, average='macro'),
            'fertility_f1': rep['Fertility']['f1-score']}


def write_result_csv(obj, filename, index=True):
    """Guarded write to results/: refuses any object carrying a participant identifier."""
    names = set(map(str, getattr(obj, 'columns', [])))
    names |= {str(getattr(obj, 'name', '') or '')}
    names |= {str(getattr(getattr(obj, 'index', None), 'name', '') or '')}
    assert not ({'id', 'subject', GROUP} & names), (
        f"refusing to write participant-level data to results/: {filename} (columns/index {names})")
    obj.to_csv(os.path.join(RESULTS_DIR, filename), index=index)


def save_confusion(y_true, y_pred, name):
    cm = confusion_matrix(y_true, y_pred, labels=PHASES)
    write_result_csv(pd.DataFrame(cm, index=PHASES, columns=PHASES), f'confusion_{name}.csv')


def run_experiment(df, named_sets, exp_name, save_confusions=False):
    rows = []
    for label, feats in named_sets.items():
        yt, yp, imp, n = loso(df, feats)
        m = summarise(yt, yp)
        m.update({'model': label, 'n_features': len(feats), 'n_rows': n})
        rows.append(m)
        write_result_csv(imp.head(20), f'importance_{exp_name}_{label}.csv')
        if save_confusions:
            save_confusion(yt, yp, f'{exp_name}_{label}')
        print(f'  {label:20s} macro-F1 {m["macro_f1"]:.3f} | Fert {m["fertility_f1"]:.3f} '
              f'| acc {m["accuracy"]:.3f} | n_feat {len(feats)} | rows {n}')
    out = pd.DataFrame(rows)[['model', 'n_features', 'n_rows', 'accuracy', 'macro_f1', 'fertility_f1']]
    write_result_csv(out, f'metrics_{exp_name}.csv', index=False)
    return out


# ----------------------------------------------------------------------
# Experiments
# ----------------------------------------------------------------------
def experiment_factorial_2022(df):
    print('\n[A] Channel factorial (2022, same rows) -------------------------')
    sets = factorial_sets()
    allf = sorted(set(sum(sets.values(), [])))
    df22 = df[df['study_interval'] == 2022].dropna(subset=allf + [LABEL]).copy()
    print(f'  rows {len(df22)} | participants {df22[GROUP].nunique()}')
    return run_experiment(df22, sets, 'factorial_2022', save_confusions=True)


def experiment_physiology_generalization(df):
    print('\n[B] Physiology-only generalisation across intervals -------------')
    phys = physiology_features()
    rows = []
    for label, sub in [('both', df),
                        ('2022', df[df['study_interval'] == 2022]),
                        ('2024', df[df['study_interval'] == 2024])]:
        yt, yp, imp, n = loso(sub, phys)
        m = summarise(yt, yp); m.update({'subset': label, 'n_rows': n})
        rows.append(m)
        print(f'  physiology {label:5s} macro-F1 {m["macro_f1"]:.3f} | '
              f'Fert {m["fertility_f1"]:.3f} | acc {m["accuracy"]:.3f} | rows {n}')
    out = pd.DataFrame(rows)[['subset', 'n_rows', 'accuracy', 'macro_f1', 'fertility_f1']]
    write_result_csv(out, 'metrics_physiology_generalization.csv', index=False)
    return out


# ----------------------------------------------------------------------
# Deployable artifact (passive model, gesture not a service)
# ----------------------------------------------------------------------
def save_final_model(df, feats=None):
    """Serialise the passive physiology model trained on all complete data (gitignored)."""
    import joblib
    feats = feats or physiology_features()
    data = df.dropna(subset=feats + [LABEL])
    clf = make_clf().fit(data[feats].to_numpy(), data[LABEL].to_numpy())
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, 'phase_rf.joblib')
    joblib.dump({'model': clf, 'features': feats, 'phases': PHASES}, path)
    print(f'\nSerialised passive model -> {path} (gitignored; trained on restricted data)')


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = build()
    print(f'\nEngineered frame: {len(df)} participant-days, {df[GROUP].nunique()} participants')
    experiment_factorial_2022(df)
    experiment_physiology_generalization(df)
    save_final_model(df)
    print(f'\nDone. Aggregate outputs written to {RESULTS_DIR}')


if __name__ == '__main__':
    main()
