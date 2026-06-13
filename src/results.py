"""
results.py — Render figures from the aggregate outputs of model.py.

Runs from the project root, after model.py:
    python src/results.py

This is the presentation layer. It reads ONLY the aggregate CSVs in results/
(metrics_*, confusion_*, importance_*) and writes figures back to results/.
It never reads the feature matrix or any raw / participant-level data, so by
construction it cannot expose participant-level information. All modelling and
all participant-level handling live in model.py and features.py.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = 'results/'


def _path(name):
    return os.path.join(RESULTS_DIR, name)


def _stem(path, prefix):
    return os.path.basename(path)[len(prefix):-len('.csv')]


def plot_confusion(csv_path):
    """Heatmap of an aggregate confusion-matrix counts table."""
    name = _stem(csv_path, 'confusion_')
    cm = pd.read_csv(csv_path, index_col=0)
    fig, ax = plt.subplots(figsize=(4, 3.5))
    im = ax.imshow(cm.values, cmap='Blues')
    ax.set_xticks(range(cm.shape[1])); ax.set_xticklabels(cm.columns, rotation=45, ha='right')
    ax.set_yticks(range(cm.shape[0])); ax.set_yticklabels(cm.index)
    ax.set_xlabel('predicted'); ax.set_ylabel('true'); ax.set_title(name)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm.values[i, j], ha='center', va='center', fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(_path(f'fig_confusion_{name}.png'), dpi=150)
    plt.close(fig)


def plot_metrics(csv_path):
    """Grouped bar chart comparing models (or subsets) within one experiment."""
    exp = _stem(csv_path, 'metrics_')
    m = pd.read_csv(csv_path)
    label_col = next((c for c in ('model', 'subset') if c in m.columns), m.columns[0])
    x = np.arange(len(m)); w = 0.38
    fig, ax = plt.subplots(figsize=(1.7 * len(m) + 2, 3.8))
    ax.bar(x - w / 2, m['macro_f1'], w, label='macro-F1')
    ax.bar(x + w / 2, m['fertility_f1'], w, label='Fertility F1')
    ax.set_xticks(x); ax.set_xticklabels(m[label_col], rotation=30, ha='right')
    ax.set_ylabel('F1'); ax.set_ylim(0, 1); ax.set_title(f'{exp}: model comparison')
    ax.legend()
    fig.tight_layout()
    fig.savefig(_path(f'fig_metrics_{exp}.png'), dpi=150)
    plt.close(fig)


def plot_importance(csv_path):
    """Horizontal bar chart of LOSO-averaged feature importances."""
    name = _stem(csv_path, 'importance_')
    imp = pd.read_csv(csv_path, index_col=0).iloc[:, 0].sort_values()
    fig, ax = plt.subplots(figsize=(6, 0.3 * len(imp) + 1.2))
    ax.barh(imp.index.astype(str), imp.values)
    ax.set_xlabel('mean importance (LOSO-averaged)'); ax.set_title(name)
    fig.tight_layout()
    fig.savefig(_path(f'fig_importance_{name}.png'), dpi=150)
    plt.close(fig)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for p in sorted(glob.glob(_path('confusion_*.csv'))):
        plot_confusion(p)
    for p in sorted(glob.glob(_path('metrics_*.csv'))):
        plot_metrics(p)
    for p in sorted(glob.glob(_path('importance_*.csv'))):
        plot_importance(p)
    print(f'Figures written to {RESULTS_DIR}')


if __name__ == '__main__':
    main()
