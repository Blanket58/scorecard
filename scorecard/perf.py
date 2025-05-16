import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import auc, precision_recall_curve, precision_score, recall_score, roc_curve


def perf_eva(y_true, y_proba, title=None):
    """
    绘制KS, ROC, PR, PR vs. threshold图

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels.
    y_proba: array-like of shape (n_samples,)
        Target scores, probability estimates of the positive class.
    title : str
        Sup title for the figure.

    Returns
    -------
    None
    """
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, figsize=(20, 5))
    _plot_ks(y_true, y_proba, ax1)
    _plot_roc(y_true, y_proba, ax2)
    _plot_pr(y_true, y_proba, ax3)
    _plot_prt(y_true, y_proba, ax4)
    if title:
        fig.suptitle(title, fontweight='bold')
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def _plot_ks(y_true, y_proba, ax):
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    x = [(x+1) / tpr.size for x in range(tpr.size)]
    ax.plot(x, tpr, linestyle=':', linewidth=1, label='TPR', alpha=0.8)
    ax.plot(x, fpr, linestyle=':', linewidth=1, label='FPR', alpha=0.8)
    ax.plot(x, tpr-fpr, color='#EA3F25', linewidth=1)
    max_ks = (tpr - fpr).max()
    max_ks_x = (np.argmax(tpr - fpr) + 1) / tpr.size
    best_threshold = thresholds[np.argmax(tpr - fpr)]
    ax.axvline(x=max_ks_x, ymin=0, ymax=max_ks, color='#EA3F25', linestyle='--', linewidth=1)
    ax.scatter(max_ks_x, max_ks, color='#EA3F25')
    ax.annotate(f'K-S = {max_ks:.4f}', (max_ks_x, max_ks), (0, 2), textcoords='offset fontsize')
    ax.annotate(f'threshold = {best_threshold:.4f}', (max_ks_x, max_ks), (0, 1), textcoords='offset fontsize')
    ax.set_xlabel('% of population')
    ax.set_ylabel('Cumulative Rate')
    ax.set_title('K-S', loc='left')
    ax.legend(loc='best')
    ax.grid(visible=True, alpha=0.3)
    ax.margins(x=0, y=0)


def _plot_roc(y_true, y_proba, ax):
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, linewidth=1)
    ax.axline((0, 0), (1, 1), linestyle=':', linewidth=1, alpha=0.8)
    best_index = np.argmax(tpr - fpr)
    best_x = fpr[best_index]
    best_y = tpr[best_index]
    best_threshold = thresholds[best_index]
    ax.scatter(best_x, best_y)
    ax.annotate(f'AUC = {roc_auc:.4f}', (best_x, best_y), (1, -1), textcoords='offset fontsize')
    ax.annotate(f'threshold = {best_threshold:.4f}', (best_x, best_y), (1, -2), textcoords='offset fontsize')
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_title('ROC', loc='left')
    ax.grid(visible=True, alpha=0.3)
    ax.margins(x=0, y=0)


def _plot_pr(y_true, y_proba, ax):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    ax.plot(recall, precision, linewidth=1)
    ax.axline((0, 1), (1, 0), linestyle=':', linewidth=1, alpha=0.8)
    best_threshold = thresholds[np.argmax(tpr - fpr)]
    best_y_pred = np.where(y_proba > best_threshold, 1, 0)
    best_precision = precision_score(y_true, best_y_pred)
    best_recall = recall_score(y_true, best_y_pred)
    ax.scatter(best_recall, best_precision)
    ax.annotate(f'threshold = {best_threshold:.4f}', (best_recall, best_precision), (1, -1), textcoords='offset fontsize')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('P-R', loc='left')
    ax.grid(visible=True, alpha=0.3)
    ax.margins(x=0, y=0)


def _plot_prt(y_true, y_proba, ax):
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    thresholds = np.concatenate(([0], thresholds, [1]))
    ax.plot(thresholds, np.insert(precision, 0, 0), linewidth=1, label='Precision')
    ax.plot(thresholds, np.insert(recall, 0, 1), linewidth=1, label='Recall')
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    best_threshold = thresholds[np.argmax(tpr - fpr)]
    ax.axvline(x=best_threshold, ymin=0, ymax=1, color='#EA3F25', linestyle='--', linewidth=1)
    best_y_pred = np.where(y_proba > best_threshold, 1, 0)
    best_precision = precision_score(y_true, best_y_pred)
    best_recall = recall_score(y_true, best_y_pred)
    ax.scatter(best_threshold, best_precision)
    ax.scatter(best_threshold, best_recall)
    ax.annotate(f'Precision = {best_precision:.4f}', (best_threshold, best_precision), (1, -1), textcoords='offset fontsize')
    ax.annotate(f'Recall = {best_recall:.4f}', (best_threshold, best_recall), (1, -1), textcoords='offset fontsize')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Score')
    ax.set_title('P-R vs. Threshold', loc='left')
    ax.legend(loc='best')
    ax.grid(visible=True, alpha=0.3)
    ax.margins(x=0, y=0)


def gains_table(label, score, bin_num=10, precision=0):
    """
    Creates a data frame including distribution of total, negative, positive, positive rate and lift by score bins.
    The gains table is used in conjunction with financial and operational considerations to make cutoff decisions.

    Parameters
    ----------
    label : dict[str, array-like of shape (n_samples,)]
        True binary labels for each data samples.
    score: dict[str, array-like of shape (n_samples,)]
        Target scores or probability estimates of the positive class, for each data samples.
    bin_num : int
        The number of score bins.
    precision : int
        The precision at which to store and display the bins labels.

    Returns
    -------
    DataFrame
        Gains table.
    """
    assert label.keys() == score.keys(), '字典key未对齐'
    bkey, blabel = next(iter(label.items()))
    bscore = score.get(bkey)
    original_intervals = pd.qcut(bscore, q=bin_num, precision=precision).categories
    intervals = []
    for index, interval in enumerate(original_intervals):
        left = -np.inf if index == 0 else interval.left
        right = np.inf if index == len(original_intervals) - 1 else interval.right
        intervals.append(pd.Interval(left=left, right=right, closed='left'))
    result = []
    for k, l in label.items():
        s = score.get(k)
        dt = pd.DataFrame({'y': l})
        dt['bin'] = pd.cut(s, bins=intervals)
        dt = dt.groupby('bin', observed=False).agg(
            count=pd.NamedAgg(column='y', aggfunc='count'),
            pos=pd.NamedAgg(column='y', aggfunc='sum')
        ).reset_index()
        dt['dataset'] = k
        dt['count_distr'] = dt['count'] / dt['count'].sum()
        dt['neg'] = dt['count'] - dt['pos']
        dt['posprob'] = dt['pos'] / dt['count']
        dt['cum_neg'] = dt['neg'].cumsum()
        dt['cum_pos'] = dt['pos'].cumsum()
        dt['lift'] = dt['posprob'] / (dt['pos'].sum() / dt['count'].sum())
        result.append(dt[['dataset', 'bin', 'count', 'count_distr', 'neg', 'pos', 'cum_neg', 'cum_pos', 'posprob', 'lift']])
    return pd.concat(result, axis=0, ignore_index=True)


def perf_psi(label, score):
    """
    计算PSI并绘图

    Parameters
    ----------
    label : dict[str, array-like of shape (n_samples,)]
        True binary labels for actual and expected data samples.
    score: dict[str, array-like of shape (n_samples,)]
        Target scores, probability estimates of the positive class, for actual and expected data samples.

    Returns
    -------
    None
    """
    assert label.keys() == score.keys(), '字典key未对齐'
    assert len(label.keys()) == 2, '字典key数量不等于2'
    expected, actual = label.keys()

    binned_expected = pd.cut(score[expected], bins=10, right=False, precision=0, include_lowest=True)
    original_intervals = binned_expected.categories
    intervals = []
    for index, interval in enumerate(original_intervals):
        left = -np.inf if index == 0 else interval.left
        right = np.inf if index == len(original_intervals) - 1 else interval.right
        intervals.append(pd.Interval(left=left, right=right, closed='left'))
    binned_expected = binned_expected.rename_categories(intervals)
    dt_expected = pd.DataFrame(
        {'bin': binned_expected, 'label': label[expected]}
    ).groupby('bin', observed=True).agg(
        count=pd.NamedAgg(column='label', aggfunc='count'),
        pos=pd.NamedAgg(column='label', aggfunc='sum')
    ).assign(
        count_distr=lambda x: x['count'] / x['count'].sum(),
        posprob=lambda x: x['pos'] / x['count']
    ).reset_index()

    binned_actual = pd.cut(score[actual], bins=intervals)
    dt_actual = pd.DataFrame(
        {'bin': binned_actual, 'label': label[actual]}
    ).groupby('bin', observed=False).agg(
        count=pd.NamedAgg(column='label', aggfunc='count'),
        pos=pd.NamedAgg(column='label', aggfunc='sum')
    ).assign(
        count_distr=lambda x: x['count'] / x['count'].sum(),
        posprob=lambda x: x['pos'] / x['count']
    ).reset_index()

    psi = ((dt_actual['count_distr'] - dt_expected['count_distr']) * np.log(dt_actual['count_distr'].map(lambda x: 0.00000001 if x == 0 else x) / dt_expected['count_distr'])).sum()

    fig, ax1 = plt.subplots(figsize=(10, 5))
    index = np.arange(10)
    ax1.bar(index - 0.2, dt_expected['count_distr'], width=0.4, label=expected, color='#E77D72')
    ax1.bar(index + 0.2, dt_actual['count_distr'], width=0.4, label=actual, color='#56BCC2')
    ax1.set_ylabel('Score distribution')
    ax1.set_title(f'PSI: {psi:.4f}', loc='left')
    ax1.legend(loc='upper right', title='Distribution', frameon=False)
    ax1.grid(visible=True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(index, dt_expected['posprob'], marker='o', markerfacecolor='none', markersize=4, linewidth=1, color='blue', label=expected)
    ax2.plot(index, dt_actual['posprob'], marker='o', markerfacecolor='none', markersize=4, linewidth=1, linestyle=':', color='blue', label=actual)
    ax2.set_ylabel('Positive probability')
    ax2.legend(loc='upper right', title='Probability', frameon=False, bbox_to_anchor=(1, 0.85))
    ax1.set_xticks(index, dt_expected['bin'].astype(str).str.replace(r'\.0| ', '', regex=True))
    fig.tight_layout()
    plt.show()
    plt.close(fig)
