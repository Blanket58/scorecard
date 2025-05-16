import numpy as np
import pandas as pd


def _ab(points0, odds0, pdo):
    b = pdo / np.log(2)
    a = points0 + b * np.log(odds0)
    return a, b


def prob2score(y_proba, points0=600, odds0=1/19, pdo=50):
    a, b = _ab(points0, odds0, pdo)
    return np.round(a - b * np.log(y_proba / (1 - y_proba)))


def scorecard(bins, intercept, coef, feature_names, points0=600, odds0=1/19, pdo=50):
    """
    对Dataframe按列分箱，输出分箱结果

    Parameters
    ----------
    bins : dict[str, DataFrame]
        bins数据框
    intercept: float
        截矩项的系数
    coef : array-like of shape (n_samples,)
        各个自变量的系数
    feature_names : array-like of shape (n_samples,)
        与系数对应位置各个自变量的名称
    points0 : int
        基准分
    odds0 : float
        基准分时的好坏比
    pdo : int
        odds每增加一倍，分数增长的值

    Returns
    -------
    DataFrame
        评分卡
    """
    a, b = _ab(points0, odds0, pdo)
    dt = pd.concat(bins, axis=0, ignore_index=True)
    dt['coef'] = dt['variable'].map({x: y for x, y in zip(feature_names, coef)})
    dt['score'] = - np.round(b * dt['coef'] * dt['woe'])
    return pd.concat([
        pd.DataFrame({'variable': 'Base Line', 'bin': None, 'score': np.round(a - b * intercept)}),
        dt[['variable', 'bin', 'score']]
    ], axis=0, ignore_index=True)
