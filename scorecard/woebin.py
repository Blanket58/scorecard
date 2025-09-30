from abc import ABC, abstractmethod
from functools import reduce

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from matplotlib import pyplot as plt
from numba import njit
from scipy.stats import chi2
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier


class EnumMixin:

    @staticmethod
    def _fit_enum(X, y):
        boundary = [tuple(x) for x in X.unique().reshape(-1, 1).tolist()]
        return boundary

    @staticmethod
    def _transform_enum(X, y, boundary):
        y.rename('y', inplace=True)
        dt = pd.concat([X, y], axis=1)
        mapping = {category: group for group in boundary for category in group}
        dt['bin'] = dt.iloc[:, 0].map(mapping)
        dt = dt.groupby('bin', observed=True, dropna=False).agg(
            count=pd.NamedAgg(column='y', aggfunc='count'),
            pos=pd.NamedAgg(column='y', aggfunc='sum')
        ).reset_index()
        dt['variable'] = X.name
        dt['count_distr'] = dt['count'] / dt['count'].sum()
        dt['neg'] = dt['count'] - dt['pos']
        dt['posprob'] = dt['pos'] / dt['count']
        dt['woe'] = np.log((dt['pos'].map(lambda x: 0.00000001 if x == 0 else x) / dt['pos'].sum()) / (dt['neg'].map(lambda x: 0.00000001 if x == 0 else x) / dt['neg'].sum()))
        dt['bin_iv'] = (dt['pos'] / dt['pos'].sum() - dt['neg'] / dt['neg'].sum()) * dt['woe']
        dt['total_iv'] = dt['bin_iv'].sum()
        dt = dt[['variable', 'bin', 'count', 'count_distr', 'neg', 'pos', 'posprob', 'woe', 'bin_iv', 'total_iv']]
        dt.sort_values(by='posprob', ascending=True, ignore_index=True, inplace=True)
        return dt

    @classmethod
    def enum(cls, X, y, **kwargs):
        boundary = cls._fit_enum(X, y)
        return cls._transform_enum(X, y, boundary)


class CategoryMixin(ABC, EnumMixin):

    @staticmethod
    @abstractmethod
    def _fit_category(X, y, **kwargs):
        pass

    @classmethod
    def _transform_category(cls, X, y, boundary):
        return cls._transform_enum(X, y, boundary)

    @classmethod
    def category(cls, X, y, **kwargs):
        boundary = cls._fit_category(X, y, **kwargs)
        return cls._transform_category(X, y, boundary)


class NumericMixin(ABC):

    @staticmethod
    @abstractmethod
    def _fit_numeric(X, y, **kwargs):
        pass

    @staticmethod
    def _transform_numeric(X, y, boundary):
        y.rename('y', inplace=True)
        dt = pd.concat([X, y], axis=1)
        dt['bin'] = pd.cut(X, bins=boundary, right=False)
        dt = dt.groupby('bin', observed=True, dropna=False).agg(
            count=pd.NamedAgg(column='y', aggfunc='count'),
            pos=pd.NamedAgg(column='y', aggfunc='sum')
        ).reset_index()
        dt['variable'] = X.name
        dt['count_distr'] = dt['count'] / dt['count'].sum()
        dt['neg'] = dt['count'] - dt['pos']
        dt['posprob'] = dt['pos'] / dt['count']
        dt['woe'] = np.log((dt['pos'].map(lambda x: 0.00000001 if x == 0 else x) / dt['pos'].sum()) / (dt['neg'].map(lambda x: 0.00000001 if x == 0 else x) / dt['neg'].sum()))
        dt['bin_iv'] = (dt['pos'] / dt['pos'].sum() - dt['neg'] / dt['neg'].sum()) * dt['woe']
        dt['total_iv'] = dt['bin_iv'].sum()
        dt = dt[['variable', 'bin', 'count', 'count_distr', 'neg', 'pos', 'posprob', 'woe', 'bin_iv', 'total_iv']]
        return dt

    @classmethod
    def numeric(cls, X, y, **kwargs):
        boundary = cls._fit_numeric(X, y, **kwargs)
        return cls._transform_numeric(X, y, boundary)


class DecisionTree(CategoryMixin, NumericMixin):

    @staticmethod
    def _fit_category(X, y, **kwargs):
        def get_parent(node_index):
            parent_index = dt.loc[dt.node_index == node_index, 'parent_index'].values[0]
            add = []
            minus = []
            if parent_index:
                row = dt.loc[dt.node_index == parent_index]
                if row.left_child.values[0] == node_index:
                    add = np.array(row.threshold.values[0].split('||'), dtype=np.int32)
                else:
                    minus = np.array(row.threshold.values[0].split('||'), dtype=np.int32)
            return parent_index, add, minus

        def recursion(node_index):
            positive = []
            negative = []
            while node_index:
                node_index, add, minus = get_parent(node_index)
                positive.append(add)
                negative.append(minus)
            positive.append(np.nan_to_num(np.unique(encoded_X), nan=-1).astype(np.int32))
            return np.setdiff1d(
                reduce(np.intersect1d, filter(lambda x: len(x) > 0, positive)),
                reduce(np.union1d, negative)
            )

        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan, encoded_missing_value=np.nan)
        encoded_X = encoder.fit_transform(X.to_frame())
        clf = LGBMClassifier(
            objective='binary',
            num_leaves=kwargs.get('bins_num'),
            learning_rate=1,
            n_estimators=1,
            min_child_samples=int(X.shape[0] * 0.05),
            random_state=1,
            verbose=-1
        )
        clf.fit(encoded_X, y, categorical_feature=[0])
        dt = clf.booster_.trees_to_dataframe()
        leaf_node = dt.loc[dt.node_index.str.contains('L'), 'node_index']
        boundary = [tuple(encoder.inverse_transform(recursion(node).reshape(-1, 1)).reshape(-1, )) for node in leaf_node]
        return boundary

    @staticmethod
    def _fit_numeric(X, y, **kwargs):
        clf = DecisionTreeClassifier(
            criterion='entropy',
            splitter='best',
            max_leaf_nodes=kwargs.get('bins_num'),
            min_samples_leaf=0.05,
            random_state=1
        )
        clf.fit(X.to_frame(), y)
        n_nodes = clf.tree_.node_count  # 决策树的节点数
        children_left = clf.tree_.children_left  # node_count大小的数组，children_left[i]表示第i个节点的左子节点
        children_right = clf.tree_.children_right  # node_count大小的数组，children_right[i]表示第i个节点的右子节点
        threshold = clf.tree_.threshold  # node_count大小的数组，threshold[i]表示第i个节点划分数据集的阈值
        boundary = []
        for i in range(n_nodes):
            if children_left[i] != children_right[i]:  # 非叶节点
                boundary.append(threshold[i])
        boundary = sorted(boundary + [-np.inf, np.inf])
        return boundary


class ChiMerge(CategoryMixin, NumericMixin):

    @staticmethod
    @njit
    def _calculate_chi2(window):
        """计算卡方值"""
        row_totals = window.sum(axis=1)  # 每行的总和（长度2）
        col_totals = window.sum(axis=0)  # 每列的总和（长度n_cols）
        grand_total = row_totals.sum()  # 总总和
        # 计算期望频数：E = (行总和 × 列总和) / 总总和
        expected = np.outer(row_totals, col_totals) / grand_total
        # 处理可能的除以零（若期望值为0，该 term 贡献0）
        expected = np.where(expected == 0, 1e-10, expected)
        # 计算卡方值：Σ[(观察值-期望值)² / 期望值]
        result = np.sum((window - expected) ** 2 / expected)
        return result

    @classmethod
    def _fit_category(cls, X, y, **kwargs):
        bins_num = kwargs.get('bins_num')
        dt = pd.crosstab(X, y).reset_index(names='var')
        while True:
            dt['chi2'] = dt.iloc[:, 1:3].rolling(window=2, method='table').apply(
                cls._calculate_chi2,
                raw=True,
                engine='numba'
            ).iloc[:, 0]
            min_idx = dt['chi2'].idxmin(skipna=True)
            min_val = dt.iloc[min_idx, 3]
            threshold = chi2.ppf(q=0.95, df=dt.shape[0]-1)
            if min_val < threshold and dt.shape[0] > bins_num:
                old_rows = dt.iloc[min_idx-1:min_idx+1, :]
                new_row = pd.concat([
                    pd.DataFrame({'var': '%,%'.join(old_rows['var'])}, index=[0]),
                    pd.DataFrame([old_rows.iloc[:, 1:].sum(axis=0)])
                ], axis=1)
                dt = pd.concat([
                    dt.iloc[:min_idx-1],
                    new_row,
                    dt.iloc[min_idx+1:]
                ], axis=0).reset_index(drop=True)
            else:
                break
        boundary = dt['var'].map(lambda x: tuple(x.split('%,%'))).tolist()
        return boundary

    @classmethod
    def _fit_numeric(cls, X, y, **kwargs):
        bins_num = kwargs.get('bins_num')
        X_tf = pd.qcut(X, q=50, precision=3, duplicates='drop').astype('interval')
        dt = pd.crosstab(X_tf, y).reset_index(names='var')
        while True:
            dt['chi2'] = dt.iloc[:, 1:3].rolling(window=2, method='table').apply(
                cls._calculate_chi2,
                raw=True,
                engine='numba'
            ).iloc[:, 0]
            min_idx = dt['chi2'].idxmin(skipna=True)
            min_val = dt.iloc[min_idx, 3]
            threshold = chi2.ppf(q=0.95, df=dt.shape[0]-1)
            if min_val < threshold and dt.shape[0] > bins_num:
                old_rows = dt.iloc[min_idx-1:min_idx+1, :]
                new_row = pd.concat([
                    pd.DataFrame({
                        'var': pd.Interval(
                            left=old_rows['var'].map(lambda x: x.left).min(),
                            right=old_rows['var'].map(lambda x: x.right).max()
                        )
                    }, index=[0]),
                    pd.DataFrame([old_rows.iloc[:, 1:].sum(axis=0)])
                ], axis=1)
                dt = pd.concat([
                    dt.iloc[:min_idx-1],
                    new_row,
                    dt.iloc[min_idx+1:]
                ], axis=0).reset_index(drop=True)
            else:
                break
        boundary = sorted(dt['var'].map(lambda x: x.right).tolist()[:-1] + [-np.inf, np.inf])
        return boundary


def _assert_type(X):
    if X.dropna().unique().size <= 3:
        return 'ENUM'
    elif pd.api.types.is_object_dtype(X):
        return 'CATEGORY'
    elif pd.api.types.is_numeric_dtype(X):
        return 'NUMERIC'
    else:
        raise NotImplementedError('Not implemented data type.')


def woebin(dt, y, method='tree', bins_num=8):
    """
    Automatically finds the best woe cut points for each varibale in specific data set.

    Parameters
    ----------
    dt : Dataframe
        The data which needs to generate woe bin result.
    y: Series
        Target value.
    method : str
        Method used to automatic cut bin. Can choose tree, chimerge, best_ks.
    bins_num: int
        Number of bins wants to cut.

    Returns
    -------
    dict[str, DataFrame]
        Bin result.
    """
    match method:
        case 'tree':
            estimator = DecisionTree
        case 'chimerge':
            estimator = ChiMerge
        # case 'best_ks':
        #     pass
        case _:
            raise NotImplementedError('Not implemented method.')
    result = []
    for _, X in dt.items():
        match _assert_type(X):
            case 'ENUM':
                handler = estimator.enum
            case 'CATEGORY':
                handler = estimator.category
            case 'NUMERIC':
                handler = estimator.numeric
            case _:
                raise NotImplementedError('Not implemented data type.')
        result.append(handler(X, y, bins_num=bins_num))
    return dict(zip(dt.columns, result))


def woebin_ply(dt, bins):
    """
    Transform the original data to woe according to the giving bins.

    Parameters
    ----------
    dt : Dataframe
        The orignial data set.
    bins: dict[str, Dataframe]
        Bins result generated by function `woebin`.

    Returns
    -------
    Dataframe
        Woe result.
    """
    dt.reset_index(drop=True, inplace=True)
    bin_vars = pd.concat(bins)['variable'].unique()
    dt_vars = dt.columns
    final_vars = list(set(bin_vars).intersection(set(dt_vars)))

    results = [_woebin_ply2var(dt[var], bins.get(var)) for var in final_vars]
    return pd.DataFrame({result.name: result for result in results})


def _woebin_ply2var(X, bin):
    """
    Transform the original data to woe according to the giving bin matrix.

    Parameters
    ----------
    X : Series
        The orignial variable.
    bin: Dataframe
        Bin matrix.

    Returns
    -------
    Series
        Woe result.
    """
    default_value = bin.sort_values(by='posprob', ascending=False, ignore_index=True).loc[0, 'woe']
    match _assert_type(X):
        case 'ENUM' | 'CATEGORY':
            mapping = {category: row.woe for row in bin.itertuples() for category in row.bin}
            X = X.map(mapping).fillna(default_value)
        case 'NUMERIC':
            X = pd.Series(np.select(
                [(X >= v.left) & (X < v.right) if isinstance(v, pd.Interval) else np.isnan(X) for v in bin['bin']],
                bin['woe'],
                default=default_value
            ), name=X.name)
        case _:
            raise NotImplementedError('Not implemented data type.')
    return X


def woebin_plot(bins):
    plt.rcParams['font.sans-serif'] = 'Arial Unicode MS'
    for _, bin in bins.items():
        fig, ax1 = plt.subplots(figsize=(10, 5))
        xaxis = bin['bin'].astype(str)
        if xaxis.map(lambda x: len(x) > 35).any():
            xaxis = [f'Group {x}' for x in range(xaxis.size)]
        ax1.bar(xaxis, bin['neg'], color='#56BCC2', label='neg')
        bar = ax1.bar(xaxis, bin['pos'], bottom=bin['neg'], color='#E77D72', label='pos')
        ax1.bar_label(
            bar,
            labels=[f'{row.count_distr:.1%}, {row.count}' for row in bin.itertuples()]
        )
        ax1.set_ylabel('Count distribution')
        ax1.set_title(f'{bin.variable.iloc[0]} (iv: {bin.total_iv.iloc[0]:.4f})', loc='left')
        ax2 = ax1.twinx()
        ax2.plot(xaxis, bin['posprob'], color='blue', marker='o', markersize=4, linewidth=1)
        for x, y in zip(xaxis, bin['posprob']):
            ax2.text(
                x, y,
                f'{y:.2%}',
                ha='center',
                va='bottom',
                color='blue',
                fontsize=10,
            )
        ax2.set_ylabel('Positive probability', color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')
        fig.legend(loc='outside lower center', ncol=2, borderaxespad=0)
        fig.tight_layout()
        plt.show()
        plt.close(fig)
