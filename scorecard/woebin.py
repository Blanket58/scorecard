from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier
from matplotlib import pyplot as plt
from numba import njit, prange, set_num_threads
from scipy.stats import chi2
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class BaseWoeEncoder(TransformerMixin, BaseEstimator, ABC):

    def __init__(self, bins_num=8, random_state=1, n_jobs=2):
        """
        Parameters
        ----------
        bins_num: int, default=8
            The number of bins you want the varibale been cut into.
        random_state: int, default=1
            Controls the random seed.
        n_jobs: int, default=1
            Number of CPU cores to use for parallel processing.
            Modify this value only if the number of input variables is extremely large.
        """
        assert bins_num >= 2, "Argument bins_num must >= 2."
        self.bins_num = bins_num
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.boundaries_ = {}
        self.bins_result_ = {}
        self.feature_types_ = {}
        self.feature_names_in_ = None
        self.n_features_in_ = None

        self._type_strategies = {
            "ENUM": (self._fit_enum, self._calc_enum),
            "CATEGORY": (self._fit_category, self._calc_enum),
            "NUMERIC": (self._fit_numeric, self._calc_numeric),
        }
        self._EPS = np.finfo(np.float64).eps

    @staticmethod
    def _validate_fit_input(X, y):
        """Input validation for estimators' fit method.

        Checks X and y for consistent length, enforces X to be 2D and y 1D.
        Checks if y is binary and is numeric.

        Parameters
        ----------
        X: {ndarray, list, sparse matrix, Dataframe}
            Input data.
        y: {ndarray, list, sparse matrix}
            Labels.

        Returns
        -------
        X_converted: Dataframe
            The converted and validated X.
        y_converted: object
            The converted and validated y.
        """
        y_type = type_of_target(y)
        assert y_type == "binary", f"Target must be binary, now is {y_type}."
        feature_names = None
        dtype_mapping = None
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
            dtype_mapping = X.dtypes.to_dict()
        X, y = check_X_y(
            X, y, accept_sparse=False, ensure_all_finite="allow-nan", y_numeric=True
        )
        if not feature_names:
            feature_names = [f"feat_{i}" for i in range(X.shape[1])]
        X = pd.DataFrame(X, columns=feature_names).fillna(np.nan)
        if dtype_mapping:
            X = X.astype(dtype_mapping)
        return X, y

    @staticmethod
    def _validate_transform_input(X):
        """Input validation for estimators' transform method.

        Enforces X to be 2D.

        Parameters
        ----------
        X: {ndarray, list, sparse matrix, Dataframe}
            Input data.

        Returns
        -------
        X_converted: Dataframe
            The converted and validated X.
        """
        feature_names = None
        dtype_mapping = None
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
            dtype_mapping = X.dtypes.to_dict()
        X = check_array(X, accept_sparse=False, ensure_all_finite="allow-nan")
        if not feature_names:
            feature_names = [f"feat_{i}" for i in range(X.shape[1])]
        X = pd.DataFrame(X, columns=feature_names).fillna(np.nan)
        if dtype_mapping:
            X = X.astype(dtype_mapping)
        return X

    @staticmethod
    def _infer_feature_type(x_col):
        """Infer which data type the variable is.

        Parameters
        ----------
        x_col: pandas.Series

        Returns
        -------
        data_type: str
        """
        if x_col.dropna().unique().size <= 3:
            return "ENUM"
        elif pd.api.types.is_string_dtype(x_col):
            return "CATEGORY"
        elif pd.api.types.is_any_real_numeric_dtype(x_col):
            return "NUMERIC"
        else:
            return x_col.dtype.name

    def _fit(self, x_col, y):
        var_type = self._infer_feature_type(x_col)
        try:
            fit_strategy, calc_strategy = self._type_strategies[var_type]
        except KeyError:
            raise NotImplementedError(
                f"Unsupported varibale type: {x_col.name}->{var_type}"
            )
        boundary = fit_strategy(x_col, y)
        bins_df = calc_strategy(x_col, y, boundary)
        return x_col.name, var_type, boundary, bins_df

    def fit(self, X, y):
        """Build a woe encoder from the training set (X, y).

        Parameters
        ----------
        X: {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples.
        y: array-like of shape (n_samples,)
            The target values (class labels) as numeric and binary.

        Returns
        -------
        self : WoeEncoder
            Fitted estimator.
        """
        X, y = self._validate_fit_input(X, y)
        self.feature_names_in_ = X.columns.tolist()
        self.n_features_in_ = X.shape[1]
        results = Parallel(n_jobs=self.n_jobs, verbose=0, prefer="processes")(
            delayed(self._fit)(X[col], y) for col in self.feature_names_in_
        )
        for col, var_type, boundary, bins_df in results:
            self.feature_types_[col] = var_type
            self.boundaries_[col] = boundary
            self.bins_result_[col] = bins_df
        return self

    def transform(self, X):
        """Transform the original data to woe with the fitted encoder.

        Parameters
        ----------
        X: {ndarray, list, sparse matrix, Dataframe}
            Input data.

        Returns
        -------
        X_transformed: Dataframe
            The transformed X.
        """
        check_is_fitted(self, ["boundaries_", "bins_result_"])
        X = self._validate_transform_input(X)
        cols = set(X.columns).intersection(self.boundaries_.keys())
        results = []
        for col in cols:
            x_col = X[col]
            var_type = self.feature_types_[col]
            bins_df = self.bins_result_[col]
            default_value = bins_df.sort_values(
                by="posprob", ascending=False, ignore_index=True
            ).loc[0, "woe"]
            if var_type in ["ENUM", "CATEGORY"]:
                mapping = {
                    item: row.woe for row in bins_df.itertuples() for item in row.bin
                }
                x_col = x_col.map(mapping).fillna(default_value)
            elif var_type == "NUMERIC":
                x_col = pd.Series(
                    np.select(
                        [
                            (
                                (x_col > v.left) & (x_col <= v.right)
                                if isinstance(v, pd.Interval)
                                else np.isnan(x_col)
                            )
                            for v in bins_df["bin"]
                        ],
                        bins_df["woe"],
                        default=default_value,
                    ),
                    name=x_col.name,
                )
            results.append(x_col)
        X_woe = pd.concat(results, axis=1)
        return X_woe

    @staticmethod
    def _fit_enum(x_col, y):  # noqa
        """Determine discretization bins for enumeration data.

        For data with 3 or fewer unique values (regardless of its dtype), it will be
        classified as ENUM type, each unique value will be treated as an independent
        discretization bin.

        Parameters
        ----------
        x_col : pandas.Series
            The variable to be discretized.
        y : array-like of shape (n_samples,)
            Target values (binary/numeric class labels) for supervised discretization.

        Returns
        -------
        list of tuples
            A list of boundary tuples where each tuple contains a single unique value,
            e.g. [(val1,), (val2,), ...], representing individual bins for each value.
        """
        boundary = [(x,) for x in x_col.unique()]
        return boundary

    @abstractmethod
    def _fit_category(self, x_col, y):
        """Determine discretization bins for categorical data.

        For string-dtype data with more than 3 unique values, it will be classified as
        CATEGORY type, the algorithm will attempt to find the optimal grouping strategy
        for all values.

        Parameters
        ----------
        x_col : pandas.Series
            The variable to be discretized.
        y : array-like of shape (n_samples,)
            Target values (binary/numeric class labels) for supervised discretization.

        Returns
        -------
        list of tuples
            A list of boundary tuples where each tuple contains one or more unique value,
            e.g. [(val1,val2,), (val3,), ...], representing the discrete bins for grouped values.
        """
        pass

    @abstractmethod
    def _fit_numeric(self, x_col, y):
        """Determine discretization bins for numerical data.

        For numeric-dtype data, it will be classified as NUMERIC type, the algorithm will
        attempt to find the optimal cut points.

        Parameters
        ----------
        x_col : pandas.Series
            The variable to be discretized.
        y : array-like of shape (n_samples,)
            Target values (binary/numeric class labels) for supervised discretization.

        Returns
        -------
        list of float
            A list of boundary values (cut points) for numeric discretization,
            e.g. [-inf, 10.0, ... , 100.0, inf], representing the split points between discrete bins.
        """
        pass

    def _calc_enum(self, x_col, y, boundary):
        return self._calc_category(x_col, y, boundary)

    def _calc_category(self, x_col, y, boundary):
        """Calculate bin statistics for categorical feature.

        Maps each categorical value to its assigned bin group (per pre-defined boundary),
        then constructs a DataFrame with bin labels and target values, and calculates
        statistical metrics (e.g., positive probability) for each bin via `_calc_bins_df`.

        Parameters
        ----------
        x_col : pandas.Series
            Single categorical feature column to be transformed (e.g., string-type with >3 unique values).
        y : array-like of shape (n_samples,)
            Target values (binary/numeric class labels) for bin statistic calculation.
        boundary : list of tuples
            Pre-determined bin boundaries for categorical data (each tuple = one bin group,
            containing multiple categorical values, e.g. [(val1, val2), (val3,), ...]).

        Returns
        -------
        pandas.DataFrame
            With bin-level statistics (e.g., posprob) for the categorical feature,
            sorted by positive probability in ascending order.
        """
        mapping = {item: group for group in boundary for item in group}
        df = pd.DataFrame({"variable": x_col.name, "bin": x_col.map(mapping), "y": y})
        bins_df = self._calc_bins_df(df)
        bins_df.sort_values(
            by="posprob", ascending=True, ignore_index=True, inplace=True
        )
        return bins_df

    def _calc_numeric(self, x_col, y, boundary):
        """Calculate bin statistics for numerical feature.

        Uses pandas.cut to split numeric values into continuous bins (per pre-defined cut points),
        constructs a DataFrame with bin labels and target values, and calculates statistical
        metrics for each bin via `_calc_bins_df`.

        Parameters
        ----------
        x_col : pandas.Series
            Single numeric feature column to be transformed (e.g., int/float type).
        y : array-like of shape (n_samples,)
            Target values (binary/numeric class labels) for bin statistic calculation.
        boundary : list of float
            Pre-determined cut points for numeric binning (e.g., [-inf, 10.0, ... , 100.0, inf]),
            defining the range of each continuous bin.

        Returns
        -------
        pandas.DataFrame
            With bin-level statistics (e.g., posprob) for the numeric feature.
        """
        df = pd.DataFrame(
            {
                "variable": x_col.name,
                "bin": pd.cut(
                    x_col, bins=boundary, right=True, precision=3, duplicates="drop"
                ),
                "y": y,
            }
        )
        bins_df = self._calc_bins_df(df)
        return bins_df

    def _calc_bins_df(self, df):
        bins_df = (
            df.groupby(["variable", "bin"], observed=True, dropna=False)
            .agg(count=("y", "count"), pos=("y", "sum"))
            .reset_index()
        )
        bins_df["count_distr"] = bins_df["count"] / bins_df["count"].sum()
        bins_df["neg"] = bins_df["count"] - bins_df["pos"]
        bins_df["posprob"] = bins_df["pos"] / bins_df["count"]
        bins_df["woe"] = np.log(
            (bins_df["pos"].replace(0, self._EPS) / bins_df["pos"].sum())
            / (bins_df["neg"].replace(0, self._EPS) / bins_df["neg"].sum())
        )
        bins_df["bin_iv"] = (
            bins_df["pos"] / bins_df["pos"].sum()
            - bins_df["neg"] / bins_df["neg"].sum()
        ) * bins_df["woe"]
        bins_df["total_iv"] = bins_df["bin_iv"].sum()
        bins_df = bins_df[["variable", "bin", "count", "count_distr", "neg", "pos", "posprob", "woe", "bin_iv", "total_iv"]]  # fmt: skip # noqa
        return bins_df

    @staticmethod
    def _plot(name, bins_df):
        plt.rcParams["font.sans-serif"] = "Arial Unicode MS"
        fig, ax1 = plt.subplots(figsize=(8, 4))
        xaxis = bins_df["bin"].astype(str)
        if xaxis.map(lambda x: len(x) > 35).any():
            xaxis = [f"Group {x}" for x in range(xaxis.size)]
        ax1.bar(xaxis, bins_df["neg"], color="#56BCC2", label="neg")
        bar = ax1.bar(
            xaxis,
            bins_df["pos"],
            bottom=bins_df["neg"],
            color="#E77D72",
            label="pos",
        )
        ax1.bar_label(
            bar,
            labels=[
                f"{row.count_distr:.1%}, {row.count}"
                for row in bins_df.itertuples()
            ],
        )
        ax1.set_ylabel("Count distribution")
        ax1.set_title(f"{name} (iv: {bins_df.total_iv.iloc[0]:.4f})", loc="left")
        ax2 = ax1.twinx()
        ax2.plot(
            xaxis,
            bins_df["posprob"],
            color="blue",
            marker="o",
            markersize=4,
            linewidth=1,
        )
        for x, y in zip(xaxis, bins_df["posprob"]):
            ax2.text(
                x,
                y,
                f"{y:.2%}",
                ha="center",
                va="bottom",
                color="blue",
                fontsize=10,
            )
        ax2.set_ylabel("Positive probability", color="blue")
        ax2.tick_params(axis="y", labelcolor="blue")
        fig.legend(loc="outside lower center", ncol=2, borderaxespad=0)
        fig.tight_layout()
        plt.show()
        plt.close(fig)

    def plot(self, feature_name: str = None):
        """Visualize the discretization bins result for one or all features.

        Generate woe plots for the pre-calculated discretization bins of a
        specified feature, or all features if no name is given.

        Parameters
        ----------
        feature_name : str, optional
            The name of the feature to plot. If not provided, plots will be
            generated for all available features in `bins_result_`.

        Raises
        ------
        KeyError
            If the specified `feature_name` was never fitted.
        """
        if feature_name:
            try:
                bins_df = self.bins_result_[feature_name]
            except KeyError:
                raise f'Variable {feature_name} never been fitted.'
            self._plot(feature_name, bins_df)
        else:
            for name, bins_df in self.bins_result_.items():
                self._plot(name, bins_df)


class DecisionTreeWoeEncoder(BaseWoeEncoder):

    def _fit_category(self, x_col, y):
        clf = LGBMClassifier(
            objective="binary",
            num_leaves=self.bins_num,
            learning_rate=1,
            n_estimators=1,
            min_child_samples=int(x_col.shape[0] * 0.05),
            random_state=self.random_state,
            verbose=-1,
        )
        x_col = x_col.astype("category")
        clf.fit(x_col.to_frame(), y, categorical_feature=0)
        x_pred = clf.predict(x_col.to_frame(), pred_leaf=True).reshape(-1)
        boundary = (
            pd.DataFrame({"value": x_col, "group": x_pred})
            .drop_duplicates()
            .groupby("group")["value"]
            .agg(tuple)
            .tolist()
        )
        return boundary

    def _fit_numeric(self, x_col, y):
        clf = DecisionTreeClassifier(
            criterion="entropy",
            splitter="best",
            max_leaf_nodes=self.bins_num,
            min_samples_leaf=0.05,
            random_state=self.random_state,
        )
        clf.fit(x_col.to_frame(), y)
        n_nodes = clf.tree_.node_count
        children_left = clf.tree_.children_left
        children_right = clf.tree_.children_right
        threshold = clf.tree_.threshold
        boundary = []
        for i in range(n_nodes):
            if children_left[i] != children_right[i]:  # not left node
                boundary.append(threshold[i])
        boundary = sorted(boundary + [-np.inf, np.inf])
        return boundary


class ChiMergeWoeEncoder(BaseWoeEncoder):

    def __init__(self, *, n_threads=1, **kwargs):
        """
        Parameters
        ----------
        n_threads: int, default=1
            Number of threads to use for numba-accelerated operations.
            Modify this value only if input data contains variables with an extremely large number of unique values.
        """
        super().__init__(**kwargs)
        self._threshold = chi2.ppf(q=0.95, df=1)
        self.n_threads = n_threads
        set_num_threads(n_threads)

    @staticmethod
    @njit(fastmath=True, parallel=True)
    def _calc_chi2(window: np.ndarray) -> np.ndarray:
        n_pairs = window.shape[0]
        chi2_vals = np.zeros(n_pairs, dtype=np.float64)
        for i in prange(n_pairs):
            pair = window[i]
            row_totals = pair.sum(axis=1)
            col_totals = pair.sum(axis=0)
            grand_total = row_totals.sum()
            if grand_total == 0:
                continue
            expected = np.outer(row_totals, col_totals) / grand_total
            expected = np.where(expected == 0, np.finfo(np.float64).eps, expected)
            chi2_vals[i] = np.sum((pair - expected) ** 2 / expected)
        return chi2_vals

    @staticmethod
    @njit(fastmath=True)
    def _merge_bins(freq_matrix: np.ndarray, min_idx: int) -> np.ndarray:
        merged_row = freq_matrix[min_idx] + freq_matrix[min_idx + 1]
        new_matrix = np.zeros((freq_matrix.shape[0] - 1, 2), dtype=np.float64)
        new_matrix[:min_idx] = freq_matrix[:min_idx]
        new_matrix[min_idx] = merged_row
        new_matrix[min_idx + 1:] = freq_matrix[min_idx + 2:]  # fmt: skip
        return new_matrix

    def _fit_category(self, x_col, y):
        freq_matrix = pd.crosstab(x_col, y, dropna=False)
        boundary = [(x,) for x in freq_matrix.index]
        freq_matrix = freq_matrix.values.astype(np.float64)
        while freq_matrix.shape[0] > self.bins_num:
            window = np.stack([freq_matrix[:-1], freq_matrix[1:]], axis=1)
            chi2_vals = self._calc_chi2(window)
            min_idx = np.argmin(chi2_vals)
            min_chi2 = chi2_vals[min_idx]
            if min_chi2 >= self._threshold:
                break
            freq_matrix = self._merge_bins(freq_matrix, min_idx)
            merged_interval = boundary[min_idx] + boundary[min_idx + 1]
            boundary.pop(min_idx + 1)
            boundary[min_idx] = merged_interval
        return boundary

    def _fit_numeric(self, x_col, y):
        try:
            x_col = pd.qcut(
                x_col, q=min(50, x_col.unique().size), precision=3, duplicates="drop"
            )
        except ValueError:
            return [-np.inf, np.inf]
        raw_interval_groups = self._fit_category(x_col, y)
        boundary = []
        for interval_group in raw_interval_groups:
            valid_intervals = []
            for elem in interval_group:
                if isinstance(elem, pd.Interval):
                    valid_intervals.append(elem)
            if not valid_intervals:
                continue
            merged_left = min([x.left for x in valid_intervals])
            merged_right = max([x.right for x in valid_intervals])
            merged_interval = pd.Interval(
                left=merged_left, right=merged_right, closed="right"
            )
            boundary.append(merged_interval)
        boundary = sorted(
            pd.IntervalIndex(boundary).right[:-1].tolist() + [-np.inf, np.inf]
        )
        return boundary
