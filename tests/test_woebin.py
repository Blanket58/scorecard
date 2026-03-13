from hashlib import md5

import numpy as np
import pandas as pd

from scorecard.woebin import DecisionTreeWoeEncoder, ChiMergeWoeEncoder


class TestWoebin:

    def test_fit_enum(self):
        X = pd.Series([1, 2], name='x')
        y = pd.Series([0, 1], name='y')
        boundary = DecisionTreeWoeEncoder()._fit_enum(X, y)
        assert boundary == [(1,), (2,)]
        boundary = ChiMergeWoeEncoder()._fit_enum(X, y)
        assert boundary == [(1,), (2,)]

    def test_fit_category(self):
        X = pd.Series(['first', 'second', 'third', 'forth'], name='x')
        y = pd.Series([0, 1, 1, 1], name='y')
        boundary = DecisionTreeWoeEncoder()._fit_category(X, y)
        assert boundary == [('first',), ('second', 'third', 'forth')]
        boundary = ChiMergeWoeEncoder(bins_num=2)._fit_category(X, y)
        assert boundary == [('first',), ('forth', 'second', 'third')]

    def test_fit_numeric(self):
        X = pd.Series([1, 2, 3, 4], name='x')
        y = pd.Series([0, 1, 1, 1], name='y')
        boundary = DecisionTreeWoeEncoder(bins_num=2)._fit_numeric(X, y)
        assert boundary == [-np.inf, 1.5, np.inf]
        boundary = ChiMergeWoeEncoder(bins_num=2)._fit_numeric(X, y)
        assert boundary == [-np.inf, 1.75, np.inf]

    def test_fit_enum_with_missing(self):
        X = pd.Series([1, np.nan], name='x')
        y = pd.Series([0, 1], name='y')
        boundary = DecisionTreeWoeEncoder()._fit_enum(X, y)
        assert np.allclose(boundary, [(1,), (np.nan,)], equal_nan=True)

    def test_fit_category_with_missing(self):
        X = pd.Series(['first', 'second', 'third', np.nan], name='x')
        y = pd.Series([0, 1, 1, 1], name='y')
        boundary = DecisionTreeWoeEncoder()._fit_category(X, y)
        assert md5(str(boundary).encode()).hexdigest() == md5("[('first',), ('second', 'third', nan)]".encode()).hexdigest()

    def test_fit_numeric_with_missing(self):
        X = pd.Series([1, 2, 3, np.nan], name='x')
        y = pd.Series([0, 1, 1, 0], name='y')
        boundary = DecisionTreeWoeEncoder()._fit_numeric(X, y)
        assert boundary == [-np.inf, 1.5, np.inf]
