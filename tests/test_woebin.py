from hashlib import md5

import numpy as np
import pandas as pd

from scorecard.woebin import DecisionTree, ChiMerge


class TestWoebin:

    def test_fit_enum(self):
        X = pd.Series([1, 2], name='x')
        y = pd.Series([0, 1], name='y')
        boundary = DecisionTree._fit_enum(X, y)
        assert boundary == [(1,), (2,)]
        boundary = ChiMerge._fit_enum(X, y)
        assert boundary == [(1,), (2,)]

    def test_fit_category(self):
        X = pd.Series(['first', 'second', 'third', 'forth'], name='x')
        y = pd.Series([0, 1, 1, 1], name='y')
        boundary = DecisionTree._fit_category(X, y)
        assert boundary == [('first',), ('forth', 'second', 'third')]
        boundary = ChiMerge._fit_category(X, y, bins_num=2)
        assert boundary == [('first',), ('forth', 'second', 'third')]

    def test_fit_numeric(self):
        X = pd.Series([1, 2, 3, 4], name='x')
        y = pd.Series([0, 1, 1, 1], name='y')
        boundary = DecisionTree._fit_numeric(X, y)
        assert boundary == [-np.inf, 1.5, np.inf]
        boundary = ChiMerge._fit_numeric(X, y, bins_num=2)
        assert boundary == [-np.inf, 1.06, np.inf]

    def test_fit_enum_with_missing(self):
        X = pd.Series([1, np.nan], name='x')
        y = pd.Series([0, 1], name='y')
        boundary = DecisionTree._fit_enum(X, y)
        assert np.allclose(boundary, [(1,), (np.nan,)], equal_nan=True)

    def test_fit_category_with_missing(self):
        X = pd.Series(['first', 'second', 'third', np.nan], name='x')
        y = pd.Series([0, 1, 1, 1], name='y')
        boundary = DecisionTree._fit_category(X, y)
        assert md5(str(boundary).encode()).hexdigest() == md5("[('first',), (nan, 'second', 'third')]".encode()).hexdigest()

    def test_fit_numeric_with_missing(self):
        X = pd.Series([1, 2, 3, np.nan], name='x')
        y = pd.Series([0, 1, 1, 0], name='y')
        boundary = DecisionTree._fit_numeric(X, y)
        assert boundary == [-np.inf, 1.5, np.inf]
