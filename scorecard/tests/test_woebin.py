import pandas as pd
import numpy as np

from ..woebin import DecisionTree


class TestWoebin:

    def test_fit_enum(self):
        X = pd.Series([1, 2], name='x')
        y = pd.Series([0, 1], name='y')
        boundary = DecisionTree._fit_enum(X, y)
        assert boundary == [(1,), (2,)]

    def test_fit_category(self):
        X = pd.Series(['first', 'second', 'third', 'forth'], name='x')
        y = pd.Series([0, 1, 1, 1], name='y')
        boundary = DecisionTree._fit_category(X, y)
        assert boundary == [('first',), ('forth', 'second', 'third')]

    def test_fit_numeric(self):
        X = pd.Series([1, 2, 3, 4], name='x')
        y = pd.Series([0, 1, 1, 1], name='y')
        boundary = DecisionTree._fit_numeric(X, y)
        assert boundary == [-np.inf, np.float64(1.5), np.inf]
