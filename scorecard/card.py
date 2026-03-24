import re

import numpy as np
import pandas as pd
import pyperclip
from jinja2 import Environment


def _ab(points0, odds0, pdo):
    b = pdo / np.log(2)
    a = points0 + b * np.log(odds0)
    return a, b


def prob2score(y_proba, points0=600, odds0=1 / 19, pdo=50):
    a, b = _ab(points0, odds0, pdo)
    return np.round(a - b * np.log(y_proba / (1 - y_proba)))


def scorecard(bins, intercept, coef, feature_names, points0=600, odds0=1 / 19, pdo=50):
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
    dt["coef"] = dt["variable"].map({x: y for x, y in zip(feature_names, coef)})
    dt["score"] = -np.round(b * dt["coef"] * dt["woe"])
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "variable": "Base Line",
                    "bin": None,
                    "score": np.round(a - b * intercept),
                }
            ),
            dt[["variable", "bin", "score"]],
        ],
        axis=0,
        ignore_index=True,
    )


def card2sql(card, to_clipboard=True):
    """
    将scorecard()输出的评分卡结果转为case when SQL

    Parameters
    ----------
    card : DataFrame
        scorecard()输出的评分卡结果（不可做任何修改）
    to_clipboard: bool
        是否同时复制到剪切板

    Returns
    -------
    str
        计算评分卡总分的SQL
    """

    def parse(dt):
        def is_tuple(value):
            return isinstance(value, tuple)

        def is_interval(value):
            return isinstance(value, pd.Interval)

        def has_nan(value):
            if not isinstance(value, tuple):
                return False
            return np.nan in value

        def filter_nan(value):
            return tuple(filter(lambda x: x == x, value))

        def is_infinite(value):
            return not np.isfinite(value)

        def format_tuple(value):
            res = []
            for x in value:
                if isinstance(x, str):
                    res.append(f"'{x}'")
                elif isinstance(x, np.number):
                    res.append(str(x.item()))
                else:
                    res.append(str(x))
            return res

        env = Environment(autoescape=False, trim_blocks=True, lstrip_blocks=True)
        env.filters["is_tuple"] = is_tuple
        env.filters["is_interval"] = is_interval
        env.filters["has_nan"] = has_nan
        env.filters["filter_nan"] = filter_nan
        env.filters["is_infinite"] = is_infinite
        env.filters["format_tuple"] = format_tuple
        templ = """
        case
        {% for row in dt.itertuples() %}
            {% if row.bin | is_tuple %}
                {% if row.bin | has_nan %}
                    {% if len(row.bin) == 1 %}
            when {{ row.variable }} is null then {{ row.score }}
                    {% else %}
            when {{ row.variable }} is null or {{ row.variable }} in ({{ row.bin | filter_nan | format_tuple | join(',') }}) then {{ row.score }}
                    {% endif %}
                {% else %}
            when {{ row.variable }} in ({{ row.bin | format_tuple | join(',') }}) then {{ row.score }}
                {% endif %}
            {% elif row.bin | is_interval %}
                {% if row.bin.right | is_infinite %}
            else {{ row.score }}
                {% else %}
            when {{ row.variable }} <= {{ row.bin.right }} then {{ row.score }}
                {% endif %}
            {% else %}
            else {{ row.score }}
            {% endif %}
        {% endfor %}
        end
        """  # noqa
        return env.from_string(templ).render({"dt": dt})

    variables = card.iloc[1:, 0].unique()
    result = [str(card.iloc[0, 2])]
    for variable in variables:
        result.append(parse(card.loc[card["variable"] == variable, :]))
    result = " + ".join(result)
    if to_clipboard:
        pyperclip.copy(result)
    return result

