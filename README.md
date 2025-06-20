# scorecard

[![Python package](https://github.com/Blanket58/scorecard/actions/workflows/python-package.yml/badge.svg)](https://github.com/Blanket58/scorecard/actions/workflows/python-package.yml)

Rewrite brand new robust scorecard toolbox, if you are looking for something similar as `scorecard` in R and `scorecardpy` in python.

## User Guide

Download from releases and install it.

```bash
pip install scorecard-0.0.1-py3-none-any.whl
```

Using germancredit dataset as example.

```python
import json
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scorecard import gains_table, iv_table, perf_eva, perf_psi, prob2score, woebin, woebin_plot, woebin_ply, scorecard, card2sql
```

```python
import ssl
from ucimlrepo import fetch_ucirepo

ssl._create_default_https_context = ssl._create_unverified_context
statlog_german_credit_data = fetch_ucirepo(id=144)
X = statlog_german_credit_data.data.features
y = statlog_german_credit_data.data.targets.squeeze().map({1:0, 2:1})
```

```python
print(json.dumps(statlog_german_credit_data.metadata, ensure_ascii=False, indent=4))
```

```python
statlog_german_credit_data.variables
```

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1, stratify=y)
print('train', y_train.value_counts())
print('test', y_test.value_counts())
```

```python
bins = woebin(X_train, y_train)
```

```python
iv = iv_table(bins)
iv
```

```python
variables = iv.loc[iv['total_iv'] >= 0.02, 'variable'].tolist()
bins = dict(filter(lambda x: x[0] in variables, bins.items()))
```

```python
woebin_plot(bins)
```

```python
X_train_woe = woebin_ply(X_train[variables], bins)
X_test_woe = woebin_ply(X_test[variables], bins)
```

```python
corr = X_train_woe.corr()
fig = px.imshow(corr, aspect='auto', width=800, height=600)
fig.show()
```

```python
clf = LogisticRegressionCV(
    Cs=10,
    fit_intercept=True,
    cv=5,
    penalty='l1',
    solver='liblinear',
    random_state=1
)
clf.fit(X_train_woe, y_train)
```

```python
vif = pd.DataFrame()
vif['variable'] = variables
vif['coefficients'] = clf.coef_.reshape(-1,)
vif['vif'] = [variance_inflation_factor(X_train_woe, i) for i in range(len(variables))]
vif.sort_values('vif', ascending=False)
```

```python
y_train_pred = clf.predict_proba(X_train_woe)[:, 1]
y_test_pred = clf.predict_proba(X_test_woe)[:, 1]
```

```python
perf_eva(y_train, y_train_pred, 'Train')
perf_eva(y_test, y_test_pred, 'Test')
```

```python
total_bad_rate = y_train.sum() / y_train.size
odds0 = total_bad_rate / (1 - total_bad_rate)
y_train_score = prob2score(
    y_train_pred,
    points0=600,
    odds0=odds0,
    pdo=50
)
y_test_score = prob2score(
    y_test_pred,
    points0=600,
    odds0=odds0,
    pdo=50
)
```

```python
gains_table(
    label={'train': y_train, 'test': y_test},
    score={'train': y_train_score, 'test': y_test_score},
    bin_num=10
)
```

```python
perf_psi(
    label={'train': y_train, 'test': y_test},
    score={'train': y_train_score, 'test': y_test_score}
)
```

