# scorecard

[![Python package](https://github.com/Blanket58/scorecard/actions/workflows/python-package.yml/badge.svg)](https://github.com/Blanket58/scorecard/actions/workflows/python-package.yml)

A robust, sklearn-style scorecard toolbox for credit risk modeling. A Python alternative to R's `scorecard` package, providing end-to-end functionality from binning and WOE encoding to scorecard generation, SQL deployment, and model performance evaluation.


## Key Features

- **Smart Binning**: Supports both ChiMerge and Decision Tree-based binning methods.
- **WOE Encoding**: Automatic Weight of Evidence (WOE) transformation for categorical and continuous features.
- **IV Calculation**: Computes Information Value (IV) for data-driven feature selection.
- **Scorecard Generation**: Creates interpretable scorecards with point values for each feature bin.
- **SQL Conversion**: Converts scorecards to executable SQL CASE WHEN statements for production deployment.
- **Performance Evaluation**: Includes KS, AUC, PSI (Population Stability Index), gains table, and lift analysis.
- **Sklearn Compatible**: Follows sklearn API conventions for seamless integration with existing machine learning workflows.


## Installation

Download the latest wheel file from [Releases](https://github.com/Blanket58/scorecard/releases) and install via pip:

```bash
pip install scorecard-0.0.8-py3-none-any.whl
```

(Note: Replace the wheel filename with the latest version from Releases.)


## Quick Start

Here's a minimal end-to-end example:

```python
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV

from scorecard import (
    ChiMergeWoeEncoder,
    scorecard,
    card2sql,
    perf_eva,
    perf_psi
)

# 1. Load sample data (German Credit Data)
statlog_german_credit_data = fetch_ucirepo(id=144)
X = statlog_german_credit_data.data.features
y = statlog_german_credit_data.data.targets.squeeze().map({1: 0, 2: 1})

# 2. WOE encoding with ChiMerge binning
woe_encoder = ChiMergeWoeEncoder()
X_woe = woe_encoder.fit_transform(X, y)

# 3. Feature selection (IV >= 0.02)
iv = woe_encoder.iv_table
variables = iv.loc[iv['total_iv'] >= 0.02, 'variable'].tolist()
bins = dict(filter(lambda x: x[0] in variables, woe_encoder.bins_result_.items()))

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_woe[variables], y, test_size=0.25, random_state=1, stratify=y
)

# 5. Train logistic regression model
clf = LogisticRegressionCV(
    Cs=10, fit_intercept=True, cv=5, l1_ratios=(1,),
    solver='liblinear', random_state=1
)
clf.fit(X_train, y_train)

# 6. Generate scorecard
card = scorecard(
    bins, clf.intercept_, clf.coef_.reshape(-1,), variables,
    points0=600, odds0=y_train.sum()/(y_train.size - y_train.sum()), pdo=50
)

# 7. Convert scorecard to SQL
sql = card2sql(card, to_clipboard=False)
print(sql)
```


## Full Example

See the complete end-to-end workflow (including performance evaluation, PSI analysis, and visualization) on nbviewer:  
[📓 Example Notebook](https://nbviewer.org/github/Blanket58/scorecard/blob/main/examples/example.ipynb)
