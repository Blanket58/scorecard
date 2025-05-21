__doc__ = r"""

  ____                           ____                 _ 
 / ___|   ___  ___   _ __  ___  / ___| __ _  _ __  __| |
 \___ \  / __|/ _ \ | '__|/ _ \| |    / _` || '__|/ _` |
  ___) || (__| (_) || |  |  __/| |___| (_| || |  | (_| |
 |____/  \___|\___/ |_|   \___| \____|\__,_||_|   \__,_|


Toolbox for building a risk scorecard.
=========================================================

Main functions:
---------------

  - scorecard.woebin
    Auto woe using different algorithms, such as decision tree, chisquare, bestks, etc.
    Observing woe plot after binning, judging if the result is comfortable from businesss perspective.

  - scorecard.iv
    Extract information value table from binning result.

  - scorecard.perf
    Evaluate any kind of binary classification model, plot K-S, ROC, PR, PR vs. threshold, PSI, 
    compute gains table, see if the model lift is significant between the lower score and higher score.

  - scorecard.card
    Transform probability to score, model to scorecard, card to SQL.

"""

from .card import card2sql, prob2score, scorecard
from .iv import iv_table
from .perf import gains_table, perf_eva, perf_psi
from .woebin import woebin, woebin_plot, woebin_ply
