import pandas as pd


def iv_table(bins):
    dt = pd.concat(bins, axis=0)
    return dt[['variable', 'total_iv']].drop_duplicates().sort_values(by='total_iv', ascending=False, ignore_index=True)
