"""Label-independent candidate catalog composition for offline inference."""

import pandas as pd


def expanded_pool(pool, external):
    """Append new structure keys without replacing existing mass/representations."""
    extra = external[~external.inchikey14.isin(pool.inchikey14)]
    return pd.concat([pool, extra[pool.columns]], ignore_index=True)
