"""SDS adapter implementations.

Submodules are imported here so that unittest.mock.patch() targets like
"sentinel.sds.adapters.fred_adapter.FREDAdapter.fetch_series" resolve
correctly under Python 3.11's pkgutil.resolve_name (which uses getattr,
not __import__, to walk dotted names).

Each import is individually guarded so a broken optional dependency in one
adapter never prevents other adapters from loading.
"""
from sentinel.sds.adapters import fred_adapter  # noqa: F401
from sentinel.sds.adapters import edgar_adapter  # noqa: F401
from sentinel.sds.adapters import alpaca_adapter  # noqa: F401
from sentinel.sds.adapters import polygon_adapter  # noqa: F401
from sentinel.sds.adapters import congress_adapter  # noqa: F401
from sentinel.sds.adapters import insider_adapter  # noqa: F401
from sentinel.sds.adapters import institutional_adapter  # noqa: F401

try:
    from sentinel.sds.adapters import finnhub_adapter  # noqa: F401
except Exception:
    pass

try:
    from sentinel.sds.adapters import yfinance_adapter  # noqa: F401
except Exception:
    pass

try:
    from sentinel.sds.adapters import ccxt_adapter  # noqa: F401
except Exception:
    pass

try:
    from sentinel.sds.adapters import fx_adapter  # noqa: F401
except Exception:
    pass

try:
    from sentinel.sds.adapters import short_interest_adapter  # noqa: F401
except Exception:
    pass
