"""Orders routes — requires SENTINEL_LIVE_TRADING=true and Alpaca credentials."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from sentinel.core.security import assert_live_trading_enabled

router = APIRouter()


class OrderRequest(BaseModel):
    ticker: str
    side: str  # 'buy' or 'sell'
    quantity: float
    order_type: str = "market"
    limit_price: Optional[float] = None
    strategy_id: Optional[str] = None


@router.post("/submit")
async def submit_order(req: OrderRequest):
    """Submit an order via Alpaca. Requires SENTINEL_LIVE_TRADING=true."""
    try:
        assert_live_trading_enabled()
    except PermissionError as exc:
        raise HTTPException(403, str(exc))

    from sentinel.see.broker import AlpacaBroker
    from sentinel.core.config import get_settings
    s = get_settings()
    broker = AlpacaBroker(api_key=s.alpaca_api_key, secret_key=s.alpaca_secret_key,
                          paper=not s.is_live_trading_enabled)
    order = await broker.submit_order(
        ticker=req.ticker, side=req.side, qty=req.quantity,
        order_type=req.order_type, limit_price=req.limit_price,
    )
    return order


@router.get("/positions")
async def get_positions():
    """Get current positions from Alpaca."""
    from sentinel.see.broker import AlpacaBroker
    from sentinel.core.config import get_settings
    s = get_settings()
    broker = AlpacaBroker(api_key=s.alpaca_api_key, secret_key=s.alpaca_secret_key,
                          paper=True)
    return await broker.get_positions()
