"""Alpaca broker adapter — paper and live order execution via alpaca-py."""
from __future__ import annotations
import asyncio
from datetime import datetime
from decimal import Decimal
from typing import Optional
from sentinel.core.types import Order, OrderSide, OrderType, OrderStatus
from sentinel.core.logging import get_logger

logger = get_logger(__name__)


class AlpacaBroker:
    """Alpaca Trading API v2 wrapper — supports paper and live modes."""

    def __init__(self, api_key: str, secret_key: str, paper: bool = True) -> None:
        self._api_key = api_key
        self._secret_key = secret_key
        self._paper = paper
        self._client = None

    def _get_client(self):
        if self._client is None:
            from alpaca.trading.client import TradingClient
            self._client = TradingClient(
                api_key=self._api_key,
                secret_key=self._secret_key,
                paper=self._paper,
            )
        return self._client

    async def submit_order(
        self,
        ticker: str,
        side: str,
        qty: float,
        order_type: str = "market",
        limit_price: Optional[float] = None,
        time_in_force: str = "day",
    ) -> dict:
        from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
        from alpaca.trading.enums import OrderSide as AlpacaSide, TimeInForce

        alpaca_side = AlpacaSide.BUY if side.lower() == "buy" else AlpacaSide.SELL
        tif = TimeInForce.DAY

        loop = asyncio.get_event_loop()
        try:
            client = self._get_client()
            if order_type == "limit" and limit_price:
                req = LimitOrderRequest(
                    symbol=ticker.upper(), qty=qty, side=alpaca_side,
                    time_in_force=tif, limit_price=limit_price,
                )
            else:
                req = MarketOrderRequest(
                    symbol=ticker.upper(), qty=qty, side=alpaca_side, time_in_force=tif
                )
            order = await loop.run_in_executor(None, lambda: client.submit_order(req))
            logger.info("Order submitted", ticker=ticker, side=side, qty=qty,
                        order_id=str(order.id), paper=self._paper)
            return {
                "order_id": str(order.id),
                "ticker": ticker,
                "side": side,
                "qty": qty,
                "status": order.status.value,
                "submitted_at": order.submitted_at.isoformat() if order.submitted_at else None,
                "paper": self._paper,
            }
        except Exception as exc:
            logger.error("Order submission failed", ticker=ticker, error=str(exc))
            raise

    async def get_positions(self) -> list[dict]:
        loop = asyncio.get_event_loop()
        try:
            client = self._get_client()
            positions = await loop.run_in_executor(None, client.get_all_positions)
            return [
                {
                    "ticker": p.symbol,
                    "qty": float(p.qty),
                    "avg_entry_price": float(p.avg_entry_price),
                    "market_value": float(p.market_value),
                    "unrealized_pl": float(p.unrealized_pl),
                    "unrealized_plpc": float(p.unrealized_plpc),
                }
                for p in positions
            ]
        except Exception as exc:
            logger.error("Get positions failed", error=str(exc))
            return []

    async def get_account(self) -> dict:
        loop = asyncio.get_event_loop()
        try:
            client = self._get_client()
            acct = await loop.run_in_executor(None, client.get_account)
            return {
                "portfolio_value": float(acct.portfolio_value),
                "cash": float(acct.cash),
                "buying_power": float(acct.buying_power),
                "equity": float(acct.equity),
                "day_trade_count": acct.daytrade_count,
                "paper": self._paper,
            }
        except Exception as exc:
            logger.error("Get account failed", error=str(exc))
            return {}

    async def cancel_order(self, order_id: str) -> bool:
        loop = asyncio.get_event_loop()
        try:
            client = self._get_client()
            await loop.run_in_executor(None, lambda: client.cancel_order_by_id(order_id))
            logger.info("Order cancelled", order_id=order_id)
            return True
        except Exception as exc:
            logger.error("Cancel order failed", order_id=order_id, error=str(exc))
            return False

    async def get_orders(self, status: str = "open") -> list[dict]:
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        loop = asyncio.get_event_loop()
        try:
            client = self._get_client()
            req = GetOrdersRequest(status=QueryOrderStatus(status))
            orders = await loop.run_in_executor(None, lambda: client.get_orders(req))
            return [
                {
                    "order_id": str(o.id),
                    "ticker": o.symbol,
                    "side": o.side.value,
                    "qty": float(o.qty or 0),
                    "status": o.status.value,
                    "type": o.type.value,
                    "submitted_at": o.submitted_at.isoformat() if o.submitted_at else None,
                }
                for o in orders
            ]
        except Exception as exc:
            logger.error("Get orders failed", error=str(exc))
            return []
