"""Portfolio API endpoints"""
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import List, Dict, Optional
from datetime import date, datetime
import logging

from Databases.app_data_db_connection import get_db
from .portfolio_models import PortfolioTrade
from Services.subscription.hierarchy_service import HierarchyService
from .portfolio_schemas import (
    WebhookPayload,
    PortfolioResponse,
    HoldingResponse,
    EquityCurvePoint,
    UserPortfolioResponse,
    UserStrategySummary
)
from .utils import get_strategy_by_run_id, calculate_brokerage, calculate_taxes
from .price_service import PriceService

logger = logging.getLogger(__name__)

portfolio_router = APIRouter(prefix="/api/portfolio", tags=["Portfolio"])


@portfolio_router.post("/webhook/trade-executed")
async def trade_executed_webhook(payload: WebhookPayload, db: Session = Depends(get_db)):
    """
    Webhook endpoint to receive trade execution notifications
    
    Payload example:
    {
        "run_id": "abc123",
        "exchange": "NSE",
        "symbol": "NIFTYBEES",
        "user_id": "user@example.com",
        "order_side": "BUY",
        "product_type": "DELIVERY",
        "clients": {"SRKH1512": "1", "VEDF1515": "5"}
    }
    """
    try:
        logger.info(f"Received trade webhook for run_id={payload.run_id}, symbol={payload.symbol}")
        
        # Parse trade_date from payload if provided, otherwise use today
        if payload.trade_date:
            try:
                trade_date = datetime.strptime(payload.trade_date, "%Y-%m-%d").date()
                logger.info(f"Using provided trade_date: {trade_date}")
            except ValueError as e:
                logger.warning(f"Invalid trade_date format '{payload.trade_date}', using today: {e}")
                trade_date = date.today()
        else:
            trade_date = date.today()
            logger.info(f"No trade_date provided, using today: {trade_date}")
        
        # Get strategy details from run_id
        strategy = get_strategy_by_run_id(payload.run_id, db)
        
        if not strategy:
            logger.error(f"Strategy not found for run_id: {payload.run_id}")
            raise HTTPException(status_code=404, detail=f"Strategy not found for run_id: {payload.run_id}")
        
        # Get price for the symbol
        price = PriceService.get_current_price(payload.symbol, payload.exchange)
        
        if price == 0.0:
            logger.warning(f"Could not fetch price for {payload.symbol}, using fallback")
            price = 100.0  # Fallback price
        
        trades_created = []
        
        # Create trade records for each client
        for client_code, quantity_str in payload.clients.items():
            try:
                quantity = int(quantity_str)
                
                # Calculate costs
                brokerage = calculate_brokerage(quantity, price, payload.order_side)
                taxes = calculate_taxes(quantity, price, payload.order_side)
                
                # Create trade record
                trade = PortfolioTrade(
                    user_email=payload.user_id,
                    run_id=payload.run_id,
                    strategy_name=strategy['strategy_name'],
                    strategy_type=strategy['strategy_type'],
                    client_code=client_code,
                    trade_date=trade_date,  # Use parsed trade_date
                    symbol=payload.symbol,
                    side=payload.order_side,
                    quantity=quantity,
                    price=price,
                    brokerage=brokerage,
                    taxes=taxes
                )
                
                db.add(trade)
                trades_created.append({
                    "client_code": client_code,
                    "symbol": payload.symbol,
                    "side": payload.order_side,
                    "quantity": quantity,
                    "price": price
                })
                
            except Exception as e:
                logger.error(f"Error creating trade for client {client_code}: {e}")
                continue
        
        db.commit()
        
        logger.info(f"Successfully created {len(trades_created)} trade records")
        
        return {
            "success": True,
            "message": f"Created {len(trades_created)} trade records",
            "trades": trades_created
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error processing trade webhook: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error processing webhook: {str(e)}")


@portfolio_router.get("/holdings/{run_id}", response_model=PortfolioResponse)
async def get_portfolio_holdings(
    run_id: str,
    client_code: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Get current portfolio holdings for a run_id
    """
    try:
        run_id = run_id.strip()
        if client_code:
            client_code = client_code.strip()
        logger.info(f"Fetching holdings for run_id={run_id}, client_code={client_code}")
        
        # Build query
        query = text("""
            SELECT 
                symbol,
                SUM(CASE WHEN side = 'BUY' THEN quantity ELSE -quantity END) as net_quantity,
                SUM(CASE WHEN side = 'BUY' THEN quantity * price ELSE 0 END) / 
                    NULLIF(SUM(CASE WHEN side = 'BUY' THEN quantity ELSE 0 END), 0) as avg_price
            FROM portfolio_trades
            WHERE run_id = :run_id
            """ + (" AND client_code = :client_code" if client_code else "") + """
            GROUP BY symbol
            HAVING SUM(CASE WHEN side = 'BUY' THEN quantity ELSE -quantity END) > 0
        """)
        
        params = {"run_id": run_id}
        if client_code:
            params["client_code"] = client_code
        
        results = db.execute(query, params).fetchall()
        
        logger.info(f"Found {len(results)} holdings for run_id={run_id}")
        
        if not results:
            return PortfolioResponse(
                client_code=client_code,
                holdings=[],
                total_value=0.0,
                total_invested=0.0,
                unrealized_pnl=0.0,
                total_return_pct=0.0,
                holdings_count=0
            )
        
        # Get current prices for all symbols
        symbols = [row[0] for row in results]
        current_prices = PriceService.get_latest_prices(symbols)
        
        # Build holdings
        holdings = []
        total_value = 0.0
        total_invested = 0.0
        
        for symbol, quantity, avg_price in results:
            current_price = current_prices.get(symbol, 0.0)
            market_value = quantity * current_price
            total_cost = quantity * avg_price
            unrealized_pnl = market_value - total_cost
            return_pct = (unrealized_pnl / total_cost * 100) if total_cost > 0 else 0.0
            
            holdings.append(HoldingResponse(
                symbol=symbol,
                quantity=int(quantity),
                avg_price=float(avg_price),
                current_price=current_price,
                market_value=market_value,
                total_cost=total_cost,
                unrealized_pnl=unrealized_pnl,
                return_pct=return_pct
            ))
            
            total_value += market_value
            total_invested += total_cost
        
        total_unrealized_pnl = total_value - total_invested
        total_return_pct = (total_unrealized_pnl / total_invested * 100) if total_invested > 0 else 0.0
        
        return PortfolioResponse(
            client_code=client_code,
            holdings=holdings,
            total_value=total_value,
            total_invested=total_invested,
            unrealized_pnl=total_unrealized_pnl,
            total_return_pct=total_return_pct,
            holdings_count=len(holdings)
        )
        
    except Exception as e:
        logger.error(f"Error fetching portfolio holdings: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error fetching holdings: {str(e)}")


@portfolio_router.get("/equity-curve/strategy/{run_id}")
async def get_strategy_equity_curve(
    run_id: str,
    db: Session = Depends(get_db)
):
    """
    Get aggregated equity curve for a strategy (all clients combined).
    Dates are automatically determined from the first and last trade.
    """
    try:
        run_id = run_id.strip()
        logger.info(f"Generating automated strategy equity curve for run_id={run_id}")
        
        # Check if the run_id is actually 'user' - this shouldn't happen with correct ordering but for safety:
        if run_id == "user":
            logger.warning("Caught 'user' as run_id in strategy endpoint")
            return {"success": False, "error": "Invalid run_id 'user'"}

        # DIAGNOSTIC: Check total trades in DB
        total_count = db.execute(text("SELECT count(*) FROM public.portfolio_trades")).scalar()
        logger.info(f"DIAGNOSTIC: Total trades in public.portfolio_trades table: {total_count}")
        
        # Get all trades for this run_id
        query = text("""
            SELECT trade_date, symbol, side, quantity, price
            FROM public.portfolio_trades
            WHERE TRIM(run_id) = :run_id
            ORDER BY trade_date ASC
        """)
        
        trades = db.execute(query, {"run_id": run_id}).fetchall()
        
        if not trades:
            logger.warning(f"No trades found for run_id='{run_id}' (length={len(run_id)})")
            # Try a case-insensitive check
            query_case = text("SELECT trade_date, symbol, side, quantity, price FROM public.portfolio_trades WHERE run_id ILIKE :run_id_pattern ORDER BY trade_date ASC")
            trades = db.execute(query_case, {"run_id_pattern": f"%{run_id}%"}).fetchall()
            if trades:
                logger.info(f"Found {len(trades)} trades using ILIKE match for '%{run_id}%'")
            else:
                return {"success": True, "data": [], "diagnostic": {"total_db_trades": total_count, "searched_run_id": run_id, "table": "public.portfolio_trades"}}
        
        logger.info(f"Found {len(trades)} trades for strategy equity curve")
        
        # Get unique trade dates
        trade_dates = sorted(set(trade[0] for trade in trades))
        
        # Build equity curve
        equity_curve = []
        holdings = {}
        trade_idx = 0
        
        for trade_date in trade_dates:
            # Process all trades up to this date
            while trade_idx < len(trades) and trades[trade_idx][0] <= trade_date:
                t_date, symbol, side, quantity, price = trades[trade_idx]
                
                if symbol not in holdings:
                    holdings[symbol] = {"quantity": 0, "total_cost": 0.0}
                
                if side == "BUY":
                    holdings[symbol]["quantity"] += quantity
                    holdings[symbol]["total_cost"] += quantity * float(price)
                else:  # SELL
                    if holdings[symbol]["quantity"] > 0:
                        cost_per_share = holdings[symbol]["total_cost"] / holdings[symbol]["quantity"]
                        holdings[symbol]["quantity"] -= quantity
                        holdings[symbol]["total_cost"] -= quantity * cost_per_share
                
                trade_idx += 1
            
            # Metrics using historical prices
            total_invested = sum(h["total_cost"] for h in holdings.values())
            portfolio_value = 0.0
            
            symbols_with_holdings = [s for s, h in holdings.items() if h["quantity"] > 0]
            
            if symbols_with_holdings:
                for symbol in symbols_with_holdings:
                    # Fetch historical price for this date
                    hist_price = PriceService.get_price_on_date(symbol, trade_date)
                    # Fallback to last trade price if hist_price is 0
                    if hist_price == 0:
                        # Find the latest trade price for this symbol up to this date
                        for t_date, s, side, q, p in trades:
                            if s == symbol and t_date <= trade_date:
                                hist_price = float(p)
                    
                    portfolio_value += holdings[symbol]["quantity"] * hist_price
            
            return_pct = ((portfolio_value - total_invested) / total_invested * 100) if total_invested > 0 else 0.0
            
            # Benchmark Calculation: Use Nifty 50 Index (^NSEI)
            benchmark_symbol = "^NSEI"
            benchmark_value = 0.0
            benchmark_return_pct = 0.0
            
            first_trade_date = trade_dates[0] if trade_dates else None
            if first_trade_date:
                benchmark_base_price = PriceService.get_price_on_date(benchmark_symbol, first_trade_date)
                if benchmark_base_price > 0:
                    current_benchmark_price = PriceService.get_price_on_date(benchmark_symbol, trade_date)
                    if current_benchmark_price > 0:
                        benchmark_return_pct = ((current_benchmark_price - benchmark_base_price) / benchmark_base_price) * 100
                        benchmark_value = total_invested * (current_benchmark_price / benchmark_base_price)

            equity_curve.append({
                "date": trade_date.isoformat(),
                "portfolio_value": round(portfolio_value, 2),
                "total_invested": round(total_invested, 2),
                "return_pct": round(return_pct, 2),
                "benchmark_value": round(benchmark_value, 2),
                "benchmark_return_pct": round(benchmark_return_pct, 2)
            })
        
        return {"success": True, "data": equity_curve}
        
    except Exception as e:
        logger.error(f"Error generating automated strategy equity curve: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error generating strategy equity curve: {str(e)}")


@portfolio_router.get("/equity-curve/user/{user_email}")
async def get_user_equity_curve(
    user_email: str,
    db: Session = Depends(get_db)
):
    """
    Get aggregated equity curve for a user (all strategies combined).
    Dates are automatically determined from the first and last trade across all strategies.
    """
    try:
        user_email = user_email.strip()
        logger.info(f"Generating automated user equity curve for user: {user_email}")
        
        # Get all trades for this user across all strategies
        query = text("""
            SELECT trade_date, symbol, side, quantity, price
            FROM public.portfolio_trades
            WHERE user_email = :user_email
            ORDER BY trade_date ASC
        """)
        
        trades = db.execute(query, {"user_email": user_email}).fetchall()
        
        if not trades:
            logger.warning(f"No trades found for user: {user_email}")
            return {"success": True, "data": []}
        
        # Build equity curve
        trade_dates = sorted(set(trade[0] for trade in trades))
        equity_curve = []
        holdings = {}
        trade_idx = 0
        
        for trade_date in trade_dates:
            while trade_idx < len(trades) and trades[trade_idx][0] <= trade_date:
                t_date, symbol, side, quantity, price = trades[trade_idx]
                
                if symbol not in holdings:
                    holdings[symbol] = {"quantity": 0, "total_cost": 0.0}
                
                if side == "BUY":
                    holdings[symbol]["quantity"] += quantity
                    holdings[symbol]["total_cost"] += quantity * float(price)
                else:  # SELL
                    if holdings[symbol]["quantity"] > 0:
                        cost_per_share = holdings[symbol]["total_cost"] / holdings[symbol]["quantity"]
                        holdings[symbol]["quantity"] -= quantity
                        holdings[symbol]["total_cost"] -= quantity * cost_per_share
                
                trade_idx += 1
            
            # Metrics using historical prices
            total_invested = sum(h["total_cost"] for h in holdings.values())
            portfolio_value = 0.0
            
            symbols_with_holdings = [s for s, h in holdings.items() if h["quantity"] > 0]
            
            if symbols_with_holdings:
                for symbol in symbols_with_holdings:
                    hist_price = PriceService.get_price_on_date(symbol, trade_date)
                    if hist_price == 0:
                        for t_d, s, side, q, p in trades:
                            if s == symbol and t_d <= trade_date:
                                hist_price = float(p)
                                
                    portfolio_value += holdings[symbol]["quantity"] * hist_price
            
            return_pct = ((portfolio_value - total_invested) / total_invested * 100) if total_invested > 0 else 0.0
            
            # Benchmark Calculation: Use Nifty 50 Index (^NSEI)
            benchmark_symbol = "^NSEI"
            benchmark_value = 0.0
            benchmark_return_pct = 0.0
            
            first_trade_date = trade_dates[0] if trade_dates else None
            if first_trade_date:
                benchmark_base_price = PriceService.get_price_on_date(benchmark_symbol, first_trade_date)
                if benchmark_base_price > 0:
                    current_benchmark_price = PriceService.get_price_on_date(benchmark_symbol, trade_date)
                    if current_benchmark_price > 0:
                        benchmark_return_pct = ((current_benchmark_price - benchmark_base_price) / benchmark_base_price) * 100
                        benchmark_value = total_invested * (current_benchmark_price / benchmark_base_price)

            equity_curve.append({
                "date": trade_date.isoformat(),
                "portfolio_value": round(portfolio_value, 2),
                "total_invested": round(total_invested, 2),
                "return_pct": round(return_pct, 2),
                "benchmark_value": round(benchmark_value, 2),
                "benchmark_return_pct": round(benchmark_return_pct, 2)
            })
        
        return {"success": True, "data": equity_curve}
        
    except Exception as e:
        logger.error(f"Error generating automated user equity curve: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error generating user equity curve: {str(e)}")


@portfolio_router.get("/equity-curve/{run_id}/{client_code}")
async def get_equity_curve(
    run_id: str,
    client_code: str,
    db: Session = Depends(get_db)
):
    """
    Get equity curve (portfolio value over time) for a specific client in a run_id.
    Dates are automatically determined from the first and last trade.
    """
    try:
        run_id = run_id.strip()
        client_code = client_code.strip()
        
        # Safety check: if run_id is 'strategy' or 'user', it means routing is mixed up
        if run_id in ["strategy", "user"]:
            logger.error(f"Routing conflict: caught '{run_id}' as a run_id variable")
            # This shouldn't happen if routes are ordered correctly, but if it does, 
            # we should redirect or handle it.
            return {"success": False, "error": f"Invalid run_id '{run_id}'"}
            
        logger.info(f"Generating automated equity curve for run_id={run_id}, client={client_code}")
        
        # Get all trades for this run_id and client to determine dates and holdings
        query = text("""
            SELECT trade_date, symbol, side, quantity, price
            FROM public.portfolio_trades
            WHERE TRIM(run_id) = :run_id AND TRIM(client_code) = :client_code
            ORDER BY trade_date ASC
        """)
        
        trades = db.execute(query, {"run_id": run_id, "client_code": client_code}).fetchall()
        
        if not trades:
            logger.warning(f"No trades found for run_id='{run_id}', client='{client_code}'")
            # Try a broader search for diagnostics
            broad_query = text("SELECT count(*) FROM public.portfolio_trades WHERE run_id LIKE :run_id_pattern")
            broad_count = db.execute(broad_query, {"run_id_pattern": f"%{run_id}%"}).scalar()
            logger.info(f"Broad search found {broad_count} trades matching run_id pattern")
            return {"success": True, "data": []}
        
        # Build equity curve
        equity_curve = []
        holdings = {}  # {symbol: {quantity, total_cost}}
        processed_trades = set()
        
        # Get unique trade dates
        trade_dates = sorted(set(trade[0] for trade in trades))
        
        for trade_date in trade_dates:
            # Accumulate holdings up to this date
            for t_date, symbol, side, quantity, price in trades:
                if t_date > trade_date:
                    continue
                
                trade_key = f"{t_date}_{symbol}_{side}_{quantity}_{price}"
                if trade_key in processed_trades:
                    continue
                
                if symbol not in holdings:
                    holdings[symbol] = {"quantity": 0, "total_cost": 0.0}
                
                if side == "BUY":
                    holdings[symbol]["quantity"] += quantity
                    holdings[symbol]["total_cost"] += quantity * float(price)
                else:  # SELL
                    if holdings[symbol]["quantity"] > 0:
                        cost_per_share = holdings[symbol]["total_cost"] / holdings[symbol]["quantity"]
                        holdings[symbol]["quantity"] -= quantity
                        holdings[symbol]["total_cost"] -= quantity * cost_per_share
                
                processed_trades.add(trade_key)
            
            # Metrics using historical prices
            total_invested = sum(h["total_cost"] for h in holdings.values())
            portfolio_value = 0.0
            
            symbols_with_holdings = [s for s, h in holdings.items() if h["quantity"] > 0]
            
            if symbols_with_holdings:
                for symbol in symbols_with_holdings:
                    hist_price = PriceService.get_price_on_date(symbol, trade_date)
                    if hist_price == 0:
                        for t_d, s, side, q, p in trades:
                            if s == symbol and t_d <= trade_date:
                                hist_price = float(p)
                    portfolio_value += holdings[symbol]["quantity"] * hist_price
            
            return_pct = ((portfolio_value - total_invested) / total_invested * 100) if total_invested > 0 else 0.0
            
            # Benchmark Calculation: Use Nifty 50 Index (^NSEI)
            benchmark_symbol = "^NSEI"
            benchmark_value = 0.0
            benchmark_return_pct = 0.0
            
            first_trade_date = trade_dates[0] if trade_dates else None
            if first_trade_date:
                benchmark_base_price = PriceService.get_price_on_date(benchmark_symbol, first_trade_date)
                if benchmark_base_price > 0:
                    current_benchmark_price = PriceService.get_price_on_date(benchmark_symbol, trade_date)
                    if current_benchmark_price > 0:
                        benchmark_return_pct = ((current_benchmark_price - benchmark_base_price) / benchmark_base_price) * 100
                        benchmark_value = total_invested * (current_benchmark_price / benchmark_base_price)

            equity_curve.append({
                "date": trade_date.isoformat(),
                "portfolio_value": round(portfolio_value, 2),
                "total_invested": round(total_invested, 2),
                "return_pct": round(return_pct, 2),
                "benchmark_value": round(benchmark_value, 2),
                "benchmark_return_pct": round(benchmark_return_pct, 2)
            })
        
        return {"success": True, "data": equity_curve}
        
    except Exception as e:
        logger.error(f"Error generating automated equity curve: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error generating equity curve: {str(e)}")




@portfolio_router.get("/clients/{run_id}")
async def get_clients_for_run(run_id: str, db: Session = Depends(get_db)):
    """Get list of all client codes that have trades for this run_id"""
    try:
        logger.info(f"Fetching clients for run_id={run_id}")
        
        query = text("""
            SELECT DISTINCT client_code
            FROM portfolio_trades
            WHERE run_id = :run_id
            ORDER BY client_code
        """)
        
        results = db.execute(query, {"run_id": run_id}).fetchall()
        
        logger.info(f"Found {len(results)} clients for run_id={run_id}")
        
        return {
            "run_id": run_id,
            "clients": [row[0] for row in results]
        }
        
    except Exception as e:
        logger.error(f"Error fetching clients: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching clients: {str(e)}")


@portfolio_router.get("/trades/{run_id}")
async def get_trades(
    run_id: str,
    client_code: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get all trades for a run_id (optionally filtered by client_code)"""
    try:
        run_id = run_id.strip()
        if client_code:
            client_code = client_code.strip()
        logger.info(f"Fetching trades for run_id={run_id}, client_code={client_code}")
        
        query = text("""
            SELECT 
                id, trade_date, symbol, side, quantity, price, 
                brokerage, taxes, client_code, created_at
            FROM portfolio_trades
            WHERE run_id = :run_id
            """ + (" AND client_code = :client_code" if client_code else "") + """
            ORDER BY trade_date DESC, created_at DESC
        """)
        
        params = {"run_id": run_id}
        if client_code:
            params["client_code"] = client_code
        
        results = db.execute(query, params).fetchall()
        
        logger.info(f"Found {len(results)} trades for run_id={run_id}")
        
        trades = []
        for row in results:
            trades.append({
                "id": row[0],
                "trade_date": row[1].isoformat() if row[1] else None,
                "symbol": row[2],
                "side": row[3],
                "quantity": row[4],
                "price": float(row[5]),
                "brokerage": float(row[6]) if row[6] else 0.0,
                "taxes": float(row[7]) if row[7] else 0.0,
                "client_code": row[8],
                "created_at": row[9].isoformat() if row[9] else None
            })
        
        return {
            "run_id": run_id,
            "client_code": client_code,
            "trades": trades,
            "count": len(trades)
        }
        
    except Exception as e:
        logger.error(f"Error fetching trades: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching trades: {str(e)}")


@portfolio_router.get("/user/{user_email}", response_model=UserPortfolioResponse)
async def get_user_portfolio_summary(
    user_email: str,
    db: Session = Depends(get_db)
):
    """
    Get high-level portfolio summary for a user across all their strategies
    """
    try:
        user_email = user_email.strip()
        logger.info(f"Fetching hierarchical portfolio summary for: {user_email}")
        
        # 0. Get all accessible users (self + descendants)
        accessible_users = HierarchyService.get_accessible_users(user_email, db)
        logger.info(f"Found {len(accessible_users)} accessible users for hierarchy summary")
        
        # 1. Get all unique strategies (run_ids) for all accessible users
        query_strategies = text("""
            SELECT DISTINCT pt.run_id, pt.strategy_name, pt.strategy_type, pt.user_email, ud.user_name
            FROM portfolio_trades pt
            LEFT JOIN user_details ud ON pt.user_email = ud.user_email
            WHERE pt.user_email IN :user_emails
        """)
        
        strategy_rows = db.execute(query_strategies, {"user_emails": tuple(accessible_users)}).fetchall()
        
        if not strategy_rows:
            return UserPortfolioResponse(
                user_email=user_email,
                total_invested=0.0,
                total_value=0.0,
                total_pnl=0.0,
                total_return_pct=0.0,
                strategies_count=0,
                strategies=[]
            )
        
        strategies_summary = []
        overall_invested = 0.0
        overall_value = 0.0
        
        # 2. For each strategy, calculate performance
        for run_id, strategy_name, strategy_type, owner_email, owner_name in strategy_rows:
            # Get client_info from saved_instances table first
            query_client_info = text("""
                SELECT client_info
                FROM saved_instances
                WHERE run_id = :run_id
            """)
            
            client_info_result = db.execute(query_client_info, {"run_id": run_id}).fetchone()
            
            # Build client details list from client_info JSON
            clients_list = []
            client_info = None
            
            if client_info_result and client_info_result[0]:
                client_info = client_info_result[0]
            else:
                # Fallback: Check legacy strategy tables for client_information_json
                legacy_tables = [
                    'etf_saved_strategy',
                    'stock_saved_strategy',
                    'rs_etf_instance',
                    'rs_stock_instance'
                ]
                
                for table in legacy_tables:
                    try:
                        query_legacy = text(f"""
                            SELECT client_information_json
                            FROM {table}
                            WHERE run_id = :run_id
                        """)
                        legacy_result = db.execute(query_legacy, {"run_id": run_id}).fetchone()
                        
                        if legacy_result and legacy_result[0]:
                            client_info = legacy_result[0]
                            logger.info(f"Found client_info in {table} for run_id={run_id}")
                            break
                    except Exception as e:
                        logger.warning(f"Error checking {table}: {e}")
                        continue
            
            if client_info:
                # client_info format: {"CLIENT001": "10000", "CLIENT002": "20000"}
                # or it might be a JSON string that needs parsing
                if isinstance(client_info, str):
                    try:
                        import json
                        client_info = json.loads(client_info)
                    except:
                        logger.warning(f"Could not parse client_info JSON for run_id={run_id}")
                        client_info = {}
                
                for client_code, capital_str in client_info.items():
                    from .portfolio_schemas import ClientDetail
                    try:
                        # Clean currency string: remove ₹, $, commas, and spaces
                        cleaned_capital = str(capital_str).replace('₹', '').replace('$', '').replace(',', '').strip()
                        clients_list.append(ClientDetail(
                            client_code=client_code,
                            capital=float(cleaned_capital)
                        ))
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Could not convert capital to float for client {client_code}: {e}")
                        continue
            
            # Get current holdings for this strategy to calculate totals
            query_holdings = text("""
                SELECT 
                    symbol,
                    SUM(CASE WHEN side = 'BUY' THEN quantity ELSE -quantity END) as net_quantity,
                    SUM(CASE WHEN side = 'BUY' THEN quantity * price ELSE 0 END) / 
                        NULLIF(SUM(CASE WHEN side = 'BUY' THEN quantity ELSE 0 END), 0) as avg_price
                FROM portfolio_trades
                WHERE run_id = :run_id AND user_email = :owner_email
                GROUP BY symbol
                HAVING SUM(CASE WHEN side = 'BUY' THEN quantity ELSE -quantity END) > 0
            """)
            
            holding_rows = db.execute(query_holdings, {
                "run_id": run_id, 
                "owner_email": owner_email
            }).fetchall()
            
            strategy_invested = 0.0
            strategy_value = 0.0
            holdings_count = len(holding_rows)
            
            if holding_rows:
                symbols = [row[0] for row in holding_rows]
                current_prices = PriceService.get_latest_prices(symbols)
                
                for symbol, quantity, avg_price in holding_rows:
                    price = current_prices.get(symbol, 0.0)
                    strategy_invested += float(quantity) * float(avg_price)
                    strategy_value += float(quantity) * price

            
            pnl = strategy_value - strategy_invested
            ret_pct = (pnl / strategy_invested * 100) if strategy_invested > 0 else 0.0
            
            # 3. Get running status
            strategy_details = get_strategy_by_run_id(run_id, db)
            running_status = strategy_details['status'] if strategy_details else 'deploy'
            
            strategies_summary.append(UserStrategySummary(
                run_id=run_id,
                strategy_name=strategy_name,
                strategy_type=strategy_type,
                total_invested=round(strategy_invested, 2),
                market_value=round(strategy_value, 2),
                pnl=round(pnl, 2),
                return_pct=round(ret_pct, 2),
                holdings_count=holdings_count,
                running_status=running_status,
                owner_email=owner_email,
                owner_name=owner_name,
                clients=clients_list
            ))
            
            overall_invested += strategy_invested
            overall_value += strategy_value
        
        overall_pnl = overall_value - overall_invested
        overall_ret_pct = (overall_pnl / overall_invested * 100) if overall_invested > 0 else 0.0
        
        return UserPortfolioResponse(
            user_email=user_email,
            total_invested=round(overall_invested, 2),
            total_value=round(overall_value, 2),
            total_pnl=round(overall_pnl, 2),
            total_return_pct=round(overall_ret_pct, 2),
            strategies_count=len(strategies_summary),
            strategies=strategies_summary
        )


        
    except Exception as e:
        logger.error(f"Error fetching user portfolio summary: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error fetching user summary: {str(e)}")


@portfolio_router.get("/strategy/{run_id}", response_model=UserStrategySummary)
async def get_strategy_summary(
    run_id: str,
    db: Session = Depends(get_db)
):
    """
    Get high-level summary for a specific strategy run
    """
    try:
        run_id = run_id.strip()
        logger.info(f"Fetching strategy summary for run_id: {run_id}")
        
        # 1. Get strategy details from run_id
        strategy = get_strategy_by_run_id(run_id, db)
        
        if not strategy:
            logger.error(f"Strategy not found for run_id: {run_id}")
            raise HTTPException(status_code=404, detail=f"Strategy not found for run_id: {run_id}")
        
        # 2. Get current holdings for this strategy
        query_holdings = text("""
            SELECT 
                symbol,
                SUM(CASE WHEN side = 'BUY' THEN quantity ELSE -quantity END) as net_quantity,
                SUM(CASE WHEN side = 'BUY' THEN quantity * price ELSE 0 END) / 
                    NULLIF(SUM(CASE WHEN side = 'BUY' THEN quantity ELSE 0 END), 0) as avg_price
            FROM portfolio_trades
            WHERE run_id = :run_id
            GROUP BY symbol
            HAVING SUM(CASE WHEN side = 'BUY' THEN quantity ELSE -quantity END) > 0
        """)
        
        holding_rows = db.execute(query_holdings, {"run_id": run_id}).fetchall()
        
        strategy_invested = 0.0
        strategy_value = 0.0
        holdings_count = len(holding_rows)
        
        if holding_rows:
            symbols = [row[0] for row in holding_rows]
            current_prices = PriceService.get_latest_prices(symbols)
            
            for symbol, quantity, avg_price in holding_rows:
                price = current_prices.get(symbol, 0.0)
                strategy_invested += float(quantity) * float(avg_price)
                strategy_value += float(quantity) * price
        
        pnl = strategy_value - strategy_invested
        ret_pct = (pnl / strategy_invested * 100) if strategy_invested > 0 else 0.0
        
        # Get owner name
        owner_email = strategy.get('user_id')
        owner_name = None
        if owner_email:
            owner_name_row = db.execute(
                text("SELECT user_name FROM user_details WHERE user_email = :email"),
                {"email": owner_email}
            ).fetchone()
            if owner_name_row:
                owner_name = owner_name_row[0]

        return UserStrategySummary(
            run_id=run_id,
            strategy_name=strategy['strategy_name'],
            strategy_type=strategy['strategy_type'],
            total_invested=round(strategy_invested, 2),
            market_value=round(strategy_value, 2),
            pnl=round(pnl, 2),
            return_pct=round(ret_pct, 2),
            holdings_count=holdings_count,
            running_status=strategy.get('status', 'deploy'),
            owner_email=owner_email,
            owner_name=owner_name
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching strategy summary: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error fetching strategy summary: {str(e)}")




@portfolio_router.get("/debug/list-run-ids")
async def debug_list_run_ids(db: Session = Depends(get_db)):
    """Debug endpoint to see what run_ids the API can see"""
    try:
        # Check total count first
        total_count = db.execute(text("SELECT count(*) FROM public.portfolio_trades")).scalar()
        
        query = text("""
            SELECT run_id, count(*), min(trade_date), max(trade_date)
            FROM public.portfolio_trades
            GROUP BY run_id
        """)
        results = db.execute(query).fetchall()
        run_ids = []
        for row in results:
            run_ids.append({
                "run_id": row[0],
                "count": row[1],
                "dates": f"[{row[2]}] to [{row[3]}]"
            })
            
        # Also try a search for the problematic ones
        search_query = text("SELECT DISTINCT run_id FROM public.portfolio_trades WHERE run_id LIKE '%Rotation_Payout%'")
        search_results = db.execute(search_query).fetchall()
        matching_ids = [r[0] for r in search_results]
        
        return {
            "success": True, 
            "total_trades_in_db": total_count,
            "visible_run_ids_count": len(run_ids),
            "run_ids_summary": run_ids[:10], # Limit to 10 for safety
            "ids_matching_payout": matching_ids
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
