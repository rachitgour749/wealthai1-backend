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
    UserStrategySummary,
    ClientDetail
)
from .utils import (
    get_strategy_by_run_id, 
    calculate_brokerage, 
    calculate_taxes, 
    get_client_allocations,
    calculate_cagr,
    calculate_xirr
)
from .price_service import PriceService

# Configure logging for portfolio service
logger = logging.getLogger('Services.portfolio.portfolio_api')
logger.setLevel(logging.DEBUG)

# Add console handler if not already present
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

logger.info("Portfolio API module loaded with DEBUG logging enabled")

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
            # Even if no holdings, we might have allocations (cash only)
            allocations = get_client_allocations(run_id, db)
            total_allocated = sum(allocations.values())
            
            # If a specific client is requested, filter allo
            if client_code and client_code in allocations:
                total_allocated = allocations[client_code]
            elif client_code:
                total_allocated = 0.0
                
            # If we have no trades, Net Flow is 0, so Cash = Allocated
            return PortfolioResponse(
                client_code=client_code,
                holdings=[],
                total_value=0.0,
                total_invested=0.0,
                unrealized_pnl=0.0,
                total_return_pct=0.0,
                holdings_count=0,
                total_allocated_capital=total_allocated,
                cash_balance=total_allocated,
                aum=total_allocated
            )
        
        # Get current prices for all symbols
        symbols = [row[0] for row in results]
        current_prices = PriceService.get_latest_prices(symbols)
        
        # Build holdings
        holdings = []
        total_market_value = 0.0
        total_invested_cost_basis = 0.0
        
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
            
            total_market_value += market_value
            total_invested_cost_basis += total_cost
        
        # Calculate Cash and AUM
        # 1. Get total allocated capital
        allocations = get_client_allocations(run_id, db)
        total_allocated = sum(allocations.values())
        if client_code:
            total_allocated = allocations.get(client_code, 0.0)
            
        # 2. Calculate Net Flow (Money In - Money Out)
        # Net Flow = (Buy Value + Costs) - (Sell Value - Costs)
        flow_query = text("""
            SELECT 
                SUM(CASE 
                    WHEN side = 'BUY' THEN (quantity * price) + COALESCE(brokerage, 0) + COALESCE(taxes, 0)
                    ELSE 0 
                END) as total_buy_cost,
                SUM(CASE 
                    WHEN side = 'SELL' THEN (quantity * price) - COALESCE(brokerage, 0) - COALESCE(taxes, 0)
                    ELSE 0 
                END) as total_sell_proceeds
            FROM portfolio_trades
            WHERE run_id = :run_id
            """ + (" AND client_code = :client_code" if client_code else "") + """
        """)
        
        flow_res = db.execute(flow_query, params).fetchone()
        buy_cost = float(flow_res[0] or 0)
        sell_proceeds = float(flow_res[1] or 0)
        
        net_money_spent = buy_cost - sell_proceeds
        cash_balance = total_allocated - net_money_spent
        
        # AUM = Market Value of Holdings + Cash Balance
        aum = total_market_value + cash_balance
        
        total_unrealized_pnl = total_market_value - total_invested_cost_basis
        total_return_pct = (total_unrealized_pnl / total_invested_cost_basis * 100) if total_invested_cost_basis > 0 else 0.0
        
        return PortfolioResponse(
            client_code=client_code,
            holdings=holdings,
            total_value=total_market_value,
            total_invested=total_invested_cost_basis,
            unrealized_pnl=total_unrealized_pnl,
            total_return_pct=total_return_pct,
            holdings_count=len(holdings),
            total_allocated_capital=total_allocated,
            cash_balance=cash_balance,
            aum=aum
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
        db.rollback()
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
        db.rollback()
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
        db.rollback()
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
            SELECT DISTINCT pt.run_id, pt.strategy_name, pt.strategy_type, pt.user_email, ud.user_name, si.status
            FROM portfolio_trades pt
            LEFT JOIN user_details ud ON pt.user_email = ud.user_email
            LEFT JOIN saved_instances si ON pt.run_id = si.run_id
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
        
        # Global Aggregators
        overall_invested = 0.0
        overall_market_value = 0.0
        overall_allocated = 0.0
        overall_cash = 0.0
        overall_aum = 0.0
        
        # 2. For each strategy, calculate performance
        for run_id, strategy_name, strategy_type, owner_email, owner_name, strat_status in strategy_rows:
            # A. Get Client Allocations
            allocations = get_client_allocations(run_id, db)
            
            # Get strategy execution count for proper allocation calculation
            rounds_query = text("""
                SELECT COUNT(DISTINCT trade_date)
                FROM portfolio_trades
                WHERE run_id = :run_id AND side = 'BUY'
            """)
            strategy_execution_count = db.execute(rounds_query, {"run_id": run_id}).scalar() or 1
            
            # B. Get Trade Flows per Client (for Cash calculation)
            flow_query = text("""
                SELECT 
                    client_code,
                    SUM(CASE 
                        WHEN side = 'BUY' THEN (quantity * price) + COALESCE(brokerage, 0) + COALESCE(taxes, 0)
                        ELSE 0 
                    END) as buy_cost,
                    SUM(CASE 
                        WHEN side = 'SELL' THEN (quantity * price) - COALESCE(brokerage, 0) - COALESCE(taxes, 0)
                        ELSE 0 
                    END) as sell_proceeds
                FROM portfolio_trades
                WHERE run_id = :run_id
                GROUP BY client_code
            """)
            flow_rows = db.execute(flow_query, {"run_id": run_id}).fetchall()
            client_flows = {row[0]: (float(row[1] or 0), float(row[2] or 0)) for row in flow_rows}
            
            # C. Get Holdings per Client (for Market Value)
            holdings_query = text("""
                SELECT 
                    client_code,
                    symbol,
                    SUM(CASE WHEN side = 'BUY' THEN quantity ELSE -quantity END) as net_qty,
                    SUM(CASE WHEN side = 'BUY' THEN quantity * price ELSE 0 END) / 
                        NULLIF(SUM(CASE WHEN side = 'BUY' THEN quantity ELSE 0 END), 0) as avg_buy_price
                FROM portfolio_trades
                WHERE run_id = :run_id
                GROUP BY client_code, symbol
                HAVING SUM(CASE WHEN side = 'BUY' THEN quantity ELSE -quantity END) > 0
            """)
            holdings_rows = db.execute(holdings_query, {"run_id": run_id}).fetchall()
            
            # Group holdings by client
            client_holdings = {}
            active_symbols = set()
            for r_client, r_symbol, r_qty, r_avg in holdings_rows:
                if r_client not in client_holdings:
                    client_holdings[r_client] = []
                client_holdings[r_client].append((r_symbol, r_qty, r_avg))
                active_symbols.add(r_symbol)
                
            # Fetch prices for all active symbols
            current_prices = PriceService.get_latest_prices(list(active_symbols))
            
            # D. Get all trades for CAGR/XIRR calculation
            trades_for_xirr_query = text("""
                SELECT client_code, trade_date, side, quantity, price, 
                       COALESCE(brokerage, 0) as brokerage, COALESCE(taxes, 0) as taxes
                FROM portfolio_trades
                WHERE run_id = :run_id
                ORDER BY trade_date ASC
            """)
            all_trades = db.execute(trades_for_xirr_query, {"run_id": run_id}).fetchall()
            
            # Group trades by client for XIRR
            client_trades = {}
            first_trade_date = None
            for t_client, t_date, t_side, t_qty, t_price, t_brok, t_tax in all_trades:
                if t_client not in client_trades:
                    client_trades[t_client] = []
                client_trades[t_client].append((t_date, t_side, t_qty, t_price, t_brok, t_tax))
                
                if first_trade_date is None or t_date < first_trade_date:
                    first_trade_date = t_date
            
            # E. Build ClientDetails list
            clients_list = []
            
            # Strategy Aggregators
            strat_allocated = 0.0
            strat_cash = 0.0
            strat_market_value = 0.0
            strat_invested_basis = 0.0
            strat_holdings_count = 0
            
            # For strategy-level XIRR
            strategy_cash_flows = []
            strategy_cash_flow_dates = []
            
            # Iterate over all clients known in allocations OR trades
            all_clients = set(allocations.keys()) | set(client_flows.keys()) | set(client_holdings.keys())
            
            for client_code in all_clients:
                # 1. Allocation
                period_capital = allocations.get(client_code, 0.0)
                client_total_allocated = strategy_execution_count * period_capital
                
                # 2. Cash Balance calculation
                c_buy, c_sell = client_flows.get(client_code, (0.0, 0.0))
                net_spent = c_buy - c_sell
                
                # 3. Market Value calculation
                client_mv = 0.0
                client_cost_basis = 0.0
                
                if client_code in client_holdings:
                    for h_sym, h_qty, h_avg in client_holdings[client_code]:
                        price = current_prices.get(h_sym, 0.0)
                        client_mv += float(h_qty) * price
                        client_cost_basis += float(h_qty) * float(h_avg)
                        strat_holdings_count += 1
                
                # Cash = Total Allocated - Invested (Cost Basis)
                client_cash = client_total_allocated - client_cost_basis
                
                # 4. AUM
                client_aum = client_cash + client_mv
                
                # 5. Metrics
                client_pnl = client_mv - client_cost_basis
                client_ret = (client_pnl / client_cost_basis * 100) if client_cost_basis > 0 else 0.0
                
                # 6. Calculate CAGR and XIRR for this client
                client_cagr = 0.0
                client_xirr = 0.0
                
                if client_code in client_trades and len(client_trades[client_code]) > 0:
                    client_first_trade = min(t[0] for t in client_trades[client_code])
                    
                    # CAGR: Use total allocated capital as initial value
                    client_cagr = calculate_cagr(
                        initial_value=client_total_allocated,
                        current_value=client_aum,
                        start_date=client_first_trade,
                        end_date=date.today()
                    )
                    
                    # XIRR: Build cash flows
                    client_cf = []
                    client_cf_dates = []
                    
                    for t_date, t_side, t_qty, t_price, t_brok, t_tax in client_trades[client_code]:
                        if t_side == 'BUY':
                            cf = -1 * float(float(t_qty) * float(t_price) + float(t_brok) + float(t_tax))
                        else:  # SELL
                            cf = float(float(t_qty) * float(t_price) - float(t_brok) - float(t_tax))
                        
                        client_cf.append(cf)
                        client_cf_dates.append(t_date)
                        
                        # Add to strategy-level cash flows
                        strategy_cash_flows.append(cf)
                        strategy_cash_flow_dates.append(t_date)
                    
                    # Add current AUM as final positive cash flow
                    client_cf.append(client_aum)
                    client_cf_dates.append(date.today())
                    
                    client_xirr = calculate_xirr(client_cf, client_cf_dates)
                
                # Add to lists and aggregators
                strat_allocated += client_total_allocated
                strat_cash += client_cash
                strat_market_value += client_mv
                strat_invested_basis += client_cost_basis
                
                clients_list.append(ClientDetail(
                    client_code=client_code,
                    capital=period_capital,
                    total_allocated_capital=round(client_total_allocated, 2),
                    market_value=round(client_mv, 2),
                    invested_amount=round(client_cost_basis, 2),
                    cash_balance=round(client_cash, 2),
                    aum=round(client_aum, 2),
                    pnl=round(client_pnl, 2),
                    return_pct=round(client_ret, 2),
                    cagr=client_cagr,
                    xirr=client_xirr
                ))
            
            # E. Strategy Totals
            strat_aum = strat_cash + strat_market_value
            strat_pnl = strat_market_value - strat_invested_basis
            strat_ret = (strat_pnl / strat_invested_basis * 100) if strat_invested_basis > 0 else 0.0
            
            # Calculate Strategy-level CAGR and XIRR
            strategy_cagr = 0.0
            strategy_xirr = 0.0
            
            if first_trade_date:
                logger.info(f"[STRATEGY CAGR] Calculating for run_id: {run_id}")
                logger.debug(f"[STRATEGY CAGR] First trade date: {first_trade_date}")
                logger.debug(f"[STRATEGY CAGR] Number of clients: {len(all_clients)}")
                
                # CAGR for strategy
                initial_strategy_value = sum(allocations.values())
                logger.debug(f"[STRATEGY CAGR] Initial value (sum of per-period allocations): ₹{initial_strategy_value:,.2f}")
                logger.debug(f"[STRATEGY CAGR] Current AUM: ₹{strat_aum:,.2f}")
                logger.debug(f"[STRATEGY CAGR] Total allocated capital: ₹{strat_allocated:,.2f}")
                logger.debug(f"[STRATEGY CAGR] Market value: ₹{strat_market_value:,.2f}")
                logger.debug(f"[STRATEGY CAGR] Cash balance: ₹{strat_cash:,.2f}")
                
                strategy_cagr = calculate_cagr(
                    initial_value=initial_strategy_value,
                    current_value=strat_aum,
                    start_date=first_trade_date,
                    end_date=date.today()
                )
                logger.info(f"[STRATEGY CAGR] Final CAGR: {strategy_cagr}%")
                
                # XIRR for strategy
                if len(strategy_cash_flows) > 0:
                    logger.debug(f"[STRATEGY XIRR] Number of cash flows: {len(strategy_cash_flows)}")
                    logger.debug(f"[STRATEGY XIRR] Total cash outflow: ₹{sum(cf for cf in strategy_cash_flows if cf < 0):,.2f}")
                    logger.debug(f"[STRATEGY XIRR] Total cash inflow: ₹{sum(cf for cf in strategy_cash_flows if cf > 0):,.2f}")
                    
                    strategy_cash_flows.append(strat_aum)
                    strategy_cash_flow_dates.append(date.today())
                    logger.debug(f"[STRATEGY XIRR] Added final AUM: ₹{strat_aum:,.2f}")
                    
                    strategy_xirr = calculate_xirr(strategy_cash_flows, strategy_cash_flow_dates)
                    logger.info(f"[STRATEGY XIRR] Final XIRR: {strategy_xirr}%")
            else:
                logger.warning(f"[STRATEGY CAGR] No trades found for run_id: {run_id}")
            
            strategies_summary.append(UserStrategySummary(
                run_id=run_id,
                strategy_name=strategy_name,
                strategy_type=strategy_type,
                total_invested=round(strat_invested_basis, 2),
                market_value=round(strat_market_value, 2),
                pnl=round(strat_pnl, 2),
                return_pct=round(strat_ret, 2),
                total_allocated_capital=round(strat_allocated, 2),
                cash_balance=round(strat_cash, 2),
                aum=round(strat_aum, 2),
                cagr=strategy_cagr,
                xirr=strategy_xirr,
                holdings_count=strat_holdings_count,
                running_status=strat_status or "deploy",
                owner_email=owner_email,
                owner_name=owner_name,
                clients=clients_list
            ))
            
            overall_allocated += strat_allocated
            overall_cash += strat_cash
            overall_market_value += strat_market_value
            overall_invested += strat_invested_basis
            overall_aum += strat_aum
            
        overall_pnl = overall_market_value - overall_invested
        overall_ret = (overall_pnl / overall_invested * 100) if overall_invested > 0 else 0.0
        
        # Calculate User-level CAGR and XIRR
        user_cagr = 0.0
        user_xirr = 0.0
        
        # Get all trades for this user to calculate CAGR/XIRR
        user_trades_query = text("""
            SELECT trade_date, side, quantity, price, 
                   COALESCE(brokerage, 0) as brokerage, COALESCE(taxes, 0) as taxes
            FROM portfolio_trades
            WHERE user_email IN :user_emails
            ORDER BY trade_date ASC
        """)
        user_all_trades = db.execute(user_trades_query, {"user_emails": tuple(accessible_users)}).fetchall()
        
        if len(user_all_trades) > 0:
            first_user_trade_date = user_all_trades[0][0]
            
            # CAGR: Use sum of all first-period allocations as starting value
            # Sum up the per-period capital from all strategies (not total allocated which includes SIP accumulation)
            user_initial_value = 0.0
            for strategy in strategies_summary:
                # Get the per-period capital for each client in this strategy
                for client in strategy.clients:
                    user_initial_value += client.capital
            
            if user_initial_value > 0:
                user_cagr = calculate_cagr(
                    initial_value=user_initial_value,
                    current_value=overall_aum,
                    start_date=first_user_trade_date,
                    end_date=date.today()
                )
            
            # XIRR: Build cash flows from all trades
            user_cf = []
            user_cf_dates = []
            
            for t_date, t_side, t_qty, t_price, t_brok, t_tax in user_all_trades:
                if t_side == 'BUY':
                    cf = -1 * float(float(t_qty) * float(t_price) + float(t_brok) + float(t_tax))
                else:  # SELL
                    cf = float(float(t_qty) * float(t_price) - float(t_brok) - float(t_tax))
                
                user_cf.append(cf)
                user_cf_dates.append(t_date)
            
            # Add current AUM as final positive cash flow
            user_cf.append(overall_aum)
            user_cf_dates.append(date.today())
            
            user_xirr = calculate_xirr(user_cf, user_cf_dates)
        
        return UserPortfolioResponse(
            user_email=user_email,
            total_invested=round(overall_invested, 2),
            total_value=round(overall_market_value, 2),
            total_pnl=round(overall_pnl, 2),
            total_return_pct=round(overall_ret, 2),
            total_allocated_capital=round(overall_allocated, 2),
            total_cash_balance=round(overall_cash, 2),
            total_aum=round(overall_aum, 2),
            cagr=user_cagr,
            xirr=user_xirr,
            strategies_count=len(strategies_summary),
            strategies=strategies_summary
        )


        
    except Exception as e:
        logger.error(f"Error fetching user portfolio summary: {e}")
        import traceback
        logger.error(traceback.format_exc())
        # Rollback the transaction to prevent "transaction is aborted" errors
        db.rollback()
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
        
        # 2. Get Client Allocations
        allocations = get_client_allocations(run_id, db)
        
        # 3. Determine Strategy Execution Count (Global Rounds)
        # We need to know how many times the strategy has "run" (executed trades)
        # We count distinct Buy dates for the entire strategy run_id
        # This assumes that on a "run day", all active clients get an allocation.
        rounds_query = text("""
            SELECT COUNT(DISTINCT trade_date)
            FROM portfolio_trades
            WHERE run_id = :run_id AND side = 'BUY'
        """)
        strategy_execution_count = db.execute(rounds_query, {"run_id": run_id}).scalar() or 0
        
        logger.info(f"[STRATEGY EXECUTION COUNT] run_id: {run_id}")
        logger.debug(f"[STRATEGY EXECUTION COUNT] Distinct BUY trade dates: {strategy_execution_count}")
        
        # If no buy trades yet (just started), count is 0, but allocation implies 1st installment ready
        # However, for cash calculation, if we haven't spent anything, we assume 1 installment if running
        if strategy_execution_count == 0:
             strategy_execution_count = 1
             logger.warning(f"[STRATEGY EXECUTION COUNT] No BUY trades found, defaulting to 1")
        
        logger.info(f"[STRATEGY EXECUTION COUNT] Final count: {strategy_execution_count}")
        # Get Flows
        flow_query = text("""
            SELECT 
                client_code,
                SUM(CASE 
                    WHEN side = 'BUY' THEN (quantity * price) + COALESCE(brokerage, 0) + COALESCE(taxes, 0)
                    ELSE 0 
                END) as buy_cost,
                SUM(CASE 
                    WHEN side = 'SELL' THEN (quantity * price) - COALESCE(brokerage, 0) - COALESCE(taxes, 0)
                    ELSE 0 
                END) as sell_proceeds
            FROM portfolio_trades
            WHERE run_id = :run_id
            GROUP BY client_code
        """)
        flow_rows = db.execute(flow_query, {"run_id": run_id}).fetchall()
        client_flows = {row[0]: (float(row[1] or 0), float(row[2] or 0)) for row in flow_rows}
        
        # 4. Get Holdings per Client (for Market Value)
        holdings_query = text("""
            SELECT 
                client_code,
                symbol,
                SUM(CASE WHEN side = 'BUY' THEN quantity ELSE -quantity END) as net_qty,
                SUM(CASE WHEN side = 'BUY' THEN quantity * price ELSE 0 END) / 
                    NULLIF(SUM(CASE WHEN side = 'BUY' THEN quantity ELSE 0 END), 0) as avg_buy_price
            FROM portfolio_trades
            WHERE run_id = :run_id
            GROUP BY client_code, symbol
            HAVING SUM(CASE WHEN side = 'BUY' THEN quantity ELSE -quantity END) > 0
        """)
        holdings_rows = db.execute(holdings_query, {"run_id": run_id}).fetchall()
        
        # Group holdings by client
        client_holdings = {}
        active_symbols = set()
        for r_client, r_symbol, r_qty, r_avg in holdings_rows:
            if r_client not in client_holdings:
                client_holdings[r_client] = []
            client_holdings[r_client].append((r_symbol, r_qty, r_avg))
            active_symbols.add(r_symbol)
            
        current_prices = PriceService.get_latest_prices(list(active_symbols))
        
        # Aggregators
        total_allocated = 0.0
        total_cash = 0.0
        total_mv = 0.0
        total_cost_basis = 0.0
        holdings_count = 0
        
        # For CAGR/XIRR calculation
        first_trade_date = None
        strategy_cash_flows = []
        strategy_cash_flow_dates = []
        
        # Get all trades with dates for XIRR
        trades_for_xirr_query = text("""
            SELECT client_code, trade_date, side, quantity, price, 
                   COALESCE(brokerage, 0) as brokerage, COALESCE(taxes, 0) as taxes
            FROM portfolio_trades
            WHERE run_id = :run_id
            ORDER BY trade_date ASC
        """)
        all_trades = db.execute(trades_for_xirr_query, {"run_id": run_id}).fetchall()
        
        # Group trades by client for XIRR
        client_trades = {}
        for t_client, t_date, t_side, t_qty, t_price, t_brok, t_tax in all_trades:
            if t_client not in client_trades:
                client_trades[t_client] = []
            client_trades[t_client].append((t_date, t_side, t_qty, t_price, t_brok, t_tax))
            
            # Track first trade date
            if first_trade_date is None or t_date < first_trade_date:
                first_trade_date = t_date
        
        clients_list = []
        all_clients = set(allocations.keys()) | set(client_flows.keys()) | set(client_holdings.keys())
        
        for client_code in all_clients:
            period_capital = round(float(allocations.get(client_code, 0.0)), 2)
            
            # Use the global strategy execution count for multiplier
            # Fallback: if a specific client has MORE unique buy days than the global logic (unlikely but safe), we could use max
            # For now, sticking to user's logic: Strategy Executed 7 times -> 7 * Capital
            
            client_total_allocated = round(float(strategy_execution_count) * float(period_capital), 2)
            
            logger.debug(f"[CLIENT ALLOCATION] Client: {client_code}")
            logger.debug(f"[CLIENT ALLOCATION]   Per-period capital: ₹{period_capital:.2f}")
            logger.debug(f"[CLIENT ALLOCATION]   Execution count: {strategy_execution_count}")
            logger.debug(f"[CLIENT ALLOCATION]   Total allocated: ₹{client_total_allocated:.2f}")
            
            client_mv = 0.0
            client_cb = 0.0
            
            if client_code in client_holdings:
                for h_sym, h_qty, h_avg in client_holdings[client_code]:
                    price = round(float(current_prices.get(h_sym, 0.0)), 2)
                    client_mv += round(float(h_qty) * price, 2)
                    # Invested Amount = Sum(Qty * AvgPrice) i.e. Cost Basis
                    client_cb += round(float(h_qty) * float(h_avg), 2)
                    holdings_count += 1
            
            # Round to 2 decimals
            client_mv = round(client_mv, 2)
            client_cb = round(client_cb, 2)
            
            # User Calculated Logic: Cash = Total Allocated - Total Invested (Cost Basis)
            # This represents "Capital Remaining to be Invested" ignoring realized PnL
            client_cash = round(client_total_allocated - client_cb, 2)
                    
            client_aum = round(client_cash + client_mv, 2)
            client_pnl = round(client_mv - client_cb, 2)
            client_ret = round((client_pnl / client_cb * 100) if client_cb > 0 else 0.0, 2)
            
            # Calculate CAGR for this client
            client_cagr = 0.0
            client_xirr = 0.0
            
            if client_code in client_trades and len(client_trades[client_code]) > 0:
                client_first_trade = min(t[0] for t in client_trades[client_code])
                
                logger.info(f"[CLIENT CAGR] Calculating for client: {client_code}")
                logger.debug(f"[CLIENT CAGR] Per-period capital: ₹{period_capital:,.2f}")
                logger.debug(f"[CLIENT CAGR] Total allocated capital: ₹{client_total_allocated:,.2f}")
                logger.debug(f"[CLIENT CAGR] Invested amount (cost basis): ₹{client_cb:,.2f}")
                logger.debug(f"[CLIENT CAGR] Market value: ₹{client_mv:,.2f}")
                logger.debug(f"[CLIENT CAGR] Cash balance: ₹{client_cash:,.2f}")
                logger.debug(f"[CLIENT CAGR] Current AUM: ₹{client_aum:,.2f}")
                logger.debug(f"[CLIENT CAGR] First trade date: {client_first_trade}")
                logger.debug(f"[CLIENT CAGR] Number of trades: {len(client_trades[client_code])}")
                
                # CAGR: Use total allocated capital as initial value (not per-period)
                # This represents the actual capital deployed over time
                client_cagr = calculate_cagr(
                    initial_value=client_total_allocated,
                    current_value=client_aum,
                    start_date=client_first_trade,
                    end_date=date.today()
                )
                logger.info(f"[CLIENT CAGR] Final CAGR for {client_code}: {client_cagr}%")
                
                # XIRR: Build cash flows
                client_cf = []
                client_cf_dates = []
                
                logger.debug(f"[CLIENT XIRR] Building cash flows for {client_code}")
                for t_date, t_side, t_qty, t_price, t_brok, t_tax in client_trades[client_code]:
                    if t_side == 'BUY':
                        # Negative cash flow (money out)
                        cf = -1 * float(float(t_qty) * float(t_price) + float(t_brok) + float(t_tax))
                    else:  # SELL
                        # Positive cash flow (money in)
                        cf = float(float(t_qty) * float(t_price) - float(t_brok) - float(t_tax))
                    
                    client_cf.append(cf)
                    client_cf_dates.append(t_date)
                    logger.debug(f"[CLIENT XIRR]   {t_date} | {t_side:4s} | ₹{cf:,.2f}")
                    
                    # Add to strategy-level cash flows
                    strategy_cash_flows.append(cf)
                    strategy_cash_flow_dates.append(t_date)
                
                # Add current AUM as final positive cash flow
                client_cf.append(client_aum)
                client_cf_dates.append(date.today())
                logger.debug(f"[CLIENT XIRR]   {date.today()} | FINAL | ₹{client_aum:,.2f}")
                
                client_xirr = calculate_xirr(client_cf, client_cf_dates)
                logger.info(f"[CLIENT XIRR] Final XIRR for {client_code}: {client_xirr}%")
            else:
                logger.warning(f"[CLIENT CAGR] No trades found for client: {client_code}")
            
            # Add to aggregators
            total_allocated += client_total_allocated
            total_cash += client_cash
            total_mv += client_mv
            total_cost_basis += client_cb
            
            clients_list.append(ClientDetail(
                client_code=client_code,
                capital=period_capital, 
                total_allocated_capital=round(client_total_allocated, 2),
                market_value=round(client_mv, 2),
                invested_amount=round(client_cb, 2),
                cash_balance=round(client_cash, 2),
                aum=round(client_aum, 2),
                pnl=round(client_pnl, 2),
                return_pct=round(client_ret, 2),
                cagr=client_cagr,
                xirr=client_xirr
            ))
            
        total_aum = total_cash + total_mv
        pnl = total_mv - total_cost_basis
        ret_pct = (pnl / total_cost_basis * 100) if total_cost_basis > 0 else 0.0
        
        # Calculate Strategy-level CAGR and XIRR
        strategy_cagr = 0.0
        strategy_xirr = 0.0
        
        if first_trade_date:
            # CAGR for strategy
            # Use sum of first period allocations as initial value
            initial_strategy_value = sum(allocations.values())
            strategy_cagr = calculate_cagr(
                initial_value=initial_strategy_value,
                current_value=total_aum,
                start_date=first_trade_date,
                end_date=date.today()
            )
            
            # XIRR for strategy
            if len(strategy_cash_flows) > 0:
                # Add final AUM as positive cash flow
                strategy_cash_flows.append(total_aum)
                strategy_cash_flow_dates.append(date.today())
                
                strategy_xirr = calculate_xirr(strategy_cash_flows, strategy_cash_flow_dates)
        
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
            total_invested=round(total_cost_basis, 2),
            market_value=round(total_mv, 2),
            pnl=round(pnl, 2),
            return_pct=round(ret_pct, 2),
            total_allocated_capital=round(total_allocated, 2),
            cash_balance=round(total_cash, 2),
            aum=round(total_aum, 2),
            cagr=strategy_cagr,
            xirr=strategy_xirr,
            holdings_count=holdings_count,
            running_status=strategy.get('status', 'deploy'),
            owner_email=owner_email,
            owner_name=owner_name,
            clients=clients_list
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching strategy summary: {e}")
        import traceback
        logger.error(traceback.format_exc())
        db.rollback()
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
