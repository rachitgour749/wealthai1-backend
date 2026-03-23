"""
Centralized Strategy APIs
Unified endpoints for all strategy operations controlled by strategy_type parameter
"""
from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Dict, Any, List, Optional
import logging
import sys
import os

# Add project root to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from APIs.unified_schemas import DateRangeRequest, SaveStrategyRequest

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router - use same tag as centralized backtest to group together
strategy_router = APIRouter(prefix="/api", tags=["Centralized Backtest API"])


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_strategy_module(strategy_type: str):
    """Get the appropriate strategy module based on strategy_type"""
    try:
        if strategy_type == "ETF_Rotation":
            from Strategies.Rotation_ETF.api import etf_routes
            return etf_routes
        elif strategy_type == "International_ETF_Rotation":
            from Strategies.Rotation_International_ETF.api import routes
            return routes
        elif strategy_type == "Rotation_Stocks":
            from Strategies.Rotation_Stocks.api import stock_routes
            return stock_routes
        elif strategy_type == "RS_ETF_Rotation":
            from Strategies.RS_ETF import api as rs_etf_api
            return rs_etf_api
        elif strategy_type == "RS_Stocks":
            from Strategies.RS_Stocks import api as rs_stocks_api
            return rs_stocks_api
        elif strategy_type == "ETF_Payout":
            from Strategies.CustomStrategies.Rotation_ETF_Payout.api import payout_routes
            return payout_routes
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
    except Exception as e:
        logger.error(f"Failed to load strategy module for {strategy_type}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load strategy: {str(e)}")


# ============================================================================
# ENDPOINTS
# ============================================================================

@strategy_router.get("/assets")
async def get_assets(
    strategy_type: Optional[str] = Query(None, description="Strategy type (legacy, optional)"),
    market: Optional[str] = Query(None, description="Market: INDIA or US"),
    asset_type: Optional[str] = Query(None, description="Asset type: ETF or STOCK")
):
    """
    Get available assets (ETFs/Stocks).

    **New Usage (preferred):**
    - `?market=INDIA&asset_type=ETF`     → Indian ETF list
    - `?market=INDIA&asset_type=STOCK`   → Indian Stock list
    - `?market=US&asset_type=ETF`        → US ETF list
    - `?market=US&asset_type=STOCK`      → US Stock list

    **Legacy Usage (still supported):**
    - `?strategy_type=ETF_Rotation`, `Rotation_Stocks`, etc.
    """
    try:
        # ── NEW: market + asset_type path ────────────────────────────────────
        if market and asset_type:
            market = market.upper()
            asset_type = asset_type.upper()
            logger.info(f"Getting assets for market={market}, asset_type={asset_type}")

            from Services.market_data_service import MarketDataService
            from Databases.app_data_db_connection import get_session

            db = get_session()
            try:
                model = MarketDataService.get_model(market, asset_type)
                rows = db.query(model.symbol).distinct().order_by(model.symbol).all()
                symbols = [r[0] for r in rows]
                
                assets = [{"ticker": s, "symbol": s} for s in symbols]
                asset_key = "stocks" if asset_type == "STOCK" else "etfs"
                return {
                    "success": True,
                    "market": market,
                    "asset_type": asset_type,
                    asset_key: assets,
                    "total": len(assets)
                }
            finally:
                db.close()

        # ── LEGACY: strategy_type path ───────────────────────────────────────
        if not strategy_type:
            raise HTTPException(status_code=400, detail="Provide either 'market'+'asset_type' or 'strategy_type'")

        logger.info(f"Getting assets for strategy: {strategy_type}")

        if strategy_type == "ETF_Rotation":
            from Strategies.Rotation_ETF.api.etf_routes import etf_backtester
            if etf_backtester is None:
                raise HTTPException(status_code=500, detail="ETF backtester not initialized")
            metadata = etf_backtester.load_metadata()
            etfs = [{"ticker": t, "name": d.get('name', t), "category": d.get('category', ''), "expense_ratio": d.get('expense_ratio', 0.0), "aum": d.get('aum', 0.0)} for t, d in metadata.items()]
            return {"etfs": etfs}

        elif strategy_type == "International_ETF_Rotation":
            from Strategies.Rotation_International_ETF.api.routes import international_etf_backtester
            if international_etf_backtester is None:
                raise HTTPException(status_code=500, detail="International ETF backtester not initialized")
            metadata = international_etf_backtester.load_metadata()
            etfs = [{"ticker": t, "name": d.get('name', t), "category": d.get('category', ''), "expense_ratio": d.get('expense_ratio', 0.0), "aum": d.get('aum', 0.0)} for t, d in metadata.items()]
            return {"etfs": etfs}

        elif strategy_type == "Rotation_Stocks":
            from Strategies.Rotation_Stocks.api.stock_routes import stock_backtester
            if stock_backtester is None:
                raise HTTPException(status_code=500, detail="Stock backtester not initialized")
            metadata = stock_backtester.load_metadata()
            stocks = [{"ticker": t, "name": d.get('name', t), "sector": d.get('sector', ''), "market_cap": d.get('market_cap', 0.0)} for t, d in metadata.items()]
            return {"stocks": stocks}

        elif strategy_type == "RS_ETF_Rotation":
            from Strategies.RS_ETF.rs_etf_backtester_core import RSETFStrategyBacktester
            from Strategies.RS_ETF.database import get_db
            db = next(get_db())
            try:
                backtester = RSETFStrategyBacktester.from_config_dict(db, {'main_index': '^NSEI', 'etf_universe': 'ALL_ETFS', 'buffer_capital_pct': 10.0, 'max_positions': 20})
                etf_list = backtester.get_custom_etf_universe()
                return {"success": True, "strategy_type": strategy_type, "etfs": [{"ticker": t, "symbol": t} for t in etf_list]}
            finally:
                db.close()

        elif strategy_type == "RS_Stocks":
            from Strategies.RS_Stocks.rs_backtester_core import RSStrategyBacktester
            from Strategies.RS_Stocks.database import get_db
            db = next(get_db())
            try:
                backtester = RSStrategyBacktester.from_config_dict(db, {'main_index': '^NSEI', 'stock_universe': 'NIFTY_500', 'buffer_capital_pct': 10.0, 'max_positions': 20})
                stock_list = backtester.get_custom_stock_universe()
                return {"success": True, "strategy_type": strategy_type, "stocks": [{"ticker": t, "symbol": t} for t in stock_list]}
            finally:
                db.close()

        elif strategy_type == "ETF_Payout":
            from Strategies.CustomStrategies.Rotation_ETF_Payout.api_routes import rotation_etf_payout_backtester
            if rotation_etf_payout_backtester is None:
                raise HTTPException(status_code=500, detail="ETF Payout backtester not initialized")
            metadata = rotation_etf_payout_backtester.load_metadata()
            etfs = [{"ticker": t, "name": d.get('name', t), "category": d.get('category', ''), "expense_ratio": d.get('expense_ratio', 0.0), "aum": d.get('aum', 0.0)} for t, d in metadata.items()]
            return {"etfs": etfs}

        elif strategy_type == "ETF_Swing_Strategy":
            from Strategies.Rotation_ETF.api.etf_routes import etf_backtester, initialize_etf_backtester
            if etf_backtester is None:
                initialize_etf_backtester()
                from Strategies.Rotation_ETF.api.etf_routes import etf_backtester
            if etf_backtester is None:
                raise HTTPException(status_code=500, detail="ETF backtester not initialized")
            metadata = etf_backtester.load_metadata()
            etfs = [{"ticker": t, "name": d.get('name', t), "category": d.get('category', ''), "expense_ratio": d.get('expense_ratio', 0.0), "aum": d.get('aum', 0.0)} for t, d in metadata.items()]
            return {"etfs": etfs}

        elif strategy_type == "SuperTrend":
            from Strategies.SuperTrend.services.backtester import SuperTrendBacktester
            # SuperTrend can be STOCK or ETF, defaults to STOCK if not specified
            backtester = SuperTrendBacktester(market=market or "INDIA", asset_type=asset_type or "STOCK")
            metadata = backtester.load_metadata()
            assets = []
            for ticker, data in metadata.items():
                asset_item = {
                    "ticker": ticker,
                    "name": data.get('name', ticker),
                    "category": data.get('category', 'Unknown'),
                    "start_date": data.get('start_date'),
                    "end_date": data.get('end_date'),
                    "years_available": data.get('years_available', 0),
                    "total_records": data.get('total_records', 0)
                }
                assets.append(asset_item)
            
            # Return as stocks or etfs based on actual asset_type
            key = "etfs" if backtester.asset_type == "ETF" else "stocks"
            return {key: assets}

        else:
            raise HTTPException(status_code=400, detail=f"Invalid strategy_type: {strategy_type}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting assets: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to get assets: {str(e)}")


@strategy_router.post("/date-range")
async def calculate_date_range(request: DateRangeRequest):
    """
    Calculate available date range for backtesting
    
    Returns the date range based on data availability for the selected assets.
    Response format varies by strategy type.
    """
    try:
        logger.info(f"Calculating date range for strategy: {request.strategy_type}")
        
        if request.strategy_type == "ETF_Rotation":
            # Use ETF date range logic with global backtester from API routes
            from Strategies.Rotation_ETF.api.etf_routes import etf_backtester
            
            if etf_backtester is None:
                raise HTTPException(status_code=500, detail="ETF backtester not initialized")
            
            tickers = request.tickers or []
            if not tickers:
                raise HTTPException(status_code=400, detail="No tickers provided")
            
            start_date, end_date, years = etf_backtester.calculate_common_date_range(tickers)
            
            if start_date and end_date:
                return {
                    "start_date": start_date,
                    "end_date": end_date,
                    "years": years
                }
            else:
                raise HTTPException(status_code=400, detail="Could not calculate date range for provided tickers")
        
        elif request.strategy_type == "International_ETF_Rotation":
            # Use International ETF date range logic
            from Strategies.Rotation_International_ETF.api.routes import international_etf_backtester
            
            if international_etf_backtester is None:
                raise HTTPException(status_code=500, detail="International ETF backtester not initialized")
            
            tickers = request.tickers or []
            if not tickers:
                raise HTTPException(status_code=400, detail="No tickers provided")
            
            start_date, end_date, years = international_etf_backtester.calculate_common_date_range(tickers)
            
            if start_date and end_date:
                return {
                    "start_date": start_date,
                    "end_date": end_date,
                    "years": years
                }
            else:
                raise HTTPException(status_code=400, detail="Could not calculate date range for provided tickers")
            
        elif request.strategy_type == "Rotation_Stocks":
            # Use stock date range logic with global backtester from API routes
            from Strategies.Rotation_Stocks.api.stock_routes import stock_backtester
            
            if stock_backtester is None:
                raise HTTPException(status_code=500, detail="Stock backtester not initialized")
            
            tickers = request.tickers or []
            if not tickers:
                raise HTTPException(status_code=400, detail="No tickers provided")
            
            start_date, end_date, years = stock_backtester.calculate_common_date_range(tickers)
            
            if start_date and end_date:
                return {
                    "start_date": start_date,
                    "end_date": end_date,
                    "years": years
                }
            else:
                raise HTTPException(status_code=400, detail="Could not calculate date range for provided tickers")
            
        elif request.strategy_type == "RS_ETF_Rotation":
            # RS ETF creates a temporary backtester instance
            from Strategies.RS_ETF.rs_etf_backtester_core import RSETFStrategyBacktester
            from Strategies.RS_ETF.database import get_db
            
            # Get database session
            db = next(get_db())
            
            try:
                # Create temporary backtester with minimal config
                temp_config = {
                    'main_index': request.main_index or '^NSEI',
                    'etf_universe': request.etf_universe or 'ALL_ETFS',
                    'max_positions': 20,
                    'position_size_pct': 5.0,
                    'total_capital': 1000000.0,
                    'stop_loss_pct': 15.0,
                    'buffer_capital_pct': 10.0,
                    'capital_reset_threshold_pct': 25.0,
                    'max_holding_period': 52,
                    'transaction_cost_pct': 0.1,
                    'min_price': 10.0,
                    'min_turnover': 1000000.0,
                    'lookback_weeks': request.lookback_weeks or 5,
                    'lookback_months': request.lookback_months or 20,
                    'lookback_quarters': request.lookback_quarters or 60
                }
                
                backtester = RSETFStrategyBacktester.from_config_dict(db, temp_config)
                # Use tickers from request (not custom_etfs)
                tickers = request.tickers or request.custom_etfs or []
                
                if not tickers:
                    raise HTTPException(status_code=400, detail="No tickers provided for RS_ETF_Rotation")
                
                start_date, end_date, years = backtester.calculate_common_date_range(tickers)
                
                if start_date and end_date:
                    return {
                        "start_date": start_date,
                        "end_date": end_date,
                        "years": years
                    }
                else:
                    raise HTTPException(status_code=400, detail="Could not calculate date range for provided tickers")
            finally:
                db.close()
            
        elif request.strategy_type == "RS_Stocks":
            # RS Stocks creates a temporary backtester instance
            from Strategies.RS_Stocks.rs_stocks_backtester_core import RSStocksStrategyBacktester
            from Strategies.RS_Stocks.database import get_db as get_stocks_db
            
            # Get database session
            db = next(get_stocks_db())
            
            try:
                # Create temporary backtester with minimal config
                temp_config = {
                    'main_index': request.main_index or '^NSEI',
                    'stock_universe': request.stock_universe or 'NIFTY_500',
                    'max_positions': 20,
                    'position_size_pct': 5.0,
                    'total_capital': 1000000.0,
                    'stop_loss_pct': 15.0,
                    'buffer_capital_pct': 10.0,
                    'capital_reset_threshold_pct': 25.0,
                    'max_holding_period': 52,
                    'transaction_cost_pct': 0.1,
                    'min_price': 10.0,
                    'min_turnover': 1000000.0,
                    'lookback_weeks': request.lookback_weeks or 5,
                    'lookback_months': request.lookback_months or 20,
                    'lookback_quarters': request.lookback_quarters or 60
                }
                
                backtester = RSStocksStrategyBacktester.from_config_dict(db, temp_config)
                # Use tickers from request (not custom_stocks)
                tickers = request.tickers or request.custom_stocks or []
                
                if not tickers:
                    raise HTTPException(status_code=400, detail="No tickers provided for RS_Stocks")
                
                start_date, end_date, years = backtester.calculate_common_date_range(tickers)
                
                if start_date and end_date:
                    return {
                        "start_date": start_date,
                        "end_date": end_date,
                        "years": years
                    }
                else:
                    raise HTTPException(status_code=400, detail="Could not calculate date range for provided tickers")
            finally:
                db.close()
        
        elif request.strategy_type == "ETF_Payout":
            # ETF_Payout uses same date range logic as ETF_Rotation
            from Strategies.CustomStrategies.Rotation_ETF_Payout.api_routes import rotation_etf_payout_backtester
            
            if rotation_etf_payout_backtester is None:
                raise HTTPException(status_code=500, detail="ETF Payout backtester not initialized")
            
            tickers = request.tickers or []
            if not tickers:
                raise HTTPException(status_code=400, detail="No tickers provided")
            
            start_date, end_date, years = rotation_etf_payout_backtester.calculate_common_date_range(tickers)
            
            if start_date and end_date:
                return {
                    "start_date": start_date,
                    "end_date": end_date,
                    "years": years
                }
            else:
                raise HTTPException(status_code=400, detail="Could not calculate date range for provided tickers")

        elif request.strategy_type == "ETF_Swing_Strategy":
            # ETF_Swing_Strategy uses same date range logic as ETF_Rotation
            from Strategies.Rotation_ETF.api.etf_routes import etf_backtester, initialize_etf_backtester
            
            if etf_backtester is None:
                initialize_etf_backtester()
                from Strategies.Rotation_ETF.api.etf_routes import etf_backtester
            
            if etf_backtester is None:
                raise HTTPException(status_code=500, detail="ETF backtester not initialized")
            
            tickers = request.tickers or []
            if not tickers:
                raise HTTPException(status_code=400, detail="No tickers provided")
            
            start_date, end_date, years = etf_backtester.calculate_common_date_range(tickers)
            
            if start_date and end_date:
                return {
                    "start_date": start_date,
                    "end_date": end_date,
                    "years": years
                }
            else:
                raise HTTPException(status_code=400, detail="Could not calculate date range for provided tickers")
                
        elif request.strategy_type == "SuperTrend":
            from Strategies.SuperTrend.services.backtester import SuperTrendBacktester
            backtester = SuperTrendBacktester(
                market=request.market or "INDIA",
                asset_type=request.asset_type or "STOCK"
            )
            tickers = request.tickers or []
            if not tickers:
                raise HTTPException(status_code=400, detail="No tickers provided")
            
            start_date, end_date, years = backtester.calculate_common_date_range(tickers)
            if start_date and end_date:
                return {
                    "start_date": start_date,
                    "end_date": end_date,
                    "years": years
                }
            else:
                raise HTTPException(status_code=400, detail="Could not calculate date range for provided symbols")

        else:
            raise HTTPException(status_code=400, detail=f"Invalid strategy_type: {request.strategy_type}")
            
    except Exception as e:
        logger.error(f"Error calculating date range for {request.strategy_type}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to calculate date range: {str(e)}")


@strategy_router.get("/assets/overview")
async def get_assets_overview(
    strategy_type: Optional[str] = Query(None, description="Strategy type (legacy, optional)"),
    market: Optional[str] = Query(None, description="Market: INDIA or US"),
    asset_type: Optional[str] = Query(None, description="Asset type: ETF or STOCK")
):
    """
    Get metadata/overview for all available assets.

    **New Usage (preferred):**
    - `?market=INDIA&asset_type=ETF`   → Indian ETF overview
    - `?market=INDIA&asset_type=STOCK` → Indian Stock overview
    - `?market=US&asset_type=ETF`      → US ETF overview
    - `?market=US&asset_type=STOCK`    → US Stock overview

    **Legacy Usage (still supported):**
    - `?strategy_type=ETF_Rotation`, `Rotation_Stocks`, etc.
    """
    try:
        # ── NEW: market + asset_type path ────────────────────────────────────
        if market and asset_type:
            market = market.upper()
            asset_type = asset_type.upper()
            logger.info(f"Getting assets overview for market={market}, asset_type={asset_type}")

            from Services.market_data_service import MarketDataService
            from Databases.app_data_db_connection import get_session
            from sqlalchemy import func

            db = get_session()
            try:
                model = MarketDataService.get_model(market, asset_type)

                # Aggregate per symbol: min date, max date, count
                rows = db.query(
                    model.symbol,
                    func.min(model.date).label('start_date'),
                    func.max(model.date).label('end_date'),
                    func.count(model.date).label('total_records')
                ).group_by(model.symbol).order_by(model.symbol).all()

                overview = []
                for row in rows:
                    start_dt = row.start_date
                    end_dt = row.end_date
                    try:
                        years = round((end_dt - start_dt).days / 365.25, 1)
                    except Exception:
                        years = 0.0
                    overview.append({
                        'symbol': row.symbol,
                        'description': MarketDataService.generate_asset_description(row.symbol, asset_type),
                        'sector': MarketDataService.get_asset_sector_classification(row.symbol, asset_type),
                        'start_date': str(start_dt),
                        'end_date': str(end_dt),
                        'years_available': years,
                        'total_records': row.total_records
                    })

                overview_key = "stock_overview" if asset_type == "STOCK" else "etf_overview"
                return {
                    "success": True,
                    "market": market,
                    "asset_type": asset_type,
                    overview_key: overview,
                    "total": len(overview)
                }
            finally:
                db.close()

        # ── LEGACY: strategy_type path ───────────────────────────────────────
        if not strategy_type:
            raise HTTPException(status_code=400, detail="Provide either 'market'+'asset_type' or 'strategy_type'")

        logger.info(f"Getting assets overview for strategy: {strategy_type}")

        if strategy_type == "ETF_Rotation":
            from Strategies.Rotation_ETF.api.etf_routes import etf_backtester
            if etf_backtester is None:
                raise HTTPException(status_code=500, detail="ETF backtester not initialized")
            metadata = etf_backtester.load_metadata()
            etf_overview = []
            for symbol, meta in metadata.items():
                description = etf_backtester.generate_asset_description(symbol)
                sector = etf_backtester.get_asset_sector_classification(symbol)
                etf_overview.append({'symbol': symbol, 'description': description, 'sector': sector, 'start_date': meta['start_date'], 'end_date': meta['end_date'], 'years_available': round(meta['years_available'], 1), 'total_records': meta['total_records']})
            etf_overview.sort(key=lambda x: x['start_date'])
            return {"etf_overview": etf_overview}

        elif strategy_type == "International_ETF_Rotation":
            from Strategies.Rotation_International_ETF.api.routes import international_etf_backtester
            if international_etf_backtester is None:
                raise HTTPException(status_code=500, detail="International ETF backtester not initialized")
            metadata = international_etf_backtester.load_metadata()
            etf_overview = []
            for symbol, meta in metadata.items():
                description = international_etf_backtester.generate_asset_description(symbol)
                sector = international_etf_backtester.get_asset_sector_classification(symbol)
                etf_overview.append({'symbol': symbol, 'description': description, 'sector': sector, 'start_date': meta['start_date'], 'end_date': meta['end_date'], 'years_available': round(meta['years_available'], 1), 'total_records': meta['total_records']})
            etf_overview.sort(key=lambda x: x['start_date'])
            return {"etf_overview": etf_overview}

        elif strategy_type == "Rotation_Stocks":
            from Strategies.Rotation_Stocks.api.stock_routes import stock_backtester
            if stock_backtester is None:
                raise HTTPException(status_code=500, detail="Stock backtester not initialized")
            metadata = stock_backtester.load_metadata()
            stock_overview = []
            for symbol, meta in metadata.items():
                description = stock_backtester.generate_asset_description(symbol)
                sector = stock_backtester.get_asset_sector_classification(symbol)
                stock_overview.append({'symbol': symbol, 'description': description, 'sector': sector, 'start_date': meta['start_date'], 'end_date': meta['end_date'], 'years_available': round(meta['years_available'], 1), 'total_records': meta['total_records']})
            stock_overview.sort(key=lambda x: x['start_date'])
            return {"stock_overview": stock_overview}

        elif strategy_type == "ETF_Swing_Strategy":
            from Strategies.Rotation_ETF.api.etf_routes import etf_backtester, initialize_etf_backtester
            if etf_backtester is None:
                initialize_etf_backtester()
                from Strategies.Rotation_ETF.api.etf_routes import etf_backtester
            if etf_backtester is None:
                raise HTTPException(status_code=500, detail="ETF backtester not initialized")
            metadata = etf_backtester.load_metadata()
            etf_overview = []
            for symbol, meta in metadata.items():
                description = etf_backtester.generate_asset_description(symbol)
                sector = etf_backtester.get_asset_sector_classification(symbol)
                etf_overview.append({'symbol': symbol, 'description': description, 'sector': sector, 'start_date': meta['start_date'], 'end_date': meta['end_date'], 'years_available': round(meta['years_available'], 1), 'total_records': meta['total_records']})
            etf_overview.sort(key=lambda x: x['start_date'])
            return {"etf_overview": etf_overview}

        elif strategy_type == "ETF_Payout":
            from Strategies.CustomStrategies.Rotation_ETF_Payout.api_routes import rotation_etf_payout_backtester
            if rotation_etf_payout_backtester is None:
                raise HTTPException(status_code=500, detail="ETF Payout backtester not initialized")
            metadata = rotation_etf_payout_backtester.load_metadata()
            etf_overview = []
            for symbol, meta in metadata.items():
                description = rotation_etf_payout_backtester.generate_asset_description(symbol)
                sector = rotation_etf_payout_backtester.get_asset_sector_classification(symbol)
                etf_overview.append({'symbol': symbol, 'description': description, 'sector': sector, 'start_date': meta['start_date'], 'end_date': meta['end_date'], 'years_available': round(meta['years_available'], 1), 'total_records': meta['total_records']})
            etf_overview.sort(key=lambda x: x['start_date'])
            return {"etf_overview": etf_overview}

        elif strategy_type == "RS_ETF_Rotation":
            from Strategies.RS_ETF.rs_etf_backtester_core import RSETFStrategyBacktester
            from Strategies.RS_ETF.database import get_db as get_rs_etf_db
            db = next(get_rs_etf_db())
            try:
                backtester = RSETFStrategyBacktester.from_config_dict(db, {'main_index': '^NSEI', 'buffer_capital_pct': 10.0, 'max_positions': 10})
                metadata = backtester.load_metadata()
                etf_overview = []
                for symbol, meta in metadata.items():
                    description = backtester.generate_asset_description(symbol)
                    sector = backtester.get_asset_sector_classification(symbol)
                    etf_overview.append({'symbol': symbol, 'description': description, 'sector': sector, 'start_date': meta['start_date'], 'end_date': meta['end_date'], 'years_available': round(meta['years_available'], 1), 'total_records': meta['total_records']})
                etf_overview.sort(key=lambda x: x['symbol'])
                return {"etf_overview": etf_overview}
            finally:
                db.close()

        elif strategy_type == "RS_Stocks":
            from Strategies.RS_Stocks.rs_backtester_core import RSStrategyBacktester
            from Strategies.RS_Stocks.database import get_db as get_rs_stocks_db
            db = next(get_rs_stocks_db())
            try:
                backtester = RSStrategyBacktester.from_config_dict(db, {'main_index': '^NSEI', 'stock_universe': 'NIFTY_500', 'buffer_capital_pct': 10.0, 'max_positions': 20})
                metadata = backtester.load_metadata()
                stock_overview = []
                for symbol, meta in metadata.items():
                    description = backtester.generate_asset_description(symbol)
                    sector = backtester.get_asset_sector_classification(symbol)
                    stock_overview.append({'symbol': symbol, 'description': description, 'sector': sector, 'start_date': meta['start_date'], 'end_date': meta['end_date'], 'years_available': round(meta['years_available'], 1), 'total_records': meta['total_records']})
                stock_overview.sort(key=lambda x: x['symbol'])
                return {"stock_overview": stock_overview}
            finally:
                db.close()

        elif strategy_type == "SuperTrend":
            from Strategies.SuperTrend.services.backtester import SuperTrendBacktester
            backtester = SuperTrendBacktester(market=market or "INDIA", asset_type=asset_type or "STOCK")
            metadata = backtester.load_metadata()
            overview = []
            for symbol, meta in metadata.items():
                overview.append({
                    'symbol': symbol,
                    'description': backtester.generate_asset_description(symbol),
                    'sector': backtester.get_asset_sector_classification(symbol),
                    'start_date': meta.get('start_date'),
                    'end_date': meta.get('end_date'),
                    'years_available': round(meta.get('years_available', 0), 1),
                    'total_records': meta.get('total_records', 0)
                })
            overview.sort(key=lambda x: x['start_date'] if x['start_date'] else '9999-99-99')
            # Use appropriate key
            key = "etf_overview" if backtester.asset_type == "ETF" else "stock_overview"
            return {key: overview}

        else:
            raise HTTPException(status_code=400, detail=f"Invalid strategy_type: {strategy_type}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting assets overview: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get assets overview: {str(e)}")





@strategy_router.get("/debug/cache")
async def debug_cache():
    """Debug endpoint to inspect cache contents"""
    try:
        from APIs.centralized_backtest import _backtest_results_cache
        
        cache_info = {}
        for strategy_type, data in _backtest_results_cache.items():
            cache_info[strategy_type] = {
                "portfolio_log_count": len(data.get('portfolio_log', [])),
                "has_weekly_nav_df": data.get('weekly_nav_df') is not None,
                "trading_summary_keys": list(data.get('trading_summary', {}).keys())
            }
        
        return {
            "cache_keys": list(_backtest_results_cache.keys()),
            "cache_details": cache_info,
            "total_strategies_cached": len(_backtest_results_cache)
        }
    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}


@strategy_router.get("/cached-transaction-log")
async def get_cached_transaction_log(strategy_type: str = Query(..., description="Strategy type")):
    """
    Get transaction log directly from cache (after running backtest)
    
    This endpoint ONLY returns cached data from the most recent backtest.
    Run /api/run_backtest first to populate the cache.
    """
    try:
        from APIs.centralized_backtest import _backtest_results_cache
        
        print(f"[CACHED-TX-LOG] Requested for: {strategy_type}")
        print(f"[CACHED-TX-LOG] Available cache keys: {list(_backtest_results_cache.keys())}")
        
        if strategy_type not in _backtest_results_cache:
            return {
                "success": False,
                "message": f"No cached data for {strategy_type}. Run backtest first.",
                "available_strategies": list(_backtest_results_cache.keys()),
                "transaction_log": [],
                "trading_summary": {}
            }
        
        cached_data = _backtest_results_cache[strategy_type]
        portfolio_log = cached_data.get('portfolio_log', [])
        trading_summary = cached_data.get('trading_summary', {})
        
        # Sanitize data to handle numpy types, NaN/Inf, etc.
        # We'll use a local sanitization function to avoid dependency on a handler instance
        def sanitize(obj):
            import math
            import numpy as np
            from datetime import date, datetime
            
            if isinstance(obj, dict):
                return {str(k): sanitize(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [sanitize(item) for item in obj]
            elif isinstance(obj, (int, float)):
                if math.isnan(obj) or math.isinf(obj):
                    return 0
                return obj
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                if np.isnan(obj) or np.isinf(obj):
                    return 0
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return sanitize(obj.tolist())
            elif isinstance(obj, (datetime, date)):
                return obj.isoformat()
            else:
                return obj

        sanitized_log = sanitize(portfolio_log)
        sanitized_summary = sanitize(trading_summary)
        
        print(f"[CACHED-TX-LOG] Returning {len(sanitized_log)} transactions")
        
        return {
            "success": True,
            "strategy_type": strategy_type,
            "transaction_log": sanitized_log,
            "trading_summary": sanitized_summary,
            "total_transactions": len(sanitized_log)
        }
        
    except Exception as e:
        import traceback
        print(f"[CACHED-TX-LOG] ERROR: {e}")
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "transaction_log": [],
            "trading_summary": {}
        }


@strategy_router.get("/transaction-log")
async def get_transaction_log(strategy_type: str = Query(..., description="Strategy type")):
    """
    Get transaction log from the last backtest
    
    **Note**: Response format varies by strategy type.
    - Rotation strategies: Returns {transaction_log: [], trading_summary: {}}
    - RS strategies: Returns array of trade objects
    """
    try:
        logger.info(f"Getting transaction log for strategy: {strategy_type}")
        print(f"[TRANSACTION-LOG] Called for strategy_type: {strategy_type}")
        
        # FIRST: Check cache for recent backtest results
        print(f"[TRANSACTION-LOG] About to check cache...")
        try:
            from APIs.centralized_backtest import get_cached_backtest_results
            print(f"[TRANSACTION-LOG] Successfully imported get_cached_backtest_results")
            
            cached_results = get_cached_backtest_results(strategy_type)
            print(f"[TRANSACTION-LOG] Cache check returned: {type(cached_results)}, has data: {bool(cached_results)}")
            
            # Check if cache has data - be explicit about checking portfolio_log
            portfolio_log = cached_results.get('portfolio_log', []) if cached_results else []
            print(f"[TRANSACTION-LOG] portfolio_log type: {type(portfolio_log)}, length: {len(portfolio_log) if isinstance(portfolio_log, list) else 'NOT A LIST'}")
            
            if portfolio_log and len(portfolio_log) > 0:
                portfolio_log_count = len(portfolio_log)
                print(f"[TRANSACTION-LOG] ✅ Using cached results: {portfolio_log_count} transactions")
                logger.info(f"✅ Found cached results for {strategy_type}: {portfolio_log_count} transactions")
                return {
                    "transaction_log": portfolio_log,
                    "trading_summary": cached_results.get('trading_summary', {})
                }
            else:
                print(f"[TRANSACTION-LOG] No cached results, falling back to global instances")
                logger.info(f"No cached results found for {strategy_type}, checking global instances...")
        except Exception as e:
            print(f"[TRANSACTION-LOG] ❌ Error checking cache: {e}")
            import traceback
            traceback.print_exc()
            logger.warning(f"Error checking cache: {e}, falling back to global instances")
        
        # FALLBACK: Check global backtester instances
        print(f"[TRANSACTION-LOG] Checking global backtester instances for {strategy_type}")
        if strategy_type == "ETF_Rotation":
            # Get ETF transaction log from backtester in API routes
            from Strategies.Rotation_ETF.api.etf_routes import etf_backtester
            
            if etf_backtester is None or not hasattr(etf_backtester, 'portfolio_log'):
                return {"transaction_log": [], "trading_summary": {}}
            
            transaction_log = []
            for log in etf_backtester.portfolio_log:
                transaction_log.append(log)
            
            return {
                "transaction_log": transaction_log,
                "trading_summary": getattr(etf_backtester, 'trading_summary', {})
            }
        
        elif strategy_type == "International_ETF_Rotation":
            # Get International ETF transaction log
            from Strategies.Rotation_International_ETF.api.routes import international_etf_backtester
            
            if international_etf_backtester is None or not hasattr(international_etf_backtester, 'portfolio_log'):
                return {"transaction_log": [], "trading_summary": {}}
            
            transaction_log = []
            for log in international_etf_backtester.portfolio_log:
                transaction_log.append(log)
            
            return {
                "transaction_log": transaction_log,
                "trading_summary": getattr(international_etf_backtester, 'trading_summary', {})
            }
            
        elif strategy_type == "Rotation_Stocks":
            # Get stock transaction log from API routes
            from Strategies.Rotation_Stocks.api.stock_routes import stock_backtester
            
            if stock_backtester is None or not hasattr(stock_backtester, 'portfolio_log'):
                return {"transaction_log": [], "trading_summary": {}}
            
            transaction_log = []
            for log in stock_backtester.portfolio_log:
                transaction_log.append(log)
            
            return {
                "transaction_log": transaction_log,
                "trading_summary": getattr(stock_backtester, 'trading_summary', {})
            }
        
        elif strategy_type == "ETF_Payout":
            # Get ETF_Payout transaction log
            from Strategies.CustomStrategies.Rotation_ETF_Payout.api_routes import rotation_etf_payout_backtester
            
            if rotation_etf_payout_backtester is None or not hasattr(rotation_etf_payout_backtester, 'portfolio_log'):
                return {"transaction_log": [], "trading_summary": {}}
            
            transaction_log = []
            for log in rotation_etf_payout_backtester.portfolio_log:
                transaction_log.append(log)
            
            return {
                "transaction_log": transaction_log,
                "trading_summary": getattr(rotation_etf_payout_backtester, 'trading_summary', {})
            }
            
        elif strategy_type in ["RS_ETF_Rotation", "RS_Stocks"]:
            # RS strategies store trades in database
            # Need backtest_id to retrieve trades
            return {
                "success": False,
                "message": "RS strategies require backtest_id. Use /api/run_backtest first, then /api/backtests/{id}/trades"
            }
            
        else:
            raise HTTPException(status_code=400, detail=f"Invalid strategy_type: {strategy_type}")
            
    except Exception as e:
        logger.error(f"Error getting transaction log for {strategy_type}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get transaction log: {str(e)}")


@strategy_router.get("/cached-costs-breakdown")
async def get_cached_costs_breakdown(strategy_type: str = Query(..., description="Strategy type")):
    """
    Get cost breakdown directly from cache (after running backtest)
    
    This endpoint ONLY returns cached cost data from the most recent backtest.
    Run /api/run_backtest first to populate the cache.
    """
    try:
        from APIs.centralized_backtest import _backtest_results_cache
        
        print(f"[CACHED-COSTS] Requested for: {strategy_type}")
        print(f"[CACHED-COSTS] Available cache keys: {list(_backtest_results_cache.keys())}")
        
        if strategy_type not in _backtest_results_cache:
            return {
                "success": False,
                "message": f"No cached data for {strategy_type}. Run backtest first.",
                "available_strategies": list(_backtest_results_cache.keys())
            }
        
        cached_data = _backtest_results_cache[strategy_type]
        cost_breakdown = cached_data.get('cost_breakdown', {})
        
        print(f"[CACHED-COSTS] Cost breakdown type: {type(cost_breakdown)}")
        print(f"[CACHED-COSTS] Cost breakdown keys: {list(cost_breakdown.keys()) if isinstance(cost_breakdown, dict) else 'NOT A DICT'}")
        print(f"[CACHED-COSTS] Cost breakdown content: {cost_breakdown}")
        
        # If cost_breakdown is empty, try to get it from portfolio_log
        if not cost_breakdown:
            print(f"[CACHED-COSTS] Cost breakdown is empty, returning empty response")
            return {
                "success": False,
                "message": "Cost breakdown not available in cache. The backtester may not have cost breakdown data.",
                "strategy_type": strategy_type
            }
        
        # Return the cost breakdown data
        return cost_breakdown
        
    except Exception as e:
        import traceback
        print(f"[CACHED-COSTS] ERROR: {e}")
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }


@strategy_router.get("/costs/breakdown")
async def get_costs_breakdown(strategy_type: str = Query(..., description="Strategy type")):
    """
    Get detailed cost breakdown from the last backtest
    
    Returns breakdown by transaction type, monthly costs, etc.
    """
    try:
        logger.info(f"Getting costs breakdown for strategy: {strategy_type}")
        
        if strategy_type == "ETF_Rotation":
            from Strategies.Rotation_ETF.api.etf_routes import etf_backtester
            
            if etf_backtester is None:
                return {"error": "No backtest data available"}
            
            breakdown = etf_backtester.get_cost_breakdown()
            return breakdown
        
        elif strategy_type == "International_ETF_Rotation":
            from Strategies.Rotation_International_ETF.api.routes import international_etf_backtester
            
            if international_etf_backtester is None:
                return {"error": "No backtest data available"}
            
            breakdown = international_etf_backtester.get_cost_breakdown()
            return breakdown
        
        elif strategy_type == "ETF_Payout":
            from Strategies.CustomStrategies.Rotation_ETF_Payout.api_routes import rotation_etf_payout_backtester
            
            if rotation_etf_payout_backtester is None:
                return {"error": "No backtest data available"}
            
            breakdown = rotation_etf_payout_backtester.get_cost_breakdown()
            return breakdown
        
        elif strategy_type == "Rotation_Stocks":
            from Strategies.Rotation_Stocks.api.stock_routes import stock_backtester
            
            if stock_backtester is None:
                return {"error": "No backtest data available"}
            
            breakdown = stock_backtester.get_cost_breakdown()
            return breakdown
            
        elif strategy_type in ["RS_ETF_Rotation", "RS_Stocks"]:
            return {
                "success": False,
                "message": "RS strategies require backtest_id. Use /api/backtests/{id}/costs"
            }
            
        else:
            raise HTTPException(status_code=400, detail=f"Invalid strategy_type: {strategy_type}")
            
    except Exception as e:
        logger.error(f"Error getting costs breakdown for {strategy_type}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get costs breakdown: {str(e)}")


@strategy_router.get("/costs/summary")
async def get_costs_summary(strategy_type: str = Query(..., description="Strategy type")):
    """
    Get high-level cost summary from the last backtest
    """
    try:
        logger.info(f"Getting costs summary for strategy: {strategy_type}")
        
        if strategy_type == "ETF_Rotation":
            from Strategies.Rotation_ETF.api.etf_routes import etf_backtester
            
            if etf_backtester is None:
                return {"error": "No backtest data available"}
            
            summary = etf_backtester.get_cost_summary()
            return summary
        
        elif strategy_type == "International_ETF_Rotation":
            from Strategies.Rotation_International_ETF.api.routes import international_etf_backtester
            
            if international_etf_backtester is None:
                return {"error": "No backtest data available"}
            
            summary = international_etf_backtester.get_cost_summary()
            return summary
        
        elif strategy_type == "ETF_Payout":
            from Strategies.CustomStrategies.Rotation_ETF_Payout.api_routes import rotation_etf_payout_backtester
            
            if rotation_etf_payout_backtester is None:
                return {"error": "No backtest data available"}
            
            summary = rotation_etf_payout_backtester.get_cost_summary()
            return summary
        
        elif strategy_type == "Rotation_Stocks":
            from Strategies.Rotation_Stocks.api.stock_routes import stock_backtester
            
            if stock_backtester is None:
                return {"error": "No backtest data available"}
            
            summary = stock_backtester.get_cost_summary()
            return summary
            
        elif strategy_type in ["RS_ETF_Rotation", "RS_Stocks"]:
            return {
                "success": False,
                "message": "RS strategies require backtest_id. Use /api/backtests/{id}/costs"
            }
            
        else:
            raise HTTPException(status_code=400, detail=f"Invalid strategy_type: {strategy_type}")
            
    except Exception as e:
        logger.error(f"Error getting costs summary for {strategy_type}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get costs summary: {str(e)}")


@strategy_router.get("/costs/analysis")
async def get_costs_analysis(strategy_type: str = Query(..., description="Strategy type")):
    """
    Get cost impact analysis from the last backtest
    
    Shows how costs affect returns, cost efficiency, etc.
    """
    try:
        logger.info(f"Getting costs analysis for strategy: {strategy_type}")
        
        if strategy_type == "ETF_Rotation":
            from Strategies.Rotation_ETF.api.etf_routes import etf_backtester
            
            if etf_backtester is None:
                return {"error": "No backtest data available"}
            
            analysis = etf_backtester.get_cost_analysis()
            return analysis
        
        elif strategy_type == "International_ETF_Rotation":
            from Strategies.Rotation_International_ETF.api.routes import international_etf_backtester
            
            if international_etf_backtester is None:
                return {"error": "No backtest data available"}
            
            analysis = international_etf_backtester.get_cost_analysis()
            return analysis
        
        elif strategy_type == "ETF_Payout":
            from Strategies.CustomStrategies.Rotation_ETF_Payout.api_routes import rotation_etf_payout_backtester
            
            if rotation_etf_payout_backtester is None:
                return {"error": "No backtest data available"}
            
            analysis = rotation_etf_payout_backtester.get_cost_analysis()
            return analysis
        
        elif strategy_type == "Rotation_Stocks":
            from Strategies.Rotation_Stocks.api.stock_routes import stock_backtester
            
            if stock_backtester is None:
                return {"error": "No backtest data available"}
            
            analysis = stock_backtester.get_cost_analysis()
            return analysis
            
        elif strategy_type in ["RS_ETF_Rotation", "RS_Stocks"]:
            return {
                "success": False,
                "message": "RS strategies require backtest_id. Use /api/backtests/{id}/costs"
            }
            
        else:
            raise HTTPException(status_code=400, detail=f"Invalid strategy_type: {strategy_type}")
            
    except Exception as e:
        logger.error(f"Error getting costs analysis for {strategy_type}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get costs analysis: {str(e)}")


@strategy_router.get("/health")
async def health_check():
    """Health check endpoint for centralized strategy API"""
    return {
        "status": "healthy",
        "service": "Centralized Strategy API",
        "supported_strategies": [
            "ETF_Rotation",
            "RS_ETF_Rotation",
            "International_ETF_Rotation",
            "Rotation_Stocks",
            "ETF_Payout",
            "SuperTrend"
        ],
        "available_endpoints": [
            "GET /api/assets",
            "POST /api/date-range",
            "GET /api/assets/overview",
            "GET /api/transaction-log",
            "GET /api/costs/breakdown",
            "GET /api/costs/summary",
            "GET /api/costs/analysis"
        ]
    }
