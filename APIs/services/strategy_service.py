"""
Strategy Service
Handles asset retrieval, data validation, and configuration for strategies
"""
import logging
from fastapi import HTTPException
from typing import Dict, Any, List, Optional, Tuple

# Import schemas needed for type hints
from APIs.unified_schemas import DateRangeRequest

# Import cache functionality
from APIs.common.cache import get_cached_backtest_results

logger = logging.getLogger(__name__)

async def get_assets(strategy_type: str = None, market: str = None, asset_type: str = None) -> Dict[str, Any]:
    """Get available assets for the specified strategy type or market+asset_type"""
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
            # Get ETF list using global backtester instance from API routes
            from Strategies.Rotation_ETF.api.etf_routes import etf_backtester, initialize_etf_backtester
            
            if etf_backtester is None:
                logger.info("Lazily initializing ETF backtester")
                initialize_etf_backtester()
                from Strategies.Rotation_ETF.api.etf_routes import etf_backtester

            if etf_backtester is None:
                raise HTTPException(status_code=500, detail="ETF backtester not initialized")
            
            # Load ETF metadata
            metadata = etf_backtester.load_metadata()
            etfs = []
            
            for ticker, data in metadata.items():
                etfs.append({
                    "ticker": ticker,
                    "name": data.get('name', ticker),
                    "category": data.get('category', 'Unknown'),
                    "expense_ratio": data.get('expense_ratio', 0.0),
                    "aum": data.get('aum', 0.0),
                    "start_date": data.get('start_date'),
                    "end_date": data.get('end_date'),
                    "years_available": data.get('years_available', 0),
                    "total_records": data.get('total_records', 0)
                })
            
            return {"etfs": etfs}
            
        elif strategy_type == "International_ETF_Rotation":
            # Get International ETF list using its own backtester instance
            from Strategies.Rotation_International_ETF.api.routes import international_etf_backtester, initialize_international_etf_backtester
            
            if international_etf_backtester is None:
                logger.info("Lazily initializing International ETF backtester")
                initialize_international_etf_backtester()
                from Strategies.Rotation_International_ETF.api.routes import international_etf_backtester

            if international_etf_backtester is None:
                raise HTTPException(status_code=500, detail="International ETF backtester not initialized")
            
            # Load International ETF metadata
            metadata = international_etf_backtester.load_metadata()
            etfs = []
            
            for ticker, data in metadata.items():
                etfs.append({
                    "ticker": ticker,
                    "name": data.get('name', ticker),
                    "category": data.get('category', 'Unknown'),
                    "expense_ratio": data.get('expense_ratio', 0.0),
                    "aum": data.get('aum', 0.0)
                })
            
            return {"etfs": etfs}
            
        elif strategy_type == "Stock_Rotation":
            # Get stock list using global backtester instance from API routes
            from Strategies.Rotation_Stocks.api.stock_routes import stock_backtester, initialize_stock_backtester
            
            if stock_backtester is None:
                logger.info("Lazily initializing Stock backtester")
                initialize_stock_backtester()
                from Strategies.Rotation_Stocks.api.stock_routes import stock_backtester

            if stock_backtester is None:
                # Fallback if stock_backtester is disabled/removed
                logger.warning("Stock backtester not initialized or disabled")
                return {"stocks": [], "message": "Stock Strategy is currently disabled"}
            
            # Load stock metadata
            metadata = stock_backtester.load_metadata()
            stocks = []
            
            for ticker, data in metadata.items():
                stocks.append({
                    "ticker": ticker,
                    "name": stock_backtester.generate_asset_description(ticker),
                    "sector": stock_backtester.get_asset_sector_classification(ticker),
                    "market_cap": data.get('market_cap', 0.0)
                })
            
            return {"stocks": stocks}
            
        elif strategy_type == "RS_ETF_Rotation":
            # Get ETF universe from RS ETF
            from Strategies.RS_ETF.rs_etf_backtester_core import RSETFStrategyBacktester
            from Strategies.RS_ETF.database import get_db
            
            db = next(get_db())
            try:
                # Create temporary backtester with minimal config
                temp_config = {
                    'main_index': '^NSEI',
                    'etf_universe': 'ALL_ETFS',
                    'buffer_capital_pct': 10.0,
                    'max_positions': 20
                }
                backtester = RSETFStrategyBacktester.from_config_dict(db, temp_config)
                etf_list = backtester.get_custom_etf_universe()
                
                # Format for frontend
                etfs = [{"ticker": ticker, "symbol": ticker} for ticker in etf_list]
                return {"success": True, "strategy_type": strategy_type, "etfs": etfs}
            finally:
                db.close()
            
        elif strategy_type == "RS_Stocks":
            # Get stock universe from RS Stocks
            from Strategies.RS_Stocks.rs_backtester_core import RSStrategyBacktester
            from Strategies.RS_Stocks.database import get_db
            
            db = next(get_db())
            try:
                # Create temporary backtester with minimal config
                temp_config = {
                    'main_index': '^NSEI',
                    'stock_universe': 'NIFTY_500',
                    'buffer_capital_pct': 10.0,
                    'max_positions': 20
                }
                backtester = RSStrategyBacktester.from_config_dict(db, temp_config)
                stock_list = backtester.get_custom_stock_universe()
                
                # Format for frontend
                stocks = [{"ticker": ticker, "symbol": ticker} for ticker in stock_list]
                return {"success": True, "strategy_type": strategy_type, "stocks": stocks}
            finally:
                db.close()
        
        elif strategy_type == "ETF_Payout":
            # ETF_Payout uses same ETF list as ETF_Rotation
            from Strategies.CustomStrategies.Rotation_ETF_Payout.api_routes import rotation_etf_payout_backtester, initialize_rotation_etf_payout_backtester
            
            if rotation_etf_payout_backtester is None:
                logger.info("Lazily initializing ETF Payout backtester")
                initialize_rotation_etf_payout_backtester()
                from Strategies.CustomStrategies.Rotation_ETF_Payout.api_routes import rotation_etf_payout_backtester
            
            if rotation_etf_payout_backtester is None:
                raise HTTPException(status_code=500, detail="ETF Payout backtester not initialized")
            
            # Load ETF metadata
            metadata = rotation_etf_payout_backtester.load_metadata()
            etfs = []
            
            for ticker, data in metadata.items():
                etfs.append({
                    "ticker": ticker,
                    "name": data.get('name', ticker),
                    "category": data.get('category', 'Unknown'),
                    "expense_ratio": data.get('expense_ratio', 0.0),
                    "aum": data.get('aum', 0.0)
                })
            
            return {"etfs": etfs}
            
        elif strategy_type == "ETF_Buy_on_Dip":
            # ETF_Buy_on_Dip uses same ETF list as ETF_Rotation
            from Strategies.Rotation_ETF.api.etf_routes import etf_backtester, initialize_etf_backtester
            
            if etf_backtester is None:
                logger.info("Lazily initializing ETF backtester for Buy-on-Dip")
                initialize_etf_backtester()
                from Strategies.Rotation_ETF.api.etf_routes import etf_backtester
            
            if etf_backtester is None:
                raise HTTPException(status_code=500, detail="ETF backtester not initialized")
            
            # Load ETF metadata
            metadata = etf_backtester.load_metadata()
            etfs = []
            
            for ticker, data in metadata.items():
                etfs.append({
                    "ticker": ticker,
                    "name": data.get('name', ticker),
                    "category": data.get('category', 'Unknown'),
                    "expense_ratio": data.get('expense_ratio', 0.0),
                    "aum": data.get('aum', 0.0)
                })
            
            return {"etfs": etfs}
            
        elif strategy_type == "ETF_Swing_Strategy":
             # ETF_Swing_Strategy uses same ETF list as ETF_Rotation
            from Strategies.Rotation_ETF.api.etf_routes import etf_backtester, initialize_etf_backtester
            
            if etf_backtester is None:
                logger.info("Lazily initializing ETF backtester for ETF Swing")
                initialize_etf_backtester()
                from Strategies.Rotation_ETF.api.etf_routes import etf_backtester
            
            if etf_backtester is None:
                raise HTTPException(status_code=500, detail="ETF backtester not initialized")
            
            # Load ETF metadata
            metadata = etf_backtester.load_metadata()
            etfs = []
            
            for ticker, data in metadata.items():
                etfs.append({
                    "ticker": ticker,
                    "name": data.get('name', ticker),
                    "category": data.get('category', 'Unknown'),
                    "expense_ratio": data.get('expense_ratio', 0.0),
                    "aum": data.get('aum', 0.0)
                })
            
            return {"etfs": etfs}

        elif strategy_type == "US_ETF_Swing_Strategy":
            # US_ETF_Swing_Strategy uses the same US ETF list as International_ETF_Rotation
            from Strategies.Rotation_International_ETF.api.routes import international_etf_backtester, initialize_international_etf_backtester

            if international_etf_backtester is None:
                logger.info("Lazily initializing International ETF backtester for US ETF Swing")
                initialize_international_etf_backtester()
                from Strategies.Rotation_International_ETF.api.routes import international_etf_backtester

            if international_etf_backtester is None:
                raise HTTPException(status_code=500, detail="International ETF backtester not initialized")

            metadata = international_etf_backtester.load_metadata()
            etfs = []

            for ticker, data in metadata.items():
                etfs.append({
                    "ticker": ticker,
                    "name": data.get('name', ticker),
                    "category": data.get('category', 'Unknown'),
                    "expense_ratio": data.get('expense_ratio', 0.0),
                    "aum": data.get('aum', 0.0)
                })

            return {"etfs": etfs}

        elif strategy_type == "SuperTrend":
            from Strategies.SuperTrend.services.backtester import SuperTrendBacktester
            # Create a defaults/metadata backtester
            backtester = SuperTrendBacktester(market=market or "INDIA", asset_type=asset_type or "STOCK")
            metadata = backtester.load_metadata()
            assets = []
            for ticker, data in metadata.items():
                assets.append({
                    "ticker": ticker,
                    "name": data.get('name', ticker),
                    "category": data.get('category', 'Unknown'),
                    "start_date": data.get('start_date'),
                    "end_date": data.get('end_date'),
                    "years_available": data.get('years_available', 0),
                    "total_records": data.get('total_records', 0)
                })
            # Return as stocks or etfs based on asset_type
            key = "etfs" if backtester.asset_type == "ETF" else "stocks"
            return {key: assets}

        else:
            raise HTTPException(status_code=400, detail=f"Invalid strategy_type: {strategy_type}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting assets for {strategy_type}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to get assets: {str(e)}")


async def calculate_date_range(request: DateRangeRequest) -> Dict[str, Any]:
    """Calculate date range for the specified strategy parameters or market+asset_type"""
    try:
        # 1. NEW: Prioritize market + asset_type + tickers path
        market = request.market
        asset_type = request.asset_type
        tickers = request.tickers

        if market and asset_type and tickers:
            logger.info(f"Calculating date range for market={market}, asset_type={asset_type}, tickers={len(tickers)}")
            from Services.market_data_service import MarketDataService
            
            start_date, end_date, years = MarketDataService.calculate_date_range(
                tickers=tickers,
                market=market,
                asset_type=asset_type
            )
            
            if start_date and end_date:
                return {
                    "success": True,
                    "market": market,
                    "asset_type": asset_type,
                    "start_date": start_date,
                    "end_date": end_date,
                    "years": years
                }
            else:
                raise HTTPException(status_code=400, detail="Could not calculate date range for provided symbols")

        # 2. LEGACY: strategy_type path (Fallback)
        if not request.strategy_type:
            raise HTTPException(status_code=400, detail="Provide either 'market'+'assets_type'+'symbols' or 'strategy_type'")

        logger.info(f"Calculating date range for strategy: {request.strategy_type}")
        
        if request.strategy_type == "ETF_Rotation":
            # Use ETF date range logic with global backtester from API routes
            from Strategies.Rotation_ETF.api.etf_routes import etf_backtester, initialize_etf_backtester
            
            if etf_backtester is None:
                logger.info("Lazily initializing ETF backtester")
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
        
        elif request.strategy_type == "International_ETF_Rotation":
            # Use International ETF date range logic
            from Strategies.Rotation_International_ETF.api.routes import international_etf_backtester, initialize_international_etf_backtester
            
            if international_etf_backtester is None:
                logger.info("Lazily initializing International ETF backtester")
                initialize_international_etf_backtester()
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
            
        elif request.strategy_type == "Stock_Rotation":
            # Use stock date range logic with global backtester from API routes
            from Strategies.Rotation_Stocks.api.stock_routes import stock_backtester, initialize_stock_backtester
            
            if stock_backtester is None:
                logger.info("Lazily initializing Stock backtester")
                initialize_stock_backtester()
                from Strategies.Rotation_Stocks.api.stock_routes import stock_backtester
            
            if stock_backtester is None:
                # Fallback if disabled
                return {
                    "start_date": "N/A",
                    "end_date": "N/A",
                    "years": 0.0,
                    "message": "Stock Strategy is disabled"
                }
            
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
            from Strategies.CustomStrategies.Rotation_ETF_Payout.api_routes import rotation_etf_payout_backtester, initialize_rotation_etf_payout_backtester
            
            if rotation_etf_payout_backtester is None:
                logger.info("Lazily initializing ETF Payout backtester")
                initialize_rotation_etf_payout_backtester()
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
                
        elif request.strategy_type == "ETF_Buy_on_Dip":
            # ETF_Buy_on_Dip uses same date range logic as ETF_Rotation
            from Strategies.Rotation_ETF.api.etf_routes import etf_backtester, initialize_etf_backtester
            
            if etf_backtester is None:
                logger.info("Lazily initializing ETF backtester for Buy-on-Dip")
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

        elif request.strategy_type == "ETF_Swing_Strategy":
             # ETF_Swing_Strategy uses same date range logic as ETF_Rotation
            from Strategies.Rotation_ETF.api.etf_routes import etf_backtester, initialize_etf_backtester
            
            if etf_backtester is None:
                logger.info("Lazily initializing ETF backtester for ETF Swing")
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

        elif request.strategy_type == "US_ETF_Swing_Strategy":
            # US_ETF_Swing_Strategy uses same date range logic as International_ETF_Rotation
            from Strategies.Rotation_International_ETF.api.routes import international_etf_backtester, initialize_international_etf_backtester

            if international_etf_backtester is None:
                logger.info("Lazily initializing International ETF backtester for US ETF Swing")
                initialize_international_etf_backtester()
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
            
    except HTTPException:
        # Re-raise HTTP exceptions to maintain proper status codes (e.g., 400)
        raise
    except Exception as e:
        logger.error(f"Error calculating date range for {request.strategy_type}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to calculate date range: {str(e)}")

async def get_cached_transaction_log(strategy_type: str) -> Dict[str, Any]:
    """Get transaction log from the latest backtest result in cache"""
    try:
        cached_results = get_cached_backtest_results(strategy_type)
        
        if not cached_results:
            return {
                "success": False,
                "message": f"No cached data for {strategy_type}. Run backtest first.",
                "transaction_log": [],
                "trading_summary": {}
            }
        
        portfolio_log = cached_results.get('portfolio_log', [])
        trading_summary = cached_results.get('trading_summary', {})
        
        # Helper to sanitize data
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
        
        return {
            "success": True,
            "strategy_type": strategy_type,
            "transaction_log": sanitized_log,
            "trading_summary": sanitized_summary,
            "total_transactions": len(sanitized_log)
        }
        
    except Exception as e:
        logger.error(f"Error getting cached transaction log: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "transaction_log": [],
            "trading_summary": {}
        }

async def get_asset_overview(strategy_type: str = None, market: str = None, asset_type: str = None) -> Dict[str, Any]:
    """Get detailed asset overview for the specified strategy type or market+asset_type"""
    try:
        # ── NEW: market + asset_type path ────────────────────────────────────
        if market and asset_type:
            market = market.upper()
            asset_type = asset_type.upper()
            logger.info(f"Getting asset overview for market={market}, asset_type={asset_type}")
            from Services.market_data_service import MarketDataService
            from Databases.app_data_db_connection import get_session
            from sqlalchemy import func
            db = get_session()
            try:
                model = MarketDataService.get_model(market, asset_type)
                rows = db.query(
                    model.symbol,
                    func.min(model.date).label('start_date'),
                    func.max(model.date).label('end_date'),
                    func.count(model.date).label('total_records')
                ).group_by(model.symbol).order_by(model.symbol).all()
                overview = []
                for row in rows:
                    try:
                        years = round((row.end_date - row.start_date).days / 365.25, 1)
                    except Exception:
                        years = 0.0
                    overview.append({
                        'symbol': row.symbol,
                        'description': MarketDataService.generate_asset_description(row.symbol, asset_type),
                        'sector': MarketDataService.get_asset_sector_classification(row.symbol, asset_type),
                        'start_date': str(row.start_date),
                        'end_date': str(row.end_date),
                        'years_available': years,
                        'total_records': row.total_records
                    })
                overview_key = "stock_overview" if asset_type == "STOCK" else "etf_overview"
                return {"success": True, "market": market, "asset_type": asset_type, overview_key: overview, "total": len(overview)}
            finally:
                db.close()

        # ── LEGACY: strategy_type path ───────────────────────────────────────
        if not strategy_type:
            raise HTTPException(status_code=400, detail="Provide either 'market'+'asset_type' or 'strategy_type'")

        logger.info(f"Getting asset overview for strategy: {strategy_type}")
        asset_overview = []

        if strategy_type == "ETF_Rotation":
            from Strategies.Rotation_ETF.api.etf_routes import etf_backtester, initialize_etf_backtester
            if etf_backtester is None:
                logger.info("Lazily initializing ETF backtester")
                initialize_etf_backtester()
                from Strategies.Rotation_ETF.api.etf_routes import etf_backtester
            
            if etf_backtester is None:
                raise HTTPException(status_code=500, detail="ETF backtester not initialized")
            
            metadata = etf_backtester.load_metadata()
            for symbol, meta in metadata.items():
                asset_overview.append({
                    'symbol': symbol,
                    'description': etf_backtester.generate_asset_description(symbol),
                    'sector': etf_backtester.get_asset_sector_classification(symbol),
                    'start_date': meta.get('start_date'),
                    'end_date': meta.get('end_date'),
                    'years_available': round(meta.get('years_available', 0), 1),
                    'total_records': meta.get('total_records', 0)
                })
            asset_overview.sort(key=lambda x: str(x['start_date']) if x['start_date'] else '9999-99-99')

        elif strategy_type == "International_ETF_Rotation":
            from Strategies.Rotation_International_ETF.api.routes import international_etf_backtester, initialize_international_etf_backtester
            if international_etf_backtester is None:
                logger.info("Lazily initializing International ETF backtester")
                initialize_international_etf_backtester()
                from Strategies.Rotation_International_ETF.api.routes import international_etf_backtester
            
            if international_etf_backtester is None:
                raise HTTPException(status_code=500, detail="International ETF backtester not initialized")
            
            metadata = international_etf_backtester.load_metadata()
            for symbol, meta in metadata.items():
                asset_overview.append({
                    'symbol': symbol,
                    'description': international_etf_backtester.generate_asset_description(symbol),
                    'sector': international_etf_backtester.get_asset_sector_classification(symbol),
                    'start_date': meta.get('start_date'),
                    'end_date': meta.get('end_date'),
                    'years_available': round(meta.get('years_available', 0), 1),
                    'total_records': meta.get('total_records', 0)
                })
            asset_overview.sort(key=lambda x: str(x['start_date']) if x['start_date'] else '9999-99-99')

        elif strategy_type == "Stock_Rotation":
            from Strategies.Rotation_Stocks.api.stock_routes import stock_backtester, initialize_stock_backtester
            if stock_backtester is None:
                logger.info("Lazily initializing Stock backtester")
                initialize_stock_backtester()
                from Strategies.Rotation_Stocks.api.stock_routes import stock_backtester

            if stock_backtester is None:
                logger.warning("Stock backtester not initialized or disabled")
                return {"asset_overview": [], "message": "Stock Strategy is currently disabled"}
            
            metadata = stock_backtester.load_metadata()
            for symbol, meta in metadata.items():
                asset_overview.append({
                    'symbol': symbol,
                    'description': stock_backtester.generate_asset_description(symbol),
                    'sector': stock_backtester.get_asset_sector_classification(symbol),
                    'start_date': meta.get('start_date'),
                    'end_date': meta.get('end_date'),
                    'years_available': round(meta.get('years_available', 0), 1),
                    'total_records': meta.get('total_records', 0)
                })
            asset_overview.sort(key=lambda x: str(x['start_date']) if x['start_date'] else '9999-99-99')

        elif strategy_type == "RS_ETF_Rotation":
            from Strategies.RS_ETF.rs_etf_backtester_core import RSETFStrategyBacktester
            from Strategies.RS_ETF.database import get_db
            db = next(get_db())
            try:
                temp_config = {'main_index': '^NSEI', 'buffer_capital_pct': 10.0, 'max_positions': 10}
                backtester = RSETFStrategyBacktester.from_config_dict(db, temp_config)
                metadata = backtester.load_metadata()
                for symbol, meta in metadata.items():
                    asset_overview.append({
                        'symbol': symbol,
                        'description': backtester.generate_asset_description(symbol),
                        'sector': backtester.get_asset_sector_classification(symbol),
                        'start_date': meta.get('start_date'),
                        'end_date': meta.get('end_date'),
                        'years_available': round(meta.get('years_available', 0), 1),
                        'total_records': meta.get('total_records', 0)
                    })
                asset_overview.sort(key=lambda x: x['symbol'])
            finally:
                db.close()

        elif strategy_type == "RS_Stocks":
            from Strategies.RS_Stocks.rs_backtester_core import RSStrategyBacktester
            from Strategies.RS_Stocks.database import get_db
            db = next(get_db())
            try:
                temp_config = {'main_index': '^NSEI', 'stock_universe': 'NIFTY_500', 'buffer_capital_pct': 10.0, 'max_positions': 20}
                backtester = RSStrategyBacktester.from_config_dict(db, temp_config)
                metadata = backtester.load_metadata()
                for symbol, meta in metadata.items():
                    asset_overview.append({
                        'symbol': symbol,
                        'description': backtester.generate_asset_description(symbol),
                        'sector': backtester.get_asset_sector_classification(symbol),
                        'start_date': meta.get('start_date'),
                        'end_date': meta.get('end_date'),
                        'years_available': round(meta.get('years_available', 0), 1),
                        'total_records': meta.get('total_records', 0)
                    })
                asset_overview.sort(key=lambda x: x['symbol'])
            finally:
                db.close()

        elif strategy_type == "ETF_Payout":
            from Strategies.CustomStrategies.Rotation_ETF_Payout.api_routes import rotation_etf_payout_backtester, initialize_rotation_etf_payout_backtester
            if rotation_etf_payout_backtester is None:
                logger.info("Lazily initializing ETF Payout backtester")
                initialize_rotation_etf_payout_backtester()
                from Strategies.CustomStrategies.Rotation_ETF_Payout.api_routes import rotation_etf_payout_backtester
            
            if rotation_etf_payout_backtester is None:
                raise HTTPException(status_code=500, detail="ETF Payout backtester not initialized")
            
            metadata = rotation_etf_payout_backtester.load_metadata()
            for symbol, meta in metadata.items():
                asset_overview.append({
                    'symbol': symbol,
                    'description': rotation_etf_payout_backtester.generate_asset_description(symbol),
                    'sector': rotation_etf_payout_backtester.get_asset_sector_classification(symbol),
                    'start_date': meta.get('start_date'),
                    'end_date': meta.get('end_date'),
                    'years_available': round(meta.get('years_available', 0), 1),
                    'total_records': meta.get('total_records', 0)
                })
            asset_overview.sort(key=lambda x: str(x['start_date']) if x['start_date'] else '9999-99-99')

        elif strategy_type == "ETF_Buy_on_Dip":
            from Strategies.Rotation_ETF.api.etf_routes import etf_backtester, initialize_etf_backtester
            if etf_backtester is None:
                logger.info("Lazily initializing ETF backtester for Buy-on-Dip")
                initialize_etf_backtester()
                from Strategies.Rotation_ETF.api.etf_routes import etf_backtester
            
            if etf_backtester is None:
                raise HTTPException(status_code=500, detail="ETF backtester not initialized")
            
            metadata = etf_backtester.load_metadata()
            for symbol, meta in metadata.items():
                asset_overview.append({
                    'symbol': symbol,
                    'description': etf_backtester.generate_asset_description(symbol),
                    'sector': etf_backtester.get_asset_sector_classification(symbol),
                    'start_date': meta.get('start_date'),
                    'end_date': meta.get('end_date'),
                    'years_available': round(meta.get('years_available', 0), 1),
                    'total_records': meta.get('total_records', 0)
                })
            asset_overview.sort(key=lambda x: str(x['start_date']) if x['start_date'] else '9999-99-99')

        elif strategy_type == "ETF_Swing_Strategy":
            from Strategies.Rotation_ETF.api.etf_routes import etf_backtester, initialize_etf_backtester
            if etf_backtester is None:
                logger.info("Lazily initializing ETF backtester for ETF Swing")
                initialize_etf_backtester()
                from Strategies.Rotation_ETF.api.etf_routes import etf_backtester
            
            if etf_backtester is None:
                raise HTTPException(status_code=500, detail="ETF backtester not initialized")
            
            metadata = etf_backtester.load_metadata()
            for symbol, meta in metadata.items():
                asset_overview.append({
                    'symbol': symbol,
                    'description': etf_backtester.generate_asset_description(symbol),
                    'sector': etf_backtester.get_asset_sector_classification(symbol),
                    'start_date': meta.get('start_date'),
                    'end_date': meta.get('end_date'),
                    'years_available': round(meta.get('years_available', 0), 1),
                    'total_records': meta.get('total_records', 0)
                })
            asset_overview.sort(key=lambda x: str(x['start_date']) if x['start_date'] else '9999-99-99')

        elif strategy_type == "US_ETF_Swing_Strategy":
            from Strategies.Rotation_International_ETF.api.routes import international_etf_backtester, initialize_international_etf_backtester
            if international_etf_backtester is None:
                logger.info("Lazily initializing International ETF backtester for US ETF Swing")
                initialize_international_etf_backtester()
                from Strategies.Rotation_International_ETF.api.routes import international_etf_backtester

            if international_etf_backtester is None:
                raise HTTPException(status_code=500, detail="International ETF backtester not initialized")

            metadata = international_etf_backtester.load_metadata()
            for symbol, meta in metadata.items():
                asset_overview.append({
                    'symbol': symbol,
                    'description': international_etf_backtester.generate_asset_description(symbol),
                    'sector': international_etf_backtester.get_asset_sector_classification(symbol),
                    'start_date': meta.get('start_date'),
                    'end_date': meta.get('end_date'),
                    'years_available': round(meta.get('years_available', 0), 1),
                    'total_records': meta.get('total_records', 0)
                })
            asset_overview.sort(key=lambda x: str(x['start_date']) if x['start_date'] else '9999-99-99')

        elif strategy_type == "SuperTrend":
            from Strategies.SuperTrend.services.backtester import SuperTrendBacktester
            backtester = SuperTrendBacktester(market=market or "INDIA", asset_type=asset_type or "STOCK")
            metadata = backtester.load_metadata()
            for symbol, meta in metadata.items():
                asset_overview.append({
                    'symbol': symbol,
                    'description': backtester.generate_asset_description(symbol),
                    'sector': backtester.get_asset_sector_classification(symbol),
                    'start_date': meta.get('start_date'),
                    'end_date': meta.get('end_date'),
                    'years_available': round(meta.get('years_available', 0), 1),
                    'total_records': meta.get('total_records', 0)
                })
            asset_overview.sort(key=lambda x: str(x['start_date']) if x['start_date'] else '9999-99-99')

        else:
             # Fallback for unknown types
             assets_data = await get_assets(strategy_type)
             items = assets_data.get("stocks") or assets_data.get("etfs") or []
             for item in items:
                 asset_overview.append({
                    'symbol': item.get('ticker') or item.get('symbol'),
                    'description': item.get('name', ''),
                    'sector': item.get('sector') or item.get('category', ''),
                    'start_date': item.get('start_date'),
                    'end_date': item.get('end_date'),
                    'years_available': round(item.get('years_available', 0), 1),
                    'total_records': item.get('total_records', 0)
                 })

        return {"asset_overview": asset_overview}

    except Exception as e:
        logger.error(f"Error getting asset overview for {strategy_type}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get asset overview: {str(e)}")
