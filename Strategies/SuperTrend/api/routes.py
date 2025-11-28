"""
API Routes for FastAPI
"""
from fastapi import APIRouter, HTTPException
from typing import List
import pandas as pd
from datetime import datetime

from .models import (
    StrategyConfig, BacktestRequest, BacktestResponse,
    EODRunResponse, CandidatesResponse, PositionsResponse,
    Candidate, Position
)
from .database import execute_query, execute_write
from ..backtester_core.backtest_engine import BacktestEngine
from ..backtester_core.indicators import add_indicators
from ..backtester_core.rs_calculator import calculate_rs_for_universe
from ..backtester_core.signal_logic import add_all_signals
from ..services.ingestion import (
    load_stock_data, load_index_data, 
    check_data_availability, get_available_symbols
)

router = APIRouter()


@router.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Supertrend + EMA + RS Strategy API",
        "version": "1.0",
        "status": "running"
    }


@router.get("/api/config", response_model=StrategyConfig)
async def get_config():
    """Get current strategy configuration"""
    try:
        df = execute_query("SELECT * FROM strategy_config ORDER BY id DESC LIMIT 1")
        
        if df.empty:
            raise HTTPException(status_code=404, detail="Configuration not found")
        
        config = df.iloc[0].to_dict()
        
        return StrategyConfig(**config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching config: {str(e)}")


@router.put("/api/config", response_model=StrategyConfig)
async def update_config(config: StrategyConfig):
    """Update strategy configuration"""
    try:
        # Delete old config
        execute_write("DELETE FROM strategy_config")
        
        # Insert new config
        query = """
            INSERT INTO strategy_config 
            (ema_short, ema_long, supertrend_period, supertrend_stop_pct, max_holdings, buffer_pct,
             price_floor, liquidity_cr, rs_window_1, rs_window_2, rs_window_3, benchmark, universe)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        params = (
            config.ema_short, config.ema_long, config.supertrend_period,
            config.supertrend_stop_pct, config.max_holdings, config.buffer_pct, config.price_floor,
            config.liquidity_cr, config.rs_window_1, config.rs_window_2,
            config.rs_window_3, config.benchmark, config.universe
        )
        
        execute_write(query, params)
        
        return config
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating config: {str(e)}")


@router.post("/api/run/backtest", response_model=BacktestResponse)
async def run_backtest(request: BacktestRequest):
    """Run full backtest"""
    try:
        # Load config
        config_df = execute_query("SELECT * FROM strategy_config ORDER BY id DESC LIMIT 1")
        if config_df.empty:
            raise HTTPException(status_code=404, detail="Configuration not found")
        
        config = config_df.iloc[0].to_dict()

        # Override configurable parameters with request values if provided
        if request.ema_short is not None:
            config['ema_short'] = request.ema_short
        if request.ema_long is not None:
            config['ema_long'] = request.ema_long
        if request.max_holdings is not None:
            config['max_holdings'] = request.max_holdings

        if config['ema_short'] >= config['ema_long']:
            raise HTTPException(status_code=400, detail="EMA short period must be less than EMA long period")
        
        # Print comprehensive backtest configuration
        print("\n" + "="*80)
        print("📋 BACKTEST CONFIGURATION")
        print("="*80)
        print(f"\n📅 DATE RANGE:")
        print(f"  Start Date: {request.start_date}")
        print(f"  End Date: {request.end_date}")
        
        print(f"\n💰 CAPITAL & COSTS:")
        print(f"  Initial Capital: ₹{request.initial_capital:,.0f}")
        print(f"  Brokerage: {request.brokerage_pct}%")
        print(f"  Buffer Reserve: {request.buffer_pct}%")
        
        print(f"\n📊 TECHNICAL INDICATORS:")
        print(f"  EMA Short: {config['ema_short']}")
        print(f"  EMA Long: {config['ema_long']}")
        print(f"  Supertrend Period: {config['supertrend_period']}")
        print(f"  Supertrend Stop: {config['supertrend_stop_pct']}%")
        
        print(f"\n🎯 PORTFOLIO SETTINGS:")
        print(f"  Max Holdings: {config['max_holdings']}")
        print(f"  Price Floor: ₹{config['price_floor']}")
        print(f"  Liquidity Minimum: ₹{config['liquidity_cr']} Cr")
        
        print(f"\n📈 RELATIVE STRENGTH (RS):")
        print(f"  RS Window 1 (Week): {config['rs_window_1']} days")
        print(f"  RS Window 2 (Month): {config['rs_window_2']} days")
        print(f"  RS Window 3 (Quarter): {config['rs_window_3']} days")
        print(f"  Benchmark Index: {config['benchmark']}")
        
        print(f"\n🎲 STOCK UNIVERSE:")
        if request.symbols:
            print(f"  Custom Selection: {len(request.symbols)} stocks")
            print(f"  Symbols: {', '.join(sorted(request.symbols))}")
        else:
            print(f"  Universe: {config['universe']}")
        
        print("="*80 + "\n")
        
        # Calculate data load start date (need 100 days before backtest for RS calculation)
        # 63 days for longest RS window + 37 days buffer for weekends/holidays
        from datetime import datetime, timedelta
        backtest_start_dt = datetime.strptime(request.start_date, '%Y-%m-%d')
        data_load_start = (backtest_start_dt - timedelta(days=100)).strftime('%Y-%m-%d')
        
        # Load data with buffer for RS calculation
        print(f"✓ Loading data from {data_load_start} to {request.end_date} (100-day buffer for RS)")
        stock_data = load_stock_data(start_date=data_load_start, end_date=request.end_date)
        index_data = load_index_data(start_date=data_load_start, end_date=request.end_date)
        
        if stock_data.empty:
            raise HTTPException(status_code=404, detail="No stock data found for date range")
        
        if index_data.empty:
            raise HTTPException(status_code=404, detail="No index data found for date range")
        
        # Filter to selected symbols if provided
        if request.symbols:
            stock_data = stock_data[stock_data['symbol'].isin(request.symbols)].copy()
            if stock_data.empty:
                raise HTTPException(status_code=404, detail="No stock data found for selected symbols in date range")
            print(f"✓ Filtered to {len(request.symbols)} selected symbols")
        
        print(f"✓ Data loaded: {len(stock_data)} stock records, {len(index_data)} index records")
        print(f"✓ Starting backtest simulation...")
        
        # Run backtest with user-defined brokerage and buffer
        engine = BacktestEngine(stock_data, index_data, config)
        results = engine.run_backtest(
            request.start_date, 
            request.end_date, 
            request.initial_capital,
            brokerage_pct=request.brokerage_pct,
            buffer_pct=request.buffer_pct
        )
        print(f"✓ Backtest completed with {results['total_trades']} trades")
        
        # Save results to database
        print(f"✓ Saving backtest results to database...")
        from Databases.app_data_db_connection import get_session
        from Databases.strategy_models import SuperTrendBacktestResult
        from sqlalchemy.dialects.postgresql import insert
        
        session = get_session()
        try:
            # Clear old results
            session.query(SuperTrendBacktestResult).delete()
            
            # Save trades
            run_date = datetime.now().strftime('%Y-%m-%d')
            for trade in results['trades']:
                backtest_result = SuperTrendBacktestResult(
                    run_date=run_date,
                    trade_date=trade['date'],
                    symbol=trade['symbol'],
                    action=trade['action'],
                    price=trade['price'],
                    quantity=trade['quantity'],
                    position_value=trade['value'],
                    portfolio_value=0,  # Portfolio value not stored per trade
                    cash=trade['cash_after']
                )
                session.add(backtest_result)
            
            session.commit()
            print(f"✓ Saved {len(results['trades'])} backtest results")
        except Exception as e:
            session.rollback()
            print(f"Error saving backtest results: {e}")
            raise
        finally:
            session.close()
        
        return BacktestResponse(
            status="success",
            message=f"Backtest completed with {results['total_trades']} trades",
            metrics=results['metrics'],
            equity_curve=results['equity_curve'],
            trades=results['trades'],
            cost_analysis=results.get('cost_analysis'),
            total_costs=results.get('total_costs', 0),
            selected_symbols=request.symbols or []
        )
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Backtest error details: {error_details}")
        raise HTTPException(status_code=500, detail=f"Backtest error: {str(e)}")


@router.post("/api/run/eod", response_model=EODRunResponse)
async def run_eod():
    """Run end-of-day signal generation"""
    try:
        # Load config
        config_df = execute_query("SELECT * FROM strategy_config ORDER BY id DESC LIMIT 1")
        if config_df.empty:
            raise HTTPException(status_code=404, detail="Configuration not found")
        
        config = config_df.iloc[0].to_dict()
        
        # Load latest data (last 2 years to optimize performance)
        from datetime import datetime, timedelta
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
        
        print(f"Loading data from {start_date} to {end_date}")
        stock_data = load_stock_data(start_date=start_date, end_date=end_date)
        index_data = load_index_data(start_date=start_date, end_date=end_date)
        
        if stock_data.empty or index_data.empty:
            raise HTTPException(status_code=404, detail="No data available")
        
        print(f"✓ Data loaded: {len(stock_data)} stock records, {len(index_data)} index records")
        
        # Get latest date
        latest_date = stock_data['date'].max()
        print(f"✓ Processing latest date: {latest_date}")
        
        # Prepare data
        symbols = stock_data['symbol'].unique()
        print(f"✓ Calculating indicators for {len(symbols)} symbols...")
        all_data = []
        
        for i, symbol in enumerate(symbols):
            if (i + 1) % 100 == 0:  # Progress every 100 symbols
                print(f"  Processing symbol {i + 1}/{len(symbols)}: {symbol}")
            symbol_df = stock_data[stock_data['symbol'] == symbol].copy()
            symbol_df = symbol_df.sort_values('date').reset_index(drop=True)
            
            # Add indicators
            symbol_df = add_indicators(
                symbol_df,
                config['ema_short'],
                config['ema_long'],
                config['supertrend_period'],
                config['supertrend_stop_pct']
            )
            
            all_data.append(symbol_df)
        
        all_stocks = pd.concat(all_data, ignore_index=True)
        print(f"✓ Indicators calculated for all symbols")
        
        # Calculate RS
        print(f"✓ Calculating Relative Strength scores...")
        print(f"  Stock data shape: {all_stocks.shape}")
        print(f"  Index data shape: {index_data.shape}")
        print(f"  RS windows: {[config['rs_window_1'], config['rs_window_2'], config['rs_window_3']]}")
        
        all_stocks = calculate_rs_for_universe(
            all_stocks, index_data,
            [config['rs_window_1'], config['rs_window_2'], config['rs_window_3']]
        )
        print(f"✓ RS scores calculated")
        
        # Debug RS scores
        rs_sample = all_stocks[['symbol', 'date', 'rs_score']].dropna().head(10)
        print(f"  RS sample data:")
        print(f"  {rs_sample}")
        
        # Check for NULL RS scores
        null_rs_count = all_stocks['rs_score'].isna().sum()
        print(f"  NULL RS scores count: {null_rs_count}")
        
        # Add signals
        print(f"✓ Generating entry/exit signals...")
        all_stocks = add_all_signals(all_stocks, config['price_floor'], config['liquidity_cr'])
        print(f"✓ Signals generated")
        
        # Get latest candidates
        latest_data = all_stocks[all_stocks['date'] == latest_date].copy()
        candidates = latest_data[
            (latest_data['entry_signal'] == 1) & 
            (latest_data['eligible'] == 1)
        ].copy()
        
        candidates = candidates.sort_values('rs_score', ascending=False, na_position='last')
        candidates['rank'] = range(1, len(candidates) + 1)
        print(f"✓ Found {len(candidates)} eligible candidates")
        
        # Save candidates to database
        print(f"✓ Saving results to database...")
        from Databases.app_data_db_connection import get_session
        from Databases.strategy_models import SuperTrendCandidate, SuperTrendCurrentPosition
        
        session = get_session()
        try:
            # Clear old candidates
            session.query(SuperTrendCandidate).delete()
            
            # Insert new candidates
            for _, row in candidates.iterrows():
                rs_score_value = row['rs_score'] if pd.notna(row['rs_score']) else None
                print(f"  Inserting {row['symbol']}: rs_score={rs_score_value}")
                
                candidate = SuperTrendCandidate(
                    symbol=row['symbol'],
                    date=row['date'],
                    adj_close=row['adj_close'],
                    ema10=row['ema10'],
                    ema20=row['ema20'],
                    supertrend=row['supertrend_color'],
                    rs_score=rs_score_value,
                    rank=row['rank'],
                    eligible=row['eligible']
                )
                session.add(candidate)
            
            session.commit()
            
            # Get current positions count
            positions_count = session.query(SuperTrendCurrentPosition).count()
        except Exception as e:
            session.rollback()
            print(f"Error saving candidates: {e}")
            raise
        finally:
            session.close()
        print(f"✓ EOD processing completed successfully!")
        
        return EODRunResponse(
            status="success",
            message=f"EOD processing completed for {latest_date}",
            run_date=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            candidates_count=len(candidates),
            positions_count=positions_count
        )
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"EOD error details: {error_details}")
        raise HTTPException(status_code=500, detail=f"EOD processing error: {str(e)}")


@router.get("/api/candidates", response_model=CandidatesResponse)
async def get_candidates():
    """Get current candidate stocks"""
    try:
        df = execute_query("SELECT * FROM candidates ORDER BY rank")
        
        candidates = []
        for _, row in df.iterrows():
            candidates.append(Candidate(
                symbol=row['symbol'],
                date=row['date'],
                adj_close=row['adj_close'],
                ema10=row['ema10'],
                ema20=row['ema20'],
                supertrend=row['supertrend'],
                rs_score=row['rs_score'],
                rank=row['rank'],
                eligible=row['eligible']
            ))
        
        return CandidatesResponse(
            status="success",
            candidates=candidates
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching candidates: {str(e)}")


@router.get("/api/positions", response_model=PositionsResponse)
async def get_positions():
    """Get current positions"""
    try:
        df = execute_query("SELECT * FROM current_positions")
        
        positions = []
        total_value = 0
        total_pnl = 0
        
        for _, row in df.iterrows():
            positions.append(Position(
                symbol=row['symbol'],
                entry_date=row['entry_date'],
                entry_price=row['entry_price'],
                quantity=row['quantity'],
                current_price=row['current_price'],
                current_value=row['current_value'],
                pnl=row['pnl'],
                pnl_pct=row['pnl_pct']
            ))
            total_value += row['current_value']
            total_pnl += row['pnl']
        
        total_pnl_pct = (total_pnl / (total_value - total_pnl)) * 100 if total_value > 0 else 0
        
        return PositionsResponse(
            status="success",
            positions=positions,
            total_value=total_value,
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching positions: {str(e)}")


@router.get("/api/data/availability")
async def get_data_availability():
    """Get data availability statistics"""
    try:
        stats = check_data_availability()
        return {
            "status": "success",
            **stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking data: {str(e)}")


@router.get("/api/symbols")
async def get_symbols():
    """Get list of available symbols"""
    try:
        symbols = get_available_symbols()
        return {
            "status": "success",
            "symbols": symbols,
            "count": len(symbols)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching symbols: {str(e)}")

